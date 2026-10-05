//! DeepSpeed ZeRO checkpoints: `<dir>/latest` -> `<dir>/<tag>/` containing
//!   `mp_rank_00_model_states.pt` (stage 1/2) or `zero_pp_rank_R_mp_rank_00_model_states.pt` (stage 3)
//!   and `[bf16_]zero_pp_rank_R_mp_rank_00_optim_states.pt` (one per data-parallel rank).
//!
//! The fp32 master weights are rebuilt exactly as DeepSpeed's `zero_to_fp32.py` does:
//! * stage 1/2: per param group, the ranks' `single_partition_of_fp32_groups[g]` concatenated form one
//!   flat vector. The params of `param_shapes[g]` follow each other in it (plus optional
//!   `param_alignment_paddings`). The consumed size must match the available size modulo the
//!   2*world_size alignment.
//! * stage 3: each rank's `fp32_flat_groups` are concatenated. Every param occupies
//!   `ceil(numel / world_size)` elements at the same offset on every rank. The full param is the
//!   ranks' slices concatenated and cut to `numel`.
//! * buffers come from the model states `module`, frozen params from `frozen_param_fragments`, and
//!   shared (tied) params are emitted again under their alias name.
//!
//! No tensor data is copied while opening: every output tensor is a list of views into the
//! optimizer-state files, read (and written out) one tensor at a time.
//! Difference to zero_to_fp32: buffers keep their stored dtype. zero_to_fp32 applies `.float()` to
//! them; use `--dtype fp32` for the same result on floating-point buffers.

use crate::ckpt::{Checkpoint, Format, Loc, Place, TensorInfo, ViewPart, mmap_file};
use crate::pickle::{Allow, Node, Value};
use crate::torchsave::{TorchSave, TsTensor, key_str};
use anyhow::{Context, Result, anyhow, bail};
use serde_json::json;
use std::path::{Path, PathBuf};

pub fn latest_target(dir: &Path) -> Result<Option<PathBuf>> {
    let f = dir.join("latest");
    if !f.is_file() {
        return Ok(None);
    }
    let tag = std::fs::read_to_string(&f)?.trim().to_string();
    if tag.is_empty() || tag.contains('/') || tag.contains('\\') || tag.contains("..") {
        bail!("{}: suspicious tag {tag:?}", f.display());
    }
    let sub = dir.join(&tag);
    if !sub.is_dir() {
        bail!("{} points to missing {}", f.display(), sub.display());
    }
    Ok(Some(sub))
}

fn files_ending(dir: &Path, suffix: &str) -> Result<Vec<PathBuf>> {
    let mut v: Vec<PathBuf> = std::fs::read_dir(dir)?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| {
            p.is_file()
                && p.file_name()
                    .and_then(|n| n.to_str())
                    .is_some_and(|n| n.ends_with(suffix))
        })
        .collect();
    v.sort_by_key(|p| natural_key(&p.file_name().unwrap().to_string_lossy()));
    Ok(v)
}

/// zero_to_fp32's natural_keys: digit runs compare numerically.
fn natural_key(s: &str) -> Vec<(String, u64)> {
    let mut out = Vec::new();
    let mut cur = String::new();
    let mut digits = false;
    for c in s.chars() {
        if c.is_ascii_digit() != digits && !cur.is_empty() {
            out.push(if digits {
                (String::new(), cur.parse().unwrap_or(u64::MAX))
            } else {
                (cur.clone(), 0)
            });
            cur.clear();
        }
        digits = c.is_ascii_digit();
        cur.push(c);
    }
    if !cur.is_empty() {
        out.push(if digits {
            (String::new(), cur.parse().unwrap_or(u64::MAX))
        } else {
            (cur, 0)
        });
    }
    out
}

pub fn is_ds_dir(dir: &Path) -> Result<bool> {
    Ok(!files_ending(dir, "_model_states.pt")?.is_empty())
}

struct ModelStates {
    file: usize,
    ts: TorchSave,
}

/// (name, shape) list of an OrderedDict name -> torch.Size.
fn shapes_of(ts: &TorchSave, v: &Value) -> Result<Vec<(String, Vec<u64>)>> {
    let pk = &ts.pickle;
    let d = pk
        .dict(v)
        .ok_or_else(|| anyhow!("param_shapes entry is not a dict"))?;
    d.iter()
        .map(|(k, s)| {
            let name = key_str(pk, k);
            let shape = pk
                .int_list(s)
                .ok_or_else(|| anyhow!("{name}: bad shape"))?
                .into_iter()
                .map(|x| u64::try_from(x).map_err(|_| anyhow!("{name}: negative dim")))
                .collect::<Result<Vec<u64>>>()?;
            // the fp32 partitions are F32; the shape must be a sane F32 tensor
            crate::dtype::checked_nbytes(&shape, crate::dtype::DType::F32)
                .with_context(|| name.clone())?;
            Ok((name, shape))
        })
        .collect()
}

fn tensor_list(ts: &TorchSave, v: &Value, what: &str) -> Result<Vec<TsTensor>> {
    let pk = &ts.pickle;
    let items = pk.seq(v).ok_or_else(|| anyhow!("{what} is not a list"))?;
    items
        .iter()
        .enumerate()
        .map(|(i, x)| {
            ts.tensor_at(x, &format!("{what}.{i}"))?
                .ok_or_else(|| anyhow!("{what}[{i}] is not a tensor"))
        })
        .collect()
}

fn as_int_or_enum(ts: &TorchSave, v: &Value) -> Option<i64> {
    let pk = &ts.pickle;
    pk.as_int(v).or_else(|| {
        // ZeroStageEnum(2): an enum constructor call with the int value
        match pk.node(v)? {
            Node::Object { args, .. } => args.first().and_then(|a| pk.as_int(a)),
            _ => None,
        }
    })
}

fn flat_part(file: usize, t: &TsTensor, start: u64, len: u64, at: u64) -> Result<ViewPart> {
    if t.sizes.len() != 1 {
        bail!("{}: flat partition is not 1-D", t.path);
    }
    if start.checked_add(len).is_none_or(|e| e > t.sizes[0]) {
        bail!("{}: flat partition too short", t.path);
    }
    Ok(ViewPart::from_ts(file, t, Place::Linear(at)).narrow(0, start, len))
}

pub fn open(dir: &Path) -> Result<Checkpoint> {
    let model_files = files_ending(dir, "_model_states.pt")?;
    let optim_files = files_ending(dir, "_optim_states.pt")?;
    if model_files.is_empty() {
        bail!("{}: no *_model_states.pt", dir.display());
    }
    if optim_files.is_empty() {
        bail!(
            "{}: no *_optim_states.pt; without ZeRO optimizer partitions use `ckpt inspect --raw <file>`",
            dir.display()
        );
    }
    let mut ck = Checkpoint::new(Format::DeepSpeed, dir);
    let mut models = Vec::new();
    for f in &model_files {
        let mf = mmap_file(f)?;
        let ts = TorchSave::open(&mf.map, Allow::DeepSpeed)
            .with_context(|| format!("reading {}", f.display()))?;
        ck.files.push(mf);
        models.push(ModelStates {
            file: ck.files.len() - 1,
            ts,
        });
    }
    let mut optims = Vec::new();
    for f in &optim_files {
        let mf = mmap_file(f)?;
        let ts = TorchSave::open(&mf.map, Allow::DeepSpeed)
            .with_context(|| format!("reading {}", f.display()))?;
        ck.files.push(mf);
        optims.push((ck.files.len() - 1, ts));
    }
    let m0 = &models[0].ts;
    let pk0 = &m0.pickle;
    if m0.root_get("buffer_names").is_none() {
        bail!(
            "{} is not a DeepSpeed model-states file",
            model_files[0].display()
        );
    }
    // ---- optimizer metadata
    let osd = |ts: &TorchSave| -> Result<Value> {
        ts.root_get("optimizer_state_dict")
            .cloned()
            .ok_or_else(|| anyhow!("optim states without optimizer_state_dict"))
    };
    let (of0, o0) = &optims[0];
    let _ = of0;
    let osd0 = osd(o0)?;
    let get0 = |k: &str| o0.pickle.dict_get(&osd0, k);
    let stage = get0("zero_stage")
        .and_then(|v| as_int_or_enum(o0, v))
        .ok_or_else(|| {
            anyhow!(
                "{} is not a ZeRO checkpoint (no zero_stage)",
                optim_files[0].display()
            )
        })?;
    let ws = match get0("partition_count") {
        Some(v) => match o0.pickle.seq(v) {
            Some(l) => l
                .iter()
                .filter_map(|x| o0.pickle.as_int(x))
                .max()
                .unwrap_or(0),
            None => o0.pickle.as_int(v).unwrap_or(0),
        },
        None => bail!("no partition_count"),
    } as usize;
    if ws != optims.len() {
        bail!(
            "expected {ws} *_optim_states.pt files (partition_count) but found {} -- incomplete checkpoint?",
            optims.len()
        );
    }
    let key = match stage {
        1 | 2 => "single_partition_of_fp32_groups",
        3 => "fp32_flat_groups",
        s => bail!("unknown ZeRO stage {s}"),
    };
    let mut flat: Vec<(usize, Vec<TsTensor>)> = Vec::new();
    for (f, ts) in &optims {
        let o = osd(ts)?;
        let v = ts
            .pickle
            .dict_get(&o, key)
            .ok_or_else(|| anyhow!("optimizer_state_dict has no {key}"))?;
        let list = tensor_list(ts, v, key)?;
        if crate::zipread::verify_enabled() {
            for t in &list {
                ts.verify_storage(&ck.files[*f].map, &t.storage_key)?;
            }
        }
        flat.push((*f, list));
    }
    // every rank must hold the same param groups; stage 3 partitions are equal-sized per group
    for (r, (_, l)) in flat.iter().enumerate() {
        if l.len() != flat[0].1.len() {
            bail!(
                "rank {r} has {} fp32 param groups but rank 0 has {}",
                l.len(),
                flat[0].1.len()
            );
        }
        if stage == 3 {
            for (g, t) in l.iter().enumerate() {
                if t.numel() != flat[0].1[g].numel() {
                    bail!("ZeRO-3 group {g}: rank {r} partition size differs from rank 0");
                }
            }
        }
    }
    let paddings: Option<Vec<Vec<u64>>> = match get0("param_alignment_paddings") {
        Some(v) if !matches!(v, Value::None) => {
            let groups = o0
                .pickle
                .seq(v)
                .ok_or_else(|| anyhow!("bad param_alignment_paddings"))?;
            Some(
                groups
                    .iter()
                    .map(|g| {
                        o0.pickle
                            .seq(g)
                            .map(|x| {
                                x.iter()
                                    .map(|y| {
                                        o0.pickle
                                            .as_int(y)
                                            .and_then(|v| u64::try_from(v).ok())
                                            .unwrap_or(0)
                                    })
                                    .collect()
                            })
                            .unwrap_or_default()
                    })
                    .collect(),
            )
        }
        _ => None,
    };
    // ---- model metadata
    let param_shapes: Vec<Vec<(String, Vec<u64>)>> = pk0
        .seq(
            m0.root_get("param_shapes")
                .ok_or_else(|| anyhow!("no param_shapes"))?,
        )
        .ok_or_else(|| anyhow!("param_shapes is not a list"))?
        .iter()
        .map(|g| shapes_of(m0, g))
        .collect::<Result<_>>()?;
    let buffer_names: Vec<String> = m0
        .root_get("buffer_names")
        .and_then(|v| pk0.seq(v))
        .map(|l| {
            l.iter()
                .filter_map(|x| pk0.as_str(x).map(|s| s.to_string()))
                .collect()
        })
        .unwrap_or_default();
    let dtype_of = |list: &[TsTensor]| list.first().map(|t| t.dtype);
    let fp32 = dtype_of(&flat[0].1).ok_or_else(|| anyhow!("empty fp32 partitions"))?;

    // buffers (from rank 0's module state dict)
    if let Some(module) = m0.root_get("module") {
        for (k, v) in pk0.dict(module).unwrap_or(&[]) {
            let name = key_str(pk0, k);
            if !buffer_names.contains(&name) {
                continue;
            }
            if let Some(t) = m0.tensor_at(v, &name)? {
                let zeros = vec![0; t.sizes.len()];
                ck.tensors.push(TensorInfo {
                    name,
                    dtype: t.dtype,
                    shape: t.sizes.clone(),
                    loc: Loc::Views {
                        parts: vec![ViewPart::from_ts(models[0].file, &t, Place::Block(zeros))],
                    },
                });
            }
        }
    }
    // frozen params
    if let Some(fs) = m0.root_get("frozen_param_shapes")
        && !matches!(fs, Value::None)
    {
        let shapes = shapes_of(m0, fs)?;
        for (name, shape) in shapes {
            let n: u64 = crate::dtype::numel(&shape);
            let mut parts = Vec::new();
            let mut fdtype = None;
            let frag = |ms: &ModelStates| -> Result<TsTensor> {
                let d = ms
                    .ts
                    .root_get("frozen_param_fragments")
                    .ok_or_else(|| anyhow!("no frozen_param_fragments"))?;
                let v = ms
                    .ts
                    .pickle
                    .dict_get(d, &name)
                    .ok_or_else(|| anyhow!("frozen fragment {name} missing"))?;
                ms.ts
                    .tensor_at(v, &name)?
                    .ok_or_else(|| anyhow!("frozen fragment {name} is not a tensor"))
            };
            if stage <= 2 {
                let t = frag(&models[0])?;
                fdtype = Some(t.dtype);
                if t.numel() != n {
                    bail!(
                        "frozen param {name}: fragment has {} elements, shape needs {n}",
                        t.numel()
                    );
                }
                let mut v = ViewPart::from_ts(models[0].file, &t, Place::Linear(0));
                if t.sizes.len() != 1 {
                    v.place = Place::Block(vec![0; t.sizes.len()]);
                }
                parts.push(v);
                let shape_ok = t.sizes == shape;
                if !shape_ok && t.sizes.len() != 1 {
                    bail!(
                        "frozen param {name}: fragment shape {:?} != {:?}",
                        t.sizes,
                        shape
                    );
                }
            } else {
                if models.len() != ws {
                    bail!(
                        "stage 3 needs {ws} model-states files, found {}",
                        models.len()
                    );
                }
                let mut at = 0u64;
                for ms in &models {
                    let t = frag(ms)?;
                    if fdtype.is_some_and(|d| d != t.dtype) {
                        bail!("frozen param {name}: fragments have different dtypes");
                    }
                    fdtype = Some(t.dtype);
                    let len = t.numel().min(n.saturating_sub(at));
                    if len > 0 {
                        let flat_t = flatten_ts(&t);
                        parts.push(flat_part(ms.file, &flat_t, 0, len, at)?);
                    }
                    at = at.saturating_add(len);
                }
                if at != n {
                    bail!("frozen param {name}: fragments hold {at} of {n} elements");
                }
            }
            ck.tensors.push(TensorInfo {
                name,
                dtype: fdtype.unwrap_or(fp32),
                shape,
                loc: Loc::Views { parts },
            });
        }
    }
    // trainable params
    let mut total = 0u64;
    if stage <= 2 {
        if param_shapes.len() != flat[0].1.len() {
            bail!(
                "{} param groups in param_shapes but {} fp32 partitions",
                param_shapes.len(),
                flat[0].1.len()
            );
        }
        for (g, shapes) in param_shapes.iter().enumerate() {
            // rank r's partition covers [starts[r], starts[r] + len_r) of the merged group vector
            let mut starts = Vec::new();
            let mut avail = 0u64;
            for (_, parts) in &flat {
                starts.push(avail);
                avail = avail.saturating_add(parts[g].numel());
            }
            let mut offset = 0u64;
            for (pi, (name, shape)) in shapes.iter().enumerate() {
                let n: u64 = crate::dtype::numel(shape);
                if offset.checked_add(n).is_none_or(|e| e > avail) {
                    bail!("param {name} runs past the end of group {g}'s fp32 partitions");
                }
                let mut parts = Vec::new();
                for (r, (f, ranks)) in flat.iter().enumerate() {
                    let t = &ranks[g];
                    let (lo, hi) = (
                        offset.max(starts[r]),
                        (offset + n).min(starts[r].saturating_add(t.numel())),
                    );
                    if lo < hi {
                        parts.push(flat_part(*f, t, lo - starts[r], hi - lo, lo - offset)?);
                    }
                }
                ck.tensors.push(TensorInfo {
                    name: name.clone(),
                    dtype: fp32,
                    shape: shape.clone(),
                    loc: Loc::Views { parts },
                });
                offset += n;
                total = total.saturating_add(n);
                if let Some(p) = &paddings {
                    offset = offset
                        .saturating_add(p.get(g).and_then(|x| x.get(pi)).copied().unwrap_or(0));
                }
            }
            let align = 2 * ws as u64;
            let a = |x: u64| x.div_ceil(align).saturating_mul(align);
            if a(offset) != a(avail) {
                bail!(
                    "group {g}: consumed {offset} numels out of {avail} -- param_shapes do not match the partitions"
                );
            }
        }
    } else {
        // per-rank flat vector = concatenation of its groups
        let group_starts: Vec<u64> = {
            let mut v = vec![0u64];
            for t in &flat[0].1 {
                v.push(v.last().unwrap().saturating_add(t.numel()));
            }
            v
        };
        let avail = group_starts.last().unwrap().saturating_mul(ws as u64);
        let mut offset = 0u64;
        for (name, shape) in param_shapes.iter().flatten() {
            let n: u64 = crate::dtype::numel(shape);
            let pn = n.div_ceil(ws as u64);
            if offset
                .checked_add(pn)
                .is_none_or(|e| e > *group_starts.last().unwrap())
            {
                bail!("param {name} runs past the fp32 partitions");
            }
            let mut parts = Vec::new();
            for (r, (f, groups)) in flat.iter().enumerate() {
                // elements [r*pn, (r+1)*pn) of the param live at [offset, offset+pn) of rank r's flat
                let want = pn.min(n.saturating_sub(r as u64 * pn));
                let mut got = 0u64;
                while got < want {
                    let pos = offset + got;
                    let g = group_starts
                        .windows(2)
                        .position(|w| w[0] <= pos && pos < w[1])
                        .ok_or_else(|| anyhow!("param {name} runs past the fp32 partitions"))?;
                    let in_g = pos - group_starts[g];
                    // groups[g] has the same size as rank 0's (checked above), so len >= 1
                    let len = (want - got).min(groups[g].numel() - in_g);
                    parts.push(flat_part(*f, &groups[g], in_g, len, r as u64 * pn + got)?);
                    got += len;
                }
            }
            ck.tensors.push(TensorInfo {
                name: name.clone(),
                dtype: fp32,
                shape: shape.clone(),
                loc: Loc::Views { parts },
            });
            offset += pn;
            total = total.saturating_add(n);
        }
        if offset.saturating_mul(ws as u64) != avail {
            bail!(
                "consumed {} numels out of {avail} -- param_shapes do not match the partitions",
                offset.saturating_mul(ws as u64)
            );
        }
    }
    // shared (tied) params: alias -> source
    if let Some(sp) = m0.root_get("shared_params") {
        for (k, v) in pk0.dict(sp).unwrap_or(&[]) {
            let (alias, src) = (key_str(pk0, k), pk0.as_str(v).unwrap_or("").to_string());
            if let Some(t) = ck.tensors.iter().find(|t| t.name == src).cloned() {
                if let Some(existing) = ck.tensors.iter_mut().find(|t| t.name == alias) {
                    *existing = TensorInfo { name: alias, ..t };
                } else {
                    ck.tensors.push(TensorInfo { name: alias, ..t });
                }
            }
        }
    }
    ck.info.insert("zero_stage".into(), json!(stage));
    ck.info.insert("world_size".into(), json!(ws));
    ck.info
        .insert("param_groups".into(), json!(flat[0].1.len()));
    ck.info.insert("reconstructed_params".into(), json!(total));
    if let Some(v) = m0.root_get("ds_version") {
        ck.info.insert("ds_version".into(), pk0.to_json(v));
    }
    for k in [
        "global_steps",
        "global_samples",
        "dp_world_size",
        "mp_world_size",
    ] {
        if let Some(v) = m0.root_get(k) {
            ck.info.insert(k.into(), pk0.to_json(v));
        }
    }
    let _ = models;
    Ok(ck)
}

/// A contiguous tensor seen as 1-D (frozen fragments of stage 3 are flat already).
fn flatten_ts(t: &TsTensor) -> TsTensor {
    let mut f = t.clone();
    f.sizes = vec![t.numel()];
    f.strides = vec![1];
    f
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn natural_order() {
        assert!(natural_key("zero_pp_rank_2_mp") < natural_key("zero_pp_rank_10_mp"));
    }
}
