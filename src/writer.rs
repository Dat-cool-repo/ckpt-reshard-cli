//! Streaming writers: single safetensors file, or HF-style sharded safetensors + index.json.

use crate::cast::{CastSpec, cast_stream};
use crate::ckpt::{Checkpoint, Format};
use crate::dtype::{DType, human_bytes, numel};
use crate::safetensors::{OutTensor, StWriter};
use anyhow::{Context, Result, bail};
use std::collections::BTreeMap;
use std::fs::File;
use std::io::BufWriter;
use std::path::{Path, PathBuf};

/// `*` / `?` wildcard match.
pub fn glob_match(pat: &str, s: &str) -> bool {
    let p: Vec<char> = pat.chars().collect();
    let t: Vec<char> = s.chars().collect();
    let (mut pi, mut ti, mut star, mut mark) = (0usize, 0usize, None::<usize>, 0usize);
    while ti < t.len() {
        if pi < p.len() && (p[pi] == '?' || p[pi] == t[ti]) {
            pi += 1;
            ti += 1;
        } else if pi < p.len() && p[pi] == '*' {
            star = Some(pi);
            mark = ti;
            pi += 1;
        } else if let Some(sp) = star {
            pi = sp + 1;
            mark += 1;
            ti = mark;
        } else {
            return false;
        }
    }
    while pi < p.len() && p[pi] == '*' {
        pi += 1;
    }
    pi == p.len()
}

#[derive(Clone, Debug, Default)]
pub struct SelectOpts {
    pub include: Vec<String>,
    pub exclude: Vec<String>,
    pub strip_prefix: Option<String>,
    pub add_prefix: Option<String>,
}

/// One output tensor: index into ck.tensors + output name + output dtype (`--dtype`).
#[derive(Clone, Debug)]
pub struct Sel {
    pub idx: usize,
    pub name: String,
    pub dtype: DType,
}

/// Apply `--dtype` (floating-point tensors only, minus `--keep-dtype` globs).
pub fn apply_cast(ck: &Checkpoint, sel: &mut [Sel], cast: &Option<CastSpec>) {
    for s in sel {
        s.dtype = CastSpec::target(cast, &s.name, ck.tensors[s.idx].dtype);
    }
}

pub fn select(ck: &Checkpoint, o: &SelectOpts) -> Result<Vec<Sel>> {
    let mut out = Vec::new();
    let mut seen = std::collections::HashSet::new();
    for (i, t) in ck.tensors.iter().enumerate() {
        if !o.include.is_empty() && !o.include.iter().any(|p| glob_match(p, &t.name)) {
            continue;
        }
        if o.exclude.iter().any(|p| glob_match(p, &t.name)) {
            continue;
        }
        let mut name = t.name.clone();
        if let Some(sp) = &o.strip_prefix
            && let Some(r) = name.strip_prefix(sp.as_str())
        {
            name = r.to_string();
        }
        if let Some(ap) = &o.add_prefix {
            name = format!("{ap}{name}");
        }
        if !seen.insert(name.clone()) {
            bail!("output name collision: {name}");
        }
        out.push(Sel {
            idx: i,
            name,
            dtype: t.dtype,
        });
    }
    if out.is_empty() {
        bail!("no tensors selected");
    }
    Ok(out)
}

#[derive(Debug, Default, serde::Serialize)]
pub struct WriteStats {
    pub files: Vec<String>,
    pub tensors: usize,
    pub bytes: u64,
}

fn tmp_path(p: &Path) -> PathBuf {
    let mut s = p.as_os_str().to_owned();
    s.push(".tmp-ckpt");
    PathBuf::from(s)
}

/// Produces the bytes of output tensor `i` (row-major), possibly in several pieces.
pub type Producer<'a> = dyn Fn(usize, &mut dyn FnMut(&[u8]) -> Result<()>) -> Result<()> + 'a;

/// Write tensors `ids` of `decl` into one safetensors file (atomically via a temp file).
pub fn write_file_with(
    decl: &[OutTensor],
    ids: &[usize],
    path: &Path,
    metadata: &BTreeMap<String, String>,
    produce: &Producer,
) -> Result<u64> {
    let part: Vec<OutTensor> = ids.iter().map(|&i| decl[i].clone()).collect();
    let tmp = tmp_path(path);
    let f = File::create(&tmp).with_context(|| format!("creating {}", tmp.display()))?;
    let mut w = StWriter::new(BufWriter::with_capacity(8 << 20, f), &part, metadata)?;
    let mut bytes = 0;
    for &i in ids {
        produce(i, &mut |b| w.write_part(b))
            .with_context(|| format!("writing {}", decl[i].name))?;
        w.end_tensor()?;
        bytes += decl[i].nbytes();
    }
    let bw = w.finish()?;
    let f = bw.into_inner().map_err(|e| e.into_error())?;
    drop(f);
    std::fs::rename(&tmp, path)?;
    Ok(bytes)
}

fn pt_metadata() -> BTreeMap<String, String> {
    let mut m = BTreeMap::new();
    m.insert("format".to_string(), "pt".to_string());
    m
}

pub fn ensure_parent(path: &Path) -> Result<()> {
    if let Some(parent) = path.parent()
        && !parent.as_os_str().is_empty()
    {
        std::fs::create_dir_all(parent)?;
    }
    Ok(())
}

pub fn write_single_with(
    decl: &[OutTensor],
    path: &Path,
    metadata: BTreeMap<String, String>,
    produce: &Producer,
) -> Result<WriteStats> {
    ensure_parent(path)?;
    let mut md = metadata;
    md.entry("format".into()).or_insert_with(|| "pt".into());
    let ids: Vec<usize> = (0..decl.len()).collect();
    let bytes = write_file_with(decl, &ids, path, &md, produce)?;
    eprintln!(
        "wrote {} tensors, {} to {}",
        decl.len(),
        human_bytes(bytes),
        path.display()
    );
    Ok(WriteStats {
        files: vec![path.display().to_string()],
        tensors: decl.len(),
        bytes,
    })
}

#[derive(Clone, Copy, Debug)]
pub enum ShardSpec {
    MaxSize(u64),
    Count(usize),
}

/// Partition tensors (keeping order) into shards.
pub fn plan_shards(sizes: &[u64], spec: ShardSpec) -> Vec<Vec<usize>> {
    let mut shards: Vec<Vec<usize>> = Vec::new();
    match spec {
        ShardSpec::MaxSize(max) => {
            let mut cur = 0u64;
            for (i, &s) in sizes.iter().enumerate() {
                if shards.is_empty() || (cur > 0 && cur + s > max) {
                    shards.push(vec![]);
                    cur = 0;
                }
                shards.last_mut().unwrap().push(i);
                cur += s;
            }
        }
        ShardSpec::Count(n) => {
            let n = n.max(1);
            let total: u64 = sizes.iter().sum::<u64>().max(1);
            shards = vec![vec![]; n];
            let mut cum = 0u64;
            for (i, &s) in sizes.iter().enumerate() {
                let mid = cum as u128 * 2 + s as u128; // 2*(cum + s/2)
                let k = ((mid * n as u128) / (2 * total as u128)).min(n as u128 - 1) as usize;
                shards[k].push(i);
                cum += s;
            }
            shards.retain(|s| !s.is_empty());
        }
    }
    shards
}

/// Write HF-style sharded safetensors + `<prefix>.safetensors.index.json` into `dir`.
pub fn write_sharded_with(
    decl: &[OutTensor],
    dir: &Path,
    spec: ShardSpec,
    prefix: &str,
    produce: &Producer,
) -> Result<WriteStats> {
    std::fs::create_dir_all(dir)?;
    let sizes: Vec<u64> = decl.iter().map(|t| t.nbytes()).collect();
    let plan = plan_shards(&sizes, spec);
    if let ShardSpec::Count(n) = spec
        && plan.len() != n
    {
        eprintln!(
            "note: produced {} shards instead of {n} (tensors are too large to balance)",
            plan.len()
        );
    }
    let n = plan.len();
    let mut stats = WriteStats::default();
    let mut weight_map = BTreeMap::new();
    let md = pt_metadata();
    for (si, ids) in plan.iter().enumerate() {
        let fname = if n == 1 {
            format!("{prefix}.safetensors")
        } else {
            format!("{prefix}-{:05}-of-{:05}.safetensors", si + 1, n)
        };
        let b = write_file_with(decl, ids, &dir.join(&fname), &md, produce)?;
        for &i in ids {
            weight_map.insert(decl[i].name.clone(), fname.clone());
        }
        stats.bytes += b;
        stats.files.push(dir.join(&fname).display().to_string());
    }
    stats.tensors = decl.len();
    if n > 1 {
        let total_params: u64 = decl.iter().map(|t| numel(&t.shape)).sum();
        let idx = serde_json::json!({
            "metadata": {"total_size": stats.bytes, "total_parameters": total_params},
            "weight_map": weight_map,
        });
        let ip = dir.join(format!("{prefix}.safetensors.index.json"));
        std::fs::write(
            &ip,
            serde_json::to_string_pretty(&idx)?
                + "
",
        )?;
        stats.files.push(ip.display().to_string());
    }
    eprintln!(
        "wrote {} tensors, {} in {} shard(s) to {}",
        stats.tensors,
        human_bytes(stats.bytes),
        n,
        dir.display()
    );
    Ok(stats)
}

/// Declarations for a selection of checkpoint tensors.
pub fn decl_of(ck: &Checkpoint, sel: &[Sel]) -> Vec<OutTensor> {
    sel.iter()
        .map(|s| OutTensor {
            name: s.name.clone(),
            dtype: s.dtype,
            shape: ck.tensors[s.idx].shape.clone(),
        })
        .collect()
}

/// Stream selected tensor `s` (converted to its output dtype) into `w`.
pub fn produce_sel(ck: &Checkpoint, s: &Sel, w: &mut dyn FnMut(&[u8]) -> Result<()>) -> Result<()> {
    let t = &ck.tensors[s.idx];
    cast_stream(t.dtype, s.dtype, w, &mut |w2| ck.stream(t, w2))
}

pub fn write_single(ck: &Checkpoint, sel: &[Sel], path: &Path) -> Result<WriteStats> {
    let md = if ck.format == Format::Safetensors {
        ck.metadata.clone()
    } else {
        BTreeMap::new()
    };
    let decl = decl_of(ck, sel);
    write_single_with(&decl, path, md, &|i, w| produce_sel(ck, &sel[i], w))
}

pub fn write_sharded(
    ck: &Checkpoint,
    sel: &[Sel],
    dir: &Path,
    spec: ShardSpec,
    prefix: &str,
) -> Result<WriteStats> {
    let decl = decl_of(ck, sel);
    write_sharded_with(&decl, dir, spec, prefix, &|i, w| {
        produce_sel(ck, &sel[i], w)
    })
}

/// Copy non-weight files (config.json, tokenizer files, ...) from an HF directory.
pub fn copy_aux_files(src_root: &Path, dst: &Path) -> Result<Vec<String>> {
    let mut copied = Vec::new();
    if !src_root.is_dir() || src_root == dst {
        return Ok(copied);
    }
    for e in std::fs::read_dir(src_root)? {
        let p = e?.path();
        if !p.is_file() {
            continue;
        }
        let n = p.file_name().unwrap().to_string_lossy().to_string();
        if n.ends_with(".safetensors")
            || n.ends_with(".safetensors.index.json")
            || n.starts_with('.')
            || n.ends_with(".tmp-ckpt")
        {
            continue;
        }
        let d = dst.join(&n);
        if !d.exists() {
            std::fs::copy(&p, &d)?;
            copied.push(n);
        }
    }
    Ok(copied)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn globbing() {
        assert!(glob_match("model.*", "model.a.b"));
        assert!(glob_match(
            "*.attn.c_attn.weight",
            "transformer.h.0.attn.c_attn.weight"
        ));
        assert!(!glob_match(
            "*.attn.c_attn.weight",
            "transformer.h.0.attn.c_attn.bias"
        ));
        assert!(glob_match("h.?.x", "h.1.x"));
        assert!(glob_match("*", ""));
    }
    #[test]
    fn shards() {
        assert_eq!(
            plan_shards(&[5, 5, 5, 5], ShardSpec::MaxSize(10)),
            vec![vec![0, 1], vec![2, 3]]
        );
        assert_eq!(
            plan_shards(&[50, 5], ShardSpec::MaxSize(10)),
            vec![vec![0], vec![1]]
        );
        assert_eq!(
            plan_shards(&[5, 5, 5, 5], ShardSpec::Count(2)),
            vec![vec![0, 1], vec![2, 3]]
        );
        assert_eq!(plan_shards(&[1, 1, 1], ShardSpec::Count(3)).len(), 3);
    }
}
