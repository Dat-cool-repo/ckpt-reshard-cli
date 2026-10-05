//! Megatron-LM checkpoints.
//!
//! * **legacy / `torch` format**: `<save>/latest_checkpointed_iteration.txt` and
//!   `<save>/iter_XXXXXXX/mp_rank_TT[_PPP]/model_optim_rng.pt`. Each file is a `torch.save` dict
//!   `{args, checkpoint_version, iteration, model, optimizer, rng_state, ...}`, read by the restricted
//!   pickle VM with the `Allow::Megatron` list. The TP shards are merged with built-in name rules
//!   (column-parallel = dim 0, row-parallel = dim 1, SwiGLU fc1 = per-rank `[gate_r; up_r]` blocks,
//!   everything else replicated). The PP stages' local layer numbers are shifted to global ones.
//!   Nothing is copied: every merged tensor is a list of strided views into the rank files.
//! * **`torch_dist` format**: a DCP directory (+ `metadata.json` with `sharded_backend: torch_dist`).
//!   It holds global tensors already. mcore stacks the transformer layers along a leading axis
//!   (`decoder.layers.self_attention.linear_qkv.weight: [L, ...]`), which is unstacked here into
//!   `decoder.layers.{i}.…`. The `args` come from the `common_state` item (or a legacy `common.pt`).
//!
//! Either way the result uses the same TP=1 Megatron names. `OpenOpts::megatron_hf` maps them to
//! HF Llama/Qwen2 names (de-interleaving the GQA `linear_qkv` into q/k/v, splitting the SwiGLU fc1
//! into gate/up, and unpadding the vocab) and produces a matching `config.json`.

use crate::ckpt::{
    Checkpoint, Format, Loc, OpenOpts, Piece, Place, TensorInfo, ViewPart, mmap_file,
};
use crate::dtype::DType;
use crate::pickle::Allow;
use crate::torchsave::{Collected, TorchSave};
use anyhow::{Context, Result, anyhow, bail};
use serde_json::{Value as J, json};
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::path::{Path, PathBuf};

// ------------------------------------------------------------------ detection

/// Most transformer layers accepted (stacked `torch_dist` tensors, PP layer offsets). Real models
/// have a few hundred at most; the bound stops a hostile shape from creating billions of entries.
pub const MAX_LAYERS: u64 = 1 << 16;

pub fn is_torch_dist(dir: &Path) -> bool {
    std::fs::read(dir.join("metadata.json"))
        .ok()
        .and_then(|b| serde_json::from_slice::<J>(&b).ok())
        .is_some_and(|j| j["sharded_backend"] == "torch_dist")
}

/// `latest_checkpointed_iteration.txt` -> `iter_XXXXXXX` (or `release`) subdirectory.
pub fn tracker_target(dir: &Path) -> Result<Option<PathBuf>> {
    let t = dir.join("latest_checkpointed_iteration.txt");
    if !t.is_file() {
        return Ok(None);
    }
    let s = std::fs::read_to_string(&t)?;
    let s = s.trim();
    let sub = if s == "release" {
        dir.join("release")
    } else {
        let it: u64 = s
            .parse()
            .with_context(|| format!("{}: bad iteration {s:?}", t.display()))?;
        dir.join(format!("iter_{it:07}"))
    };
    if !sub.is_dir() {
        bail!("{} points to missing {}", t.display(), sub.display());
    }
    Ok(Some(sub))
}

fn parse_rank_dir(name: &str) -> Option<(usize, usize, Option<usize>)> {
    let rest = name.strip_prefix("mp_rank_")?;
    let parts: Vec<&str> = rest.split('_').collect();
    let num = |s: &str| s.parse::<usize>().ok();
    match parts.as_slice() {
        [t] => Some((num(t)?, 0, None)),
        [t, p] => Some((num(t)?, num(p)?, None)),
        [t, p, e] => Some((num(t)?, num(p)?, Some(num(e)?))),
        _ => None,
    }
}

fn rank_file(d: &Path) -> Option<PathBuf> {
    ["model_optim_rng.pt", "model_rng.pt"]
        .iter()
        .map(|f| d.join(f))
        .find(|p| p.is_file())
}

pub fn is_legacy_dir(dir: &Path) -> Result<bool> {
    for e in std::fs::read_dir(dir)? {
        let p = e?.path();
        if p.is_dir()
            && p.file_name()
                .and_then(|n| n.to_str())
                .and_then(parse_rank_dir)
                .is_some()
            && rank_file(&p).is_some()
        {
            return Ok(true);
        }
    }
    Ok(false)
}

// ------------------------------------------------------------------ args

/// The handful of Megatron arguments needed to merge and map a checkpoint.
#[derive(Clone, Debug)]
pub struct MArgs {
    pub raw: J,
}

impl MArgs {
    fn u(&self, k: &str) -> Option<u64> {
        self.raw.get(k).and_then(|v| v.as_u64())
    }
    fn b(&self, k: &str) -> bool {
        self.raw.get(k).and_then(|v| v.as_bool()).unwrap_or(false)
    }
    fn s(&self, k: &str) -> Option<&str> {
        self.raw.get(k).and_then(|v| v.as_str())
    }
    fn f(&self, k: &str) -> Option<f64> {
        self.raw.get(k).and_then(|v| v.as_f64())
    }
    pub fn num_layers(&self) -> Option<u64> {
        self.u("num_layers")
    }
    pub fn heads(&self) -> Result<u64> {
        self.u("num_attention_heads")
            .ok_or_else(|| anyhow!("args.num_attention_heads missing"))
    }
    pub fn groups(&self) -> Result<u64> {
        if self.b("group_query_attention") {
            self.u("num_query_groups")
                .ok_or_else(|| anyhow!("args.num_query_groups missing"))
        } else {
            self.heads()
        }
    }
    pub fn hidden(&self) -> Result<u64> {
        self.u("hidden_size")
            .ok_or_else(|| anyhow!("args.hidden_size missing"))
    }
    pub fn head_dim(&self) -> Result<u64> {
        match self.u("kv_channels") {
            Some(k) => Ok(k),
            None => Ok(self.hidden()? / self.heads()?.max(1)),
        }
    }
    pub fn swiglu(&self) -> bool {
        self.b("swiglu")
    }
    fn summary(&self) -> J {
        let keys = [
            "num_layers",
            "hidden_size",
            "ffn_hidden_size",
            "num_attention_heads",
            "group_query_attention",
            "num_query_groups",
            "kv_channels",
            "max_position_embeddings",
            "position_embedding_type",
            "normalization",
            "swiglu",
            "add_qkv_bias",
            "add_bias_linear",
            "untie_embeddings_and_output_weights",
            "vocab_size",
            "padded_vocab_size",
            "make_vocab_size_divisible_by",
            "tensor_model_parallel_size",
            "pipeline_model_parallel_size",
            "params_dtype",
            "ckpt_format",
        ];
        let mut m = serde_json::Map::new();
        for k in keys {
            if let Some(v) = self.raw.get(k) {
                m.insert(k.into(), v.clone());
            }
        }
        J::Object(m)
    }
}

fn args_from(ts: &TorchSave) -> Option<MArgs> {
    let pk = &ts.pickle;
    let a = ts.root_get("args")?;
    let st = pk.obj_state(a)?;
    Some(MArgs {
        raw: pk.to_json(st),
    })
}

// ------------------------------------------------------------------ TP merge rules

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Rule {
    Replicated,
    Column,
    Row,
    /// SwiGLU fc1: each TP rank holds [gate_r; up_r] along dim 0
    Gated,
}

fn rule_for(name: &str, swiglu: bool) -> Rule {
    const COL: &[&str] = &[
        ".linear_qkv.weight",
        ".linear_qkv.bias",
        ".query_key_value.weight",
        ".query_key_value.bias",
        "word_embeddings.weight",
        "output_layer.weight",
        ".linear_q_proj.weight",
        ".linear_kv_up_proj.weight",
    ];
    const FC1: &[&str] = &[
        ".linear_fc1.weight",
        ".linear_fc1.bias",
        ".dense_h_to_4h.weight",
        ".dense_h_to_4h.bias",
    ];
    const ROW: &[&str] = &[
        ".linear_proj.weight",
        ".linear_fc2.weight",
        ".self_attention.dense.weight",
        ".dense_4h_to_h.weight",
    ];
    let ends = |l: &[&str]| l.iter().any(|s| name.ends_with(s) || name == &s[1..]);
    if ends(FC1) {
        return if swiglu { Rule::Gated } else { Rule::Column };
    }
    if ends(COL) {
        return Rule::Column;
    }
    if ends(ROW) {
        return Rule::Row;
    }
    Rule::Replicated
}

/// Split `a.b.layers.12.c` into ("a.b.layers.", 12, ".c").
fn split_layer(name: &str) -> Option<(&str, u64, &str)> {
    let mut search = 0;
    while let Some(i) = name[search..].find("layers.") {
        let at = search + i;
        if at == 0 || name.as_bytes()[at - 1] == b'.' {
            let rest = &name[at + 7..];
            let n = rest.bytes().take_while(|c| c.is_ascii_digit()).count();
            if n > 0 && (rest.len() == n || rest.as_bytes()[n] == b'.') {
                let idx = rest[..n].parse().ok()?;
                return Some((&name[..at + 7], idx, &rest[n..]));
            }
        }
        search = at + 7;
    }
    None
}

// ------------------------------------------------------------------ legacy

pub fn open_legacy(dir: &Path, o: &OpenOpts) -> Result<Checkpoint> {
    let mut grid: BTreeMap<(usize, usize), PathBuf> = BTreeMap::new();
    for e in std::fs::read_dir(dir)? {
        let p = e?.path();
        let Some((t, pp, ep)) = p
            .file_name()
            .and_then(|n| n.to_str())
            .and_then(parse_rank_dir)
        else {
            continue;
        };
        if ep.is_some() {
            bail!(
                "{}: expert-parallel Megatron checkpoints (mp_rank_TT_PPP_EEE) are not supported",
                dir.display()
            );
        }
        if let Some(f) = rank_file(&p) {
            grid.insert((t, pp), f);
        }
    }
    let tp = grid.keys().map(|k| k.0).max().unwrap() + 1;
    let pp = grid.keys().map(|k| k.1).max().unwrap() + 1;
    for t in 0..tp {
        for p in 0..pp {
            if !grid.contains_key(&(t, p)) {
                bail!("{}: missing rank file for tp={t} pp={p}", dir.display());
            }
        }
    }
    let mut ck = Checkpoint::new(Format::Megatron, dir);
    // open every rank file (mmap + restricted pickle)
    let mut rank: HashMap<(usize, usize), (usize, TorchSave)> = HashMap::new();
    for (&k, f) in &grid {
        let mf = mmap_file(f)?;
        let ts = TorchSave::open(&mf.map, Allow::Megatron)
            .with_context(|| format!("reading {}", f.display()))?;
        ck.files.push(mf);
        rank.insert(k, (ck.files.len() - 1, ts));
    }
    let (_, ts0) = &rank[&(0, 0)];
    let args = args_from(ts0).unwrap_or(MArgs { raw: json!({}) });
    for k in ["iteration", "checkpoint_version"] {
        if let Some(v) = ts0.root_get(k) {
            ck.info.insert(k.into(), ts0.pickle.to_json(v));
        }
    }
    if let Some(a) = args
        .raw
        .get("tensor_model_parallel_size")
        .and_then(|v| v.as_u64())
        && a as usize != tp
    {
        bail!("args.tensor_model_parallel_size={a} but found {tp} TP rank dirs");
    }
    ck.info.insert("tensor_parallel".into(), json!(tp));
    ck.info.insert("pipeline_parallel".into(), json!(pp));
    ck.info.insert("args".into(), args.summary());
    let swiglu = args.swiglu();

    // collect model tensors per rank
    let mut per_rank: HashMap<(usize, usize), Vec<crate::torchsave::TsTensor>> = HashMap::new();
    let mut skipped_optim = 0usize;
    for (&k, (_, ts)) in &rank {
        let model = match ts.root_get("model") {
            Some(m) => m,
            None if ts.root_get("model0").is_some() => bail!(
                "virtual pipeline parallel checkpoints (model0, model1, ...) are not supported yet"
            ),
            None => bail!("rank file tp={} pp={} has no 'model' entry", k.0, k.1),
        };
        let mut col = Collected::default();
        ts.collect(model, "", &mut col)?;
        col.tensors.retain(|t| !t.path.ends_with("._extra_state"));
        if let Some(opt) = ts.root_get("optimizer") {
            let mut oc = Collected::default();
            ts.collect(opt, "", &mut oc)?;
            skipped_optim += oc.tensors.len();
        }
        if crate::zipread::verify_enabled() {
            for t in &col.tensors {
                ts.verify_storage(&ck.files[rank[&k].0].map, &t.storage_key)?;
            }
        }
        per_rank.insert(k, col.tensors);
    }
    if skipped_optim > 0 {
        ck.info.insert(
            "note".into(),
            json!(format!(
                "optimizer state ({skipped_optim} tensors across ranks) is not merged; only model weights are exposed"
            )),
        );
    }

    // PP: layer offsets from the number of local layers of earlier stages
    let mut offsets = vec![0u64; pp];
    for p in 1..pp {
        let prev: BTreeSet<u64> = per_rank[&(0, p - 1)]
            .iter()
            .filter_map(|t| split_layer(&t.path).map(|x| x.1))
            .collect();
        let local = match prev.iter().max() {
            Some(&m) if m < MAX_LAYERS => m + 1,
            Some(&m) => bail!(
                "pipeline stage {} has layer index {m}; at most {MAX_LAYERS} layers are supported",
                p - 1
            ),
            None => 0,
        };
        offsets[p] = offsets[p - 1] + local;
    }
    // global name -> per TP rank tensor (first PP stage holding it wins, e.g. tied embeddings)
    let mut order: Vec<String> = Vec::new();
    let mut by_name: HashMap<String, Vec<Option<(usize, crate::torchsave::TsTensor)>>> =
        HashMap::new();
    for p in 0..pp {
        for t in 0..tp {
            let file = rank[&(t, p)].0;
            for ten in &per_rank[&(t, p)] {
                let g = match split_layer(&ten.path) {
                    Some((pre, i, rest)) => format!("{pre}{}{rest}", i.saturating_add(offsets[p])),
                    None => ten.path.clone(),
                };
                let slot = by_name.entry(g.clone()).or_insert_with(|| {
                    order.push(g.clone());
                    vec![None; tp]
                });
                if slot[t].is_none() {
                    slot[t] = Some((file, ten.clone()));
                }
            }
        }
    }
    for name in order {
        let shards = &by_name[&name];
        let shards: Vec<&(usize, crate::torchsave::TsTensor)> = shards
            .iter()
            .enumerate()
            .map(|(t, s)| {
                s.as_ref()
                    .ok_or_else(|| anyhow!("{name}: missing on tp rank {t}"))
            })
            .collect::<Result<_>>()?;
        let t0 = &shards[0].1;
        for (_, s) in &shards {
            if s.dtype != t0.dtype || s.sizes.len() != t0.sizes.len() {
                bail!("{name}: TP shards disagree on dtype/rank");
            }
        }
        let n = t0.sizes.len();
        let rule = if tp == 1 {
            Rule::Replicated
        } else {
            rule_for(&name, swiglu)
        };
        let mut parts = Vec::new();
        let mut shape = t0.sizes.clone();
        match rule {
            Rule::Replicated => {
                if shards.iter().any(|(_, s)| s.sizes != t0.sizes) {
                    bail!("{name}: shapes differ across TP ranks but no TP rule matches this name");
                }
                parts.push(ViewPart::from_ts(shards[0].0, t0, Place::Block(vec![0; n])));
            }
            Rule::Column | Rule::Row | Rule::Gated => {
                let dim = if rule == Rule::Row { 1 } else { 0 };
                if dim >= n {
                    bail!("{name}: rank {n} too small for a dim-{dim} TP split");
                }
                if shards.iter().any(|(_, s)| s.sizes != t0.sizes) {
                    bail!("{name}: uneven TP shards are not supported");
                }
                let per = t0.sizes[dim];
                shape[dim] = per
                    .checked_mul(tp as u64)
                    .ok_or_else(|| anyhow!("{name}: merged shape overflows"))?;
                for (r, (file, s)) in shards.iter().enumerate() {
                    let base = ViewPart::from_ts(*file, s, Place::Block(vec![0; n]));
                    if rule == Rule::Gated {
                        if !per.is_multiple_of(2) {
                            bail!("{name}: SwiGLU fc1 shard has an odd number of rows");
                        }
                        let h = per / 2;
                        let mut off = vec![0; n];
                        off[0] = r as u64 * h;
                        let mut g = base.narrow(0, 0, h);
                        g.place = Place::Block(off.clone());
                        parts.push(g);
                        off[0] = tp as u64 * h + r as u64 * h;
                        let mut u = base.narrow(0, h, h);
                        u.place = Place::Block(off);
                        parts.push(u);
                    } else {
                        let mut off = vec![0; n];
                        off[dim] = r as u64 * per;
                        let mut v = base;
                        v.place = Place::Block(off);
                        parts.push(v);
                    }
                }
            }
        }
        ck.tensors.push(TensorInfo {
            name,
            dtype: t0.dtype,
            shape,
            loc: Loc::Views { parts },
        });
    }
    let mut scalars = Collected::default();
    if let Some(v) = ts0.root_get("iteration") {
        scalars
            .scalars
            .push(("iteration".into(), ts0.pickle.to_json(v)));
    }
    ck.scalars = scalars.scalars;
    drop(rank);
    finish(ck, args, o, false)
}

// ------------------------------------------------------------------ torch_dist

pub fn from_torch_dist(mut ck: Checkpoint, o: &OpenOpts) -> Result<Checkpoint> {
    ck.format = Format::MegatronDist;
    // args: current format keeps the common dict as a pickled `common_state` item; older ones use common.pt
    let mut args = None;
    if let Some(b) = ck
        .bytes_items
        .iter()
        .find(|b| b.name == "common_state" || b.name.starts_with("common_state/"))
    {
        let f = &ck.files[b.file].map;
        let blob = usize::try_from(b.info.offset)
            .ok()
            .zip(usize::try_from(b.info.length).ok())
            .and_then(|(s, l)| f.get(s..s.checked_add(l)?))
            .ok_or_else(|| anyhow!("common_state out of bounds"))?;
        let ts = TorchSave::open(blob, Allow::Megatron).context("decoding common_state")?;
        args = args_from(&ts);
        if let Some(v) = ts.root_get("iteration") {
            ck.info.insert("iteration".into(), ts.pickle.to_json(v));
        }
    } else if ck.root.join("common.pt").is_file() {
        let mf = mmap_file(&ck.root.join("common.pt"))?;
        let ts = TorchSave::open(&mf.map, Allow::Megatron).context("reading common.pt")?;
        args = args_from(&ts);
    }
    let args = args.unwrap_or(MArgs { raw: json!({}) });
    ck.info.insert("args".into(), args.summary());
    // unstack `...decoder.layers.<rest>` tensors that carry a leading layer axis
    let base = std::mem::take(&mut ck.tensors);
    let nl = args.num_layers();
    let mut out = Vec::new();
    for (i, t) in base.iter().enumerate() {
        let stacked = t.name.contains("decoder.layers.")
            && split_layer(&t.name).is_none()
            && !t.shape.is_empty()
            && nl.is_none_or(|l| t.shape[0] == l);
        if stacked {
            if t.shape[0] > MAX_LAYERS {
                bail!(
                    "{}: {} stacked layers; at most {MAX_LAYERS} are supported",
                    t.name,
                    t.shape[0]
                );
            }
            let pos = t.name.find("decoder.layers.").unwrap() + "decoder.layers.".len();
            for l in 0..t.shape[0] {
                let mut start = vec![0; t.shape.len()];
                start[0] = l;
                let mut size = t.shape.clone();
                size[0] = 1;
                out.push(TensorInfo {
                    name: format!("{}{l}.{}", &t.name[..pos], &t.name[pos..]),
                    dtype: t.dtype,
                    shape: t.shape[1..].to_vec(),
                    loc: Loc::Derived {
                        pieces: vec![Piece {
                            base: i,
                            start,
                            size,
                            out_offset: vec![0; t.shape.len() - 1],
                        }],
                    },
                });
            }
        } else {
            out.push(identity(i, t));
        }
    }
    ck.base = base;
    ck.tensors = out;
    // logical sort: layers in numeric order
    ck.tensors
        .sort_by_cached_key(|t| match split_layer(&t.name) {
            Some((pre, i, rest)) => (pre.to_string(), i, rest.to_string()),
            None => (t.name.clone(), 0, String::new()),
        });
    finish(ck, args, o, true)
}

fn identity(i: usize, t: &TensorInfo) -> TensorInfo {
    TensorInfo {
        name: t.name.clone(),
        dtype: t.dtype,
        shape: t.shape.clone(),
        loc: Loc::Derived {
            pieces: vec![Piece {
                base: i,
                start: vec![0; t.shape.len()],
                size: t.shape.clone(),
                out_offset: vec![0; t.shape.len()],
            }],
        },
    }
}

/// Apply `--vocab-size` unpadding and the optional HF mapping.
fn finish(mut ck: Checkpoint, args: MArgs, o: &OpenOpts, derived: bool) -> Result<Checkpoint> {
    if o.megatron_hf.is_none() {
        if let Some(v) = o.vocab_size {
            // unpad embeddings / output layer in the Megatron-named view
            if !derived {
                ck.base = std::mem::take(&mut ck.tensors);
                ck.tensors = ck
                    .base
                    .iter()
                    .enumerate()
                    .map(|(i, t)| identity(i, t))
                    .collect();
            }
            for t in &mut ck.tensors {
                if is_vocab(&t.name) {
                    unpad(t, v)?;
                }
            }
        }
        return Ok(ck);
    }
    let arch = o.megatron_hf.clone().unwrap();
    if !derived {
        ck.base = std::mem::take(&mut ck.tensors);
        ck.tensors = ck
            .base
            .iter()
            .enumerate()
            .map(|(i, t)| identity(i, t))
            .collect();
    }
    let vocab = o
        .vocab_size
        .or_else(|| args.u("vocab_size"))
        .or_else(|| {
            let v = ck
                .tensors
                .iter()
                .find(|t| is_vocab(&t.name))
                .and_then(|t| t.shape.first().copied());
            if let Some(v) = v {
                eprintln!(
                    "warning: vocab size unknown (no args.vocab_size); keeping the padded vocab of {v} rows. Pass --vocab-size to unpad."
                );
            }
            v
        });
    let (tensors, cfg) = hf_map(&ck, &args, &arch, vocab)?;
    ck.tensors = tensors;
    ck.hf_config = Some(cfg);
    Ok(ck)
}

fn is_vocab(name: &str) -> bool {
    name.ends_with("word_embeddings.weight") || name.ends_with("output_layer.weight")
}

/// Restrict a Derived tensor to its first `v` rows (vocab unpadding).
fn unpad(t: &mut TensorInfo, v: u64) -> Result<()> {
    if t.shape.is_empty() {
        bail!(
            "{}: a vocab tensor must have at least one dimension",
            t.name
        );
    }
    if t.shape[0] < v {
        bail!(
            "{}: --vocab-size {v} exceeds the {} stored rows",
            t.name,
            t.shape[0]
        );
    }
    let Loc::Derived { pieces } = &mut t.loc else {
        bail!("internal: unpad on non-derived tensor")
    };
    if pieces.len() != 1 || pieces[0].size.len() < t.shape.len() {
        bail!("internal: unpad of a composite tensor {}", t.name);
    }
    let drop = pieces[0].size.len() - t.shape.len();
    pieces[0].size[drop] = v;
    t.shape[0] = v;
    Ok(())
}

// ------------------------------------------------------------------ Megatron -> HF

/// Canonical (mcore) name for older Megatron-LM (`language_model.encoder...`) names.
fn canonical(name: &str) -> String {
    let mut n = name.to_string();
    for (a, b) in [
        (
            "language_model.embedding.word_embeddings.",
            "embedding.word_embeddings.",
        ),
        (
            "language_model.embedding.position_embeddings.",
            "embedding.position_embeddings.",
        ),
        (
            "language_model.encoder.final_layernorm.",
            "decoder.final_layernorm.",
        ),
        (
            "language_model.encoder.final_norm.",
            "decoder.final_layernorm.",
        ),
        ("language_model.encoder.layers.", "decoder.layers."),
        ("language_model.output_layer.", "output_layer."),
        (
            ".self_attention.query_key_value.",
            ".self_attention.linear_qkv.",
        ),
        (".self_attention.dense.", ".self_attention.linear_proj."),
        (".post_attention_layernorm.", ".pre_mlp_layernorm."),
        (".post_attention_norm.", ".pre_mlp_layernorm."),
        (".input_norm.", ".input_layernorm."),
        (".mlp.dense_h_to_4h.", ".mlp.linear_fc1."),
        (".mlp.dense_4h_to_h.", ".mlp.linear_fc2."),
    ] {
        n = n.replace(a, b);
    }
    n
}

fn hf_map(
    ck: &Checkpoint,
    args: &MArgs,
    arch: &str,
    vocab: Option<u64>,
) -> Result<(Vec<TensorInfo>, J)> {
    let req = |c: bool, what: &str| -> Result<()> {
        if !c {
            bail!("Megatron -> HF Llama/Qwen2 mapping needs {what}");
        }
        Ok(())
    };
    req(
        args.raw.get("num_attention_heads").is_some(),
        "args (num_attention_heads, ...) in the checkpoint",
    )?;
    req(args.swiglu(), "--swiglu (gated MLP)")?;
    req(
        args.s("normalization") == Some("RMSNorm"),
        "normalization=RMSNorm",
    )?;
    req(
        args.s("position_embedding_type")
            .is_none_or(|p| p == "rope"),
        "position_embedding_type=rope",
    )?;
    req(!args.b("rotary_interleaved"), "non-interleaved RoPE")?;
    req(
        !args.b("qk_layernorm"),
        "no qk_layernorm (Qwen3-style q/k norms are not mapped yet)",
    )?;
    req(
        !args.b("apply_residual_connection_post_layernorm"),
        "pre-norm residuals",
    )?;
    req(
        !args.b("layernorm_zero_centered_gamma"),
        "layernorm_zero_centered_gamma=false",
    )?;
    if args.f("rotary_percent").is_some_and(|p| p != 1.0) {
        bail!("Megatron -> HF mapping needs rotary_percent=1.0");
    }
    let nh = args.heads()?;
    let ng = args.groups()?;
    let hd = args.head_dim()?;
    let hidden = args.hidden()?;
    if nh == 0 || ng == 0 || hd == 0 {
        bail!(
            "invalid Megatron args: num_attention_heads={nh}, num_query_groups={ng}, head_dim={hd}"
        );
    }
    if nh % ng != 0 {
        bail!("num_attention_heads {nh} not divisible by num_query_groups {ng}");
    }
    let qpg = nh / ng;
    // every row count derived from the args below must fit comfortably in u64
    let big = |x: Option<u64>| x.is_none_or(|v| v > 1 << 40);
    if big(qpg
        .checked_add(2)
        .and_then(|x| x.checked_mul(hd))
        .and_then(|x| x.checked_mul(ng)))
        || big(nh.checked_mul(hd))
    {
        bail!("invalid Megatron args: heads x head_dim is absurdly large");
    }
    let qkv_bias = args.b("add_qkv_bias") || args.b("add_bias_linear");
    let lin_bias = args.b("add_bias_linear");
    let arch = match arch {
        "auto" => {
            if qkv_bias && !lin_bias {
                "qwen2"
            } else {
                "llama"
            }
        }
        "llama" | "qwen2" => arch,
        a => bail!("unknown --hf-arch {a:?} (use auto, llama or qwen2)"),
    };
    if arch == "qwen2" && (lin_bias || !qkv_bias) {
        bail!(
            "qwen2 needs q/k/v biases and no other linear biases (add_qkv_bias without add_bias_linear)"
        );
    }
    let mut out: Vec<TensorInfo> = Vec::new();
    let mut dtype = None;
    let mut ffn = args.u("ffn_hidden_size");
    let mut emit =
        |name: String, src: &TensorInfo, dim0: Vec<(u64, u64)>, shape: Vec<u64>| -> Result<()> {
            // `src` is an identity/derived view of base tensor; compose pieces along dim 0
            let Loc::Derived { pieces } = &src.loc else {
                bail!("internal: expected derived source")
            };
            if shape.is_empty() || src.shape.is_empty() {
                bail!("{}: a 0-d tensor cannot be mapped to HF", src.name);
            }
            let p0 = &pieces[0];
            let drop = p0.size.len() - src.shape.len();
            let mut new = Vec::new();
            let mut at = 0u64;
            for (s, l) in dim0 {
                let mut start = p0.start.clone();
                let mut size = p0.size.clone();
                start[drop] = start[drop].saturating_add(s);
                size[drop] = l;
                let mut off = vec![0; shape.len()];
                off[0] = at;
                at = at.saturating_add(l);
                new.push(Piece {
                    base: p0.base,
                    start,
                    size,
                    out_offset: off,
                });
            }
            out.push(TensorInfo {
                name,
                dtype: src.dtype,
                shape,
                loc: Loc::Derived { pieces: new },
            });
            Ok(())
        };
    for t in &ck.tensors {
        let Loc::Derived { pieces } = &t.loc else {
            bail!("internal: expected derived tensor")
        };
        if pieces.len() != 1 {
            bail!("internal: composite source {}", t.name);
        }
        let c = canonical(&t.name);
        if c.ends_with("._extra_state") || c.starts_with("optimizer.") || c.starts_with("rng_state")
        {
            continue;
        }
        dtype.get_or_insert(t.dtype);
        if t.shape.is_empty() {
            bail!("{}: a 0-d tensor cannot be mapped to HF", t.name);
        }
        let rows = t.shape[0];
        let full = |t: &TensorInfo| vec![(0, t.shape[0])];
        match c.as_str() {
            "embedding.word_embeddings.weight" | "output_layer.weight" => {
                let v = vocab.unwrap_or(rows);
                if v > rows {
                    bail!("{}: vocab size {v} > stored rows {rows}", t.name);
                }
                let n = if c.starts_with("embedding") {
                    "model.embed_tokens.weight"
                } else {
                    "lm_head.weight"
                };
                if t.shape.len() != 2 {
                    bail!(
                        "{}: expected a 2-d embedding, got shape {:?}",
                        t.name,
                        t.shape
                    );
                }
                emit(n.into(), t, vec![(0, v)], vec![v, t.shape[1]])?;
                continue;
            }
            "decoder.final_layernorm.weight" => {
                emit("model.norm.weight".into(), t, full(t), t.shape.clone())?;
                continue;
            }
            "embedding.position_embeddings.weight" => {
                bail!("learned position embeddings cannot map to Llama/Qwen2")
            }
            _ => {}
        }
        let Some((pre, l, rest)) = split_layer(&c) else {
            bail!("no HF mapping for Megatron tensor {}", t.name);
        };
        if pre != "decoder.layers." {
            bail!("no HF mapping for Megatron tensor {}", t.name);
        }
        let hp = format!("model.layers.{l}");
        match rest {
            ".input_layernorm.weight" | ".self_attention.linear_qkv.layer_norm_weight" => emit(
                format!("{hp}.input_layernorm.weight"),
                t,
                full(t),
                t.shape.clone(),
            )?,
            ".pre_mlp_layernorm.weight" | ".mlp.linear_fc1.layer_norm_weight" => emit(
                format!("{hp}.post_attention_layernorm.weight"),
                t,
                full(t),
                t.shape.clone(),
            )?,
            ".self_attention.linear_qkv.weight" | ".self_attention.linear_qkv.bias" => {
                // Megatron fused GQA layout: per query group [q heads of the group | k | v]
                let per = (qpg + 2) * hd;
                if rows != ng * per {
                    bail!(
                        "{}: {rows} rows, expected num_query_groups({ng}) x (heads/group({qpg}) + 2) x head_dim({hd})",
                        t.name
                    );
                }
                let kind = if rest.ends_with("weight") {
                    "weight"
                } else {
                    "bias"
                };
                let shape_of = |n: u64| {
                    let mut s = t.shape.clone();
                    s[0] = n;
                    s
                };
                let q = (0..ng).map(|g| (g * per, qpg * hd)).collect();
                let k = (0..ng).map(|g| (g * per + qpg * hd, hd)).collect();
                let v = (0..ng).map(|g| (g * per + (qpg + 1) * hd, hd)).collect();
                emit(
                    format!("{hp}.self_attn.q_proj.{kind}"),
                    t,
                    q,
                    shape_of(nh * hd),
                )?;
                emit(
                    format!("{hp}.self_attn.k_proj.{kind}"),
                    t,
                    k,
                    shape_of(ng * hd),
                )?;
                emit(
                    format!("{hp}.self_attn.v_proj.{kind}"),
                    t,
                    v,
                    shape_of(ng * hd),
                )?;
            }
            ".self_attention.linear_proj.weight" => emit(
                format!("{hp}.self_attn.o_proj.weight"),
                t,
                full(t),
                t.shape.clone(),
            )?,
            ".self_attention.linear_proj.bias" => emit(
                format!("{hp}.self_attn.o_proj.bias"),
                t,
                full(t),
                t.shape.clone(),
            )?,
            ".mlp.linear_fc1.weight" | ".mlp.linear_fc1.bias" => {
                let h = rows / 2;
                ffn.get_or_insert(h);
                let kind = if rest.ends_with("weight") {
                    "weight"
                } else {
                    "bias"
                };
                let mut s = t.shape.clone();
                s[0] = h;
                emit(
                    format!("{hp}.mlp.gate_proj.{kind}"),
                    t,
                    vec![(0, h)],
                    s.clone(),
                )?;
                emit(format!("{hp}.mlp.up_proj.{kind}"), t, vec![(h, h)], s)?;
            }
            ".mlp.linear_fc2.weight" => emit(
                format!("{hp}.mlp.down_proj.weight"),
                t,
                full(t),
                t.shape.clone(),
            )?,
            ".mlp.linear_fc2.bias" => emit(
                format!("{hp}.mlp.down_proj.bias"),
                t,
                full(t),
                t.shape.clone(),
            )?,
            _ => bail!("no HF mapping for Megatron tensor {}", t.name),
        }
    }
    let has_lm_head = out.iter().any(|t| t.name == "lm_head.weight");
    let torch_dtype = match dtype.unwrap_or(DType::F32) {
        DType::BF16 => "bfloat16",
        DType::F16 => "float16",
        _ => "float32",
    };
    let mut cfg = json!({
        "architectures": [if arch == "qwen2" { "Qwen2ForCausalLM" } else { "LlamaForCausalLM" }],
        "model_type": arch,
        "hidden_size": hidden,
        "intermediate_size": ffn.unwrap_or(hidden.saturating_mul(4)),
        "num_hidden_layers": args.num_layers(),
        "num_attention_heads": nh,
        "num_key_value_heads": ng,
        "head_dim": hd,
        "hidden_act": "silu",
        "max_position_embeddings": args.u("max_position_embeddings").or(args.u("seq_length")),
        "rms_norm_eps": args.f("layernorm_epsilon").or(args.f("norm_epsilon")).unwrap_or(1e-5),
        "rope_theta": args.f("rotary_base").unwrap_or(10000.0),
        "vocab_size": vocab,
        "tie_word_embeddings": !has_lm_head,
        "torch_dtype": torch_dtype,
        "initializer_range": 0.02,
        "use_cache": true,
    });
    if arch == "llama" {
        cfg["attention_bias"] = json!(qkv_bias);
        cfg["mlp_bias"] = json!(lin_bias);
    } else {
        cfg["use_sliding_window"] = json!(false);
    }
    Ok((out, cfg))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn layer_split() {
        assert_eq!(
            split_layer("decoder.layers.12.mlp.linear_fc1.weight"),
            Some(("decoder.layers.", 12, ".mlp.linear_fc1.weight"))
        );
        assert_eq!(split_layer("decoder.layers.self_attention.x"), None);
        assert_eq!(
            split_layer("language_model.encoder.layers.3.input_norm.weight"),
            Some(("language_model.encoder.layers.", 3, ".input_norm.weight"))
        );
        assert_eq!(split_layer("xlayers.3.a"), None);
        assert_eq!(parse_rank_dir("mp_rank_01_002"), Some((1, 2, None)));
        assert_eq!(parse_rank_dir("mp_rank_03"), Some((3, 0, None)));
        assert_eq!(
            rule_for("decoder.layers.0.mlp.linear_fc1.weight", true),
            Rule::Gated
        );
        assert_eq!(
            rule_for("decoder.layers.0.self_attention.linear_proj.weight", true),
            Rule::Row
        );
        assert_eq!(
            rule_for("embedding.word_embeddings.weight", true),
            Rule::Column
        );
        assert_eq!(
            rule_for("decoder.final_layernorm.weight", true),
            Rule::Replicated
        );
    }
}
