//! Tensor-parallel (TP) splitting and merging driven by name-pattern rules.
//!
//! rules.yaml:
//! ```yaml
//! config:                   # optional; values referenced by name below. Missing names are looked up
//!   num_attention_heads: 32 # in the source's config.json (or --model-config)
//! default: replicate        # or "error": every tensor must match a rule
//! rules:
//!   - pattern: "*.attn.c_attn.weight"   # first matching rule wins (`*`, `?` wildcards)
//!     dim: 1                            # split along this dim ...
//!     parts: 3                          # ... as 3 equal fused blocks (q,k,v), each split separately
//!   - pattern: "*.q_proj.weight"
//!     dim: 0
//!     heads: num_attention_heads        # split on head boundaries (heads % tp == 0)
//!   - pattern: "*.k_proj.weight"
//!     dim: 0
//!     kv_heads: num_key_value_heads     # GQA: like heads, but if tp > kv heads each kv head is
//!                                       # replicated on tp/kv consecutive ranks
//!   - pattern: "*.qkv_proj.weight"      # fused q|k|v with unequal widths (GQA)
//!     dim: 0
//!     sections: [{heads: num_attention_heads}, {kv_heads: num_key_value_heads}, {kv_heads: num_key_value_heads}]
//!   - pattern: "*.embed_tokens.weight"
//!     dim: 0
//!     pad_multiple: 64                  # vocab padding: rows padded with zeros to a multiple of tp*64
//!   - pattern: "*.norm.weight"
//!     replicate: true
//! ```
//! `split: column|row` is an alias for dim 0 / dim 1 (torch nn.Linear weights are [out, in]).
//! Output layout: `<out>/tp_rank_XX/model.safetensors` + `<out>/tp_plan.json`, which records each
//! tensor's resolved layout (section sizes, unpadded length), so merging and re-splitting to another
//! TP size need neither the rules nor the model config.

use crate::cast::CastSpec;
use crate::ckpt::Checkpoint;
use crate::dtype::{DType, numel};
use crate::safetensors::{OutTensor, StWriter};
use crate::writer::glob_match;
use anyhow::{Context, Result, anyhow, bail};
use serde::{Deserialize, Serialize};
use std::borrow::Cow;
use std::collections::BTreeMap;
use std::fs::File;
use std::io::BufWriter;
use std::path::{Path, PathBuf};

/// A head count: a number or the name of a config value.
#[derive(Debug, Deserialize, Clone)]
#[serde(untagged)]
pub enum Num {
    Int(u64),
    Name(String),
}

#[derive(Debug, Deserialize, Clone)]
pub struct SectionSpec {
    #[serde(default)]
    pub heads: Option<Num>,
    #[serde(default)]
    pub kv_heads: Option<Num>,
}

#[derive(Debug, Deserialize, Clone, Default)]
pub struct RuleSpec {
    pub pattern: String,
    #[serde(default)]
    pub dim: Option<usize>,
    #[serde(default)]
    pub split: Option<String>,
    #[serde(default)]
    pub parts: Option<usize>,
    #[serde(default)]
    pub replicate: bool,
    #[serde(default)]
    pub heads: Option<Num>,
    #[serde(default)]
    pub kv_heads: Option<Num>,
    #[serde(default)]
    pub sections: Option<Vec<SectionSpec>>,
    #[serde(default)]
    pub pad_multiple: Option<Num>,
}

#[derive(Debug, Deserialize, Clone)]
pub struct Rules {
    #[serde(default = "default_replicate")]
    pub default: String,
    pub rules: Vec<RuleSpec>,
    /// named integers used by `heads` / `kv_heads` / `pad_multiple`
    #[serde(default)]
    pub config: BTreeMap<String, serde_yaml::Value>,
}

fn default_replicate() -> String {
    "replicate".into()
}

fn one() -> usize {
    1
}

/// One fused section along the split dim.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct Section {
    /// length along the split dim in the full (unpadded) tensor
    pub rows: u64,
    /// number of heads in the section (0 = no head structure, just divisible by tp)
    #[serde(default)]
    pub heads: u64,
    /// GQA kv heads: allow replication when tp > heads
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub replicate_kv: bool,
}

/// How one tensor is laid out across TP ranks.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct Split {
    pub dim: usize,
    #[serde(default = "one")]
    pub parts: usize,
    /// unequal / head-structured sections (overrides `parts`)
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub sections: Vec<Section>,
    /// vocab padding: the dim is zero-padded to a multiple of tp * pad_multiple before splitting
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pad_multiple: Option<u64>,
    /// unpadded length of `dim` (recorded with pad_multiple)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub orig_len: Option<u64>,
}

impl Split {
    pub fn simple(dim: usize, parts: usize) -> Split {
        Split {
            dim,
            parts,
            sections: vec![],
            pad_multiple: None,
            orig_len: None,
        }
    }
}

impl Rules {
    pub fn load(path: &Path) -> Result<Rules> {
        let s =
            std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
        let r: Rules =
            serde_yaml::from_str(&s).with_context(|| format!("parsing {}", path.display()))?;
        if r.default != "replicate" && r.default != "error" {
            bail!("rules default must be `replicate` or `error`");
        }
        Ok(r)
    }

    /// Add values from an HF config.json (rules' own `config:` entries take precedence).
    pub fn with_model_config(mut self, cfg: &serde_json::Value) -> Rules {
        if let Some(m) = cfg.as_object() {
            for (k, v) in m {
                if let Some(n) = v.as_u64() {
                    self.config
                        .entry(k.clone())
                        .or_insert(serde_yaml::Value::Number(n.into()));
                }
            }
            // conventional defaults
            if !self.config.contains_key("num_key_value_heads")
                && let Some(h) = m.get("num_attention_heads").and_then(|v| v.as_u64())
            {
                self.config.insert(
                    "num_key_value_heads".into(),
                    serde_yaml::Value::Number(h.into()),
                );
            }
        }
        self
    }

    fn num(&self, n: &Num, rule: &str) -> Result<u64> {
        match n {
            Num::Int(i) => Ok(*i),
            Num::Name(k) => self
                .config
                .get(k)
                .and_then(|v| v.as_u64())
                .ok_or_else(|| {
                    anyhow!(
                        "rule {rule}: `{k}` is not defined (add it under `config:` in the rules, or pass --model-config config.json)"
                    )
                }),
        }
    }

    /// The rule's layout for `name` without sizes (section rows are 0; None = replicated).
    pub fn rule_split(&self, name: &str) -> Result<Option<Split>> {
        for r in &self.rules {
            if !glob_match(&r.pattern, name) {
                continue;
            }
            if r.replicate {
                return Ok(None);
            }
            let dim = match (&r.dim, r.split.as_deref()) {
                (Some(d), _) => *d,
                (None, Some("column")) => 0,
                (None, Some("row")) => 1,
                (None, Some(s)) => bail!(
                    "rule {}: unknown split {s:?} (use column|row or dim)",
                    r.pattern
                ),
                (None, None) => bail!("rule {}: needs dim, split or replicate", r.pattern),
            };
            let mut s = Split::simple(dim, r.parts.unwrap_or(1).max(1));
            let mut secs: Vec<(u64, bool)> = Vec::new();
            if let Some(h) = &r.heads {
                secs.push((self.num(h, &r.pattern)?, false));
            }
            if let Some(h) = &r.kv_heads {
                secs.push((self.num(h, &r.pattern)?, true));
            }
            if let Some(list) = &r.sections {
                if !secs.is_empty() {
                    bail!("rule {}: use either heads/kv_heads or sections", r.pattern);
                }
                for sp in list {
                    match (&sp.heads, &sp.kv_heads) {
                        (Some(h), None) => secs.push((self.num(h, &r.pattern)?, false)),
                        (None, Some(h)) => secs.push((self.num(h, &r.pattern)?, true)),
                        _ => bail!(
                            "rule {}: each section needs exactly one of heads / kv_heads",
                            r.pattern
                        ),
                    }
                }
            }
            if secs.len() > 1 && r.sections.is_none() {
                bail!(
                    "rule {}: use either heads or kv_heads (or sections)",
                    r.pattern
                );
            }
            if secs.iter().any(|x| x.0 == 0) {
                bail!("rule {}: head counts must be > 0", r.pattern);
            }
            if !secs.is_empty() {
                s.sections = secs
                    .iter()
                    .map(|&(h, kv)| Section {
                        rows: 0,
                        heads: h,
                        replicate_kv: kv,
                    })
                    .collect();
                s.parts = s.sections.len();
            }
            if let Some(m) = &r.pad_multiple {
                if !s.sections.is_empty() || s.parts != 1 {
                    bail!(
                        "rule {}: pad_multiple cannot be combined with parts/sections",
                        r.pattern
                    );
                }
                s.pad_multiple = Some(self.num(m, &r.pattern)?.max(1));
            }
            return Ok(Some(s));
        }
        if self.default == "error" {
            bail!("no TP rule matches tensor {name}");
        }
        Ok(None)
    }

    /// Resolve the layout of tensor `name` with full shape `shape` (None = replicated).
    pub fn lookup(&self, name: &str, shape: &[u64]) -> Result<Option<Split>> {
        let Some(mut s) = self.rule_split(name)? else {
            return Ok(None);
        };
        if s.dim >= shape.len() {
            bail!(
                "{name}: split dim {} out of range for shape {shape:?}",
                s.dim
            );
        }
        let len = shape[s.dim];
        if !s.sections.is_empty() {
            let total: u64 = s.sections.iter().map(|x| x.heads).sum();
            if !len.is_multiple_of(total) {
                bail!(
                    "{name}: dim {} ({len}) is not a multiple of the rule's {total} heads",
                    s.dim
                );
            }
            let hd = len / total;
            for sec in &mut s.sections {
                sec.rows = sec.heads * hd;
            }
        }
        if s.pad_multiple == Some(0) || s.parts == 0 {
            bail!("{name}: pad_multiple and parts must be at least 1");
        }
        if s.pad_multiple.is_some() {
            s.orig_len = Some(len);
        }
        Ok(Some(s))
    }
}

/// The sections of a split for a full dim length `len` (legacy `parts` = equal, headless sections).
fn sections_of(s: &Split, len: u64) -> Vec<Section> {
    if !s.sections.is_empty() {
        return s.sections.clone();
    }
    let p = s.parts.max(1) as u64;
    (0..p)
        .map(|_| Section {
            rows: len / p,
            heads: 0,
            replicate_kv: false,
        })
        .collect()
}

/// Rows `[start, start+len)` of a section owned by rank `r`, and whether `r` is the primary owner
/// (false for kv-head replicas, which must equal the primary on merge).
fn section_piece(sec: &Section, tp: u64, r: u64, name: &str) -> Result<(u64, u64, bool)> {
    if sec.heads == 0 {
        if !sec.rows.is_multiple_of(tp) {
            bail!(
                "{name}: section of {} rows is not divisible by tp({tp})",
                sec.rows
            );
        }
        let per = sec.rows / tp;
        return Ok((r * per, per, true));
    }
    let hd = sec.rows / sec.heads;
    if sec.heads.is_multiple_of(tp) {
        let per = sec.heads / tp;
        return Ok((r * per * hd, per * hd, true));
    }
    if sec.replicate_kv && tp.is_multiple_of(sec.heads) {
        let rep = tp / sec.heads;
        let h = r / rep;
        return Ok((h * hd, hd, r.is_multiple_of(rep)));
    }
    if sec.replicate_kv {
        bail!(
            "{name}: {} kv heads cannot be split over tp={tp} (need kv_heads % tp == 0 or tp % kv_heads == 0)",
            sec.heads
        );
    }
    bail!("{name}: {} heads are not divisible by tp={tp}", sec.heads)
}

/// A byte range of the full tensor that belongs to a rank shard, in shard order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Seg {
    /// bytes [off, off+len) of the full tensor; `primary` = false for replicated kv heads
    Data {
        off: usize,
        len: usize,
        primary: bool,
    },
    /// zero padding (vocab padding rows)
    Pad { len: usize },
}

impl Seg {
    pub fn bytes(&self) -> usize {
        match self {
            Seg::Data { len, .. } | Seg::Pad { len } => *len,
        }
    }
}

/// Length of `dim` in rank shards.
pub fn shard_len(full_len: u64, s: &Split, tp: usize, name: &str) -> Result<u64> {
    if let Some(m) = s.pad_multiple {
        let q = tp as u64 * m;
        return Ok(full_len.div_ceil(q) * q / tp as u64);
    }
    let mut n = 0;
    for sec in sections_of(s, full_len) {
        n += section_piece(&sec, tp as u64, 0, name)?.1;
    }
    Ok(n)
}

/// Byte segments (of the full, unpadded tensor) that make up rank `r`'s shard, in order.
pub fn shard_segments(
    shape: &[u64],
    es: usize,
    s: &Split,
    tp: usize,
    r: usize,
    name: &str,
) -> Result<Vec<Seg>> {
    if s.dim >= shape.len() {
        bail!("split dim {} out of range for shape {:?}", s.dim, shape);
    }
    let len = shape[s.dim];
    let outer: usize = shape[..s.dim].iter().product::<u64>() as usize;
    let inner: usize = shape[s.dim + 1..].iter().product::<u64>() as usize * es;
    // (row start, rows, primary) or padding rows
    let mut rows: Vec<(u64, u64, Option<bool>)> = Vec::new();
    if let Some(m) = s.pad_multiple {
        let q = tp as u64 * m;
        let per = len.div_ceil(q) * q / tp as u64;
        let a = r as u64 * per;
        let b = a + per;
        if a < len {
            rows.push((a, b.min(len) - a, Some(true)));
        }
        if b > len {
            rows.push((0, b - a.max(len), None));
        }
    } else {
        let secs = sections_of(s, len);
        let total: u64 = secs.iter().map(|x| x.rows).sum();
        if total != len {
            bail!(
                "{name}: dim {} ({len}) does not match the rule's sections ({total}){}",
                s.dim,
                if s.sections.is_empty() {
                    format!(" -- not divisible by parts({})", s.parts)
                } else {
                    String::new()
                }
            );
        }
        let mut base = 0;
        for sec in &secs {
            let (st, n, primary) = section_piece(sec, tp as u64, r as u64, name).map_err(|e| {
                if sec.heads == 0 {
                    anyhow!(
                        "dim {} of shape {:?} ({len}) is not divisible by parts({}) x tp({tp})",
                        s.dim,
                        shape,
                        secs.len()
                    )
                } else {
                    e
                }
            })?;
            rows.push((base + st, n, Some(primary)));
            base += sec.rows;
        }
    }
    let mut out = Vec::with_capacity(outer * rows.len());
    for o in 0..outer {
        for &(st, n, kind) in &rows {
            let len_b = n as usize * inner;
            match kind {
                Some(primary) => out.push(Seg::Data {
                    off: (o * len as usize + st as usize) * inner,
                    len: len_b,
                    primary,
                }),
                None => out.push(Seg::Pad { len: len_b }),
            }
        }
    }
    Ok(out)
}

/// Legacy helper: plain (offset, len) byte ranges of rank `r` (no padding / replication).
pub fn shard_ranges(
    shape: &[u64],
    es: usize,
    s: &Split,
    tp: usize,
    r: usize,
) -> Result<Vec<(usize, usize)>> {
    shard_segments(shape, es, s, tp, r, "tensor")?
        .into_iter()
        .map(|g| match g {
            Seg::Data { off, len, .. } => Ok((off, len)),
            Seg::Pad { .. } => bail!("padded split"),
        })
        .collect()
}

pub fn shard_shape(shape: &[u64], s: Option<&Split>, tp: usize, name: &str) -> Result<Vec<u64>> {
    let mut v = shape.to_vec();
    if let Some(s) = s {
        v[s.dim] = shard_len(shape[s.dim], s, tp, name)?;
    }
    Ok(v)
}

/// Full (unpadded) length of the split dim, given a shard's length.
fn full_len(s: &Split, shard: u64, tp: usize) -> u64 {
    if let Some(o) = s.orig_len {
        return o;
    }
    if !s.sections.is_empty() {
        return s
            .sections
            .iter()
            .fold(0u64, |a, x| a.saturating_add(x.rows));
    }
    shard.saturating_mul(tp as u64)
}

/// Sanity-check a split read from `tp_plan.json` (or inferred from rules) against a rank shard of
/// shape `shard`, so that merging cannot divide by zero, overflow or allocate more than the
/// rank files hold.
fn check_split(s: &Split, shard: &[u64], dtype: DType, tp: usize, name: &str) -> Result<()> {
    const BIG: u64 = 1 << 40;
    if s.dim >= shard.len() {
        bail!(
            "{name}: split dim {} out of range for shape {shard:?}",
            s.dim
        );
    }
    if s.parts == 0 || s.parts as u64 > BIG {
        bail!("{name}: invalid parts {}", s.parts);
    }
    if let Some(m) = s.pad_multiple
        && (m == 0 || m > BIG)
    {
        bail!("{name}: invalid pad_multiple {m}");
    }
    if s.sections.iter().any(|x| x.rows > BIG || x.heads > BIG) {
        bail!("{name}: invalid sections");
    }
    let mut full = shard.to_vec();
    full[s.dim] = full_len(s, shard[s.dim], tp);
    let full_bytes = crate::dtype::checked_nbytes(&full, dtype)?;
    let shard_bytes = crate::dtype::checked_nbytes(shard, dtype)?;
    if full_bytes > shard_bytes.saturating_mul(tp as u64) {
        bail!(
            "{name}: the plan implies a full shape {full:?} larger than the {tp} rank shards of {shard:?}"
        );
    }
    Ok(())
}

#[derive(Serialize, Deserialize)]
pub struct TpPlan {
    pub tp: usize,
    /// tensor name -> split (absent/null = replicated)
    pub tensors: BTreeMap<String, Option<Split>>,
}

/// A logical, full-tensor view over either a plain checkpoint or a set of TP rank checkpoints.
#[allow(clippy::large_enum_variant)]
pub enum Source {
    Plain(Checkpoint),
    Tp {
        ranks: Vec<Checkpoint>,
        plan: TpPlan,
    },
}

pub struct Entry {
    pub name: String,
    pub dtype: DType,
    pub shape: Vec<u64>,
}

impl Source {
    pub fn entries(&self) -> Vec<Entry> {
        match self {
            Source::Plain(ck) => ck
                .tensors
                .iter()
                .map(|t| Entry {
                    name: t.name.clone(),
                    dtype: t.dtype,
                    shape: t.shape.clone(),
                })
                .collect(),
            Source::Tp { ranks, plan } => ranks[0]
                .tensors
                .iter()
                .map(|t| {
                    let mut shape = t.shape.clone();
                    if let Some(Some(s)) = plan.tensors.get(&t.name) {
                        shape[s.dim] = full_len(s, shape[s.dim], plan.tp);
                    }
                    Entry {
                        name: t.name.clone(),
                        dtype: t.dtype,
                        shape,
                    }
                })
                .collect(),
        }
    }

    pub fn release(&self, i: usize) {
        match self {
            Source::Plain(ck) => ck.release(&ck.tensors[i]),
            Source::Tp { ranks, .. } => {
                let name = &ranks[0].tensors[i].name;
                for ck in ranks {
                    if let Some(t) = ck.find(name) {
                        ck.release(t);
                    }
                }
            }
        }
    }

    /// Full tensor `i` as contiguous bytes.
    pub fn read_full(&self, i: usize) -> Result<Cow<'_, [u8]>> {
        match self {
            Source::Plain(ck) => ck.read(&ck.tensors[i]),
            Source::Tp { ranks, plan } => {
                let t0 = &ranks[0].tensors[i];
                let split = plan.tensors.get(&t0.name).cloned().flatten();
                let parts: Vec<Cow<[u8]>> = ranks
                    .iter()
                    .map(|ck| {
                        let t = ck.find(&t0.name).ok_or_else(|| {
                            anyhow!("{} missing in {}", t0.name, ck.root.display())
                        })?;
                        if t.dtype != t0.dtype || t.shape != t0.shape {
                            bail!("{}: rank shards disagree on dtype/shape", t0.name);
                        }
                        ck.read(t)
                    })
                    .collect::<Result<_>>()?;
                match split {
                    None => {
                        for (r, p) in parts.iter().enumerate().skip(1) {
                            if p[..] != parts[0][..] {
                                bail!(
                                    "{}: replicated tensor differs between rank 0 and rank {r}",
                                    t0.name
                                );
                            }
                        }
                        Ok(parts.into_iter().next().unwrap())
                    }
                    Some(s) => {
                        let tp = ranks.len();
                        let mut full_shape = t0.shape.clone();
                        full_shape[s.dim] = full_len(&s, t0.shape[s.dim], tp);
                        let es = t0.dtype.size();
                        let expect = shard_len(full_shape[s.dim], &s, tp, &t0.name)?;
                        if expect != t0.shape[s.dim] {
                            bail!(
                                "{}: shard dim {} is {} but the plan implies {expect}",
                                t0.name,
                                s.dim,
                                t0.shape[s.dim]
                            );
                        }
                        let mut buf = vec![0u8; numel(&full_shape) as usize * es];
                        for (r, p) in parts.iter().enumerate() {
                            let mut pos = 0usize;
                            for g in shard_segments(&full_shape, es, &s, tp, r, &t0.name)? {
                                match g {
                                    Seg::Data {
                                        off,
                                        len,
                                        primary: true,
                                    } => buf
                                        .get_mut(off..off.saturating_add(len))
                                        .zip(p.get(pos..pos.saturating_add(len)))
                                        .map(|(d, s)| d.copy_from_slice(s))
                                        .ok_or_else(|| {
                                            anyhow!("{}: shard layout out of bounds", t0.name)
                                        })?,
                                    Seg::Data {
                                        off,
                                        len,
                                        primary: false,
                                    } => {
                                        // ranks are visited in order, so the primary copy is already in buf
                                        let (Some(d), Some(s)) = (
                                            buf.get(off..off.saturating_add(len)),
                                            p.get(pos..pos.saturating_add(len)),
                                        ) else {
                                            bail!("{}: shard layout out of bounds", t0.name);
                                        };
                                        if d != s {
                                            bail!(
                                                "{}: replicated kv head differs on tp rank {r}",
                                                t0.name
                                            );
                                        }
                                    }
                                    Seg::Pad { .. } => {}
                                }
                                pos += g.bytes();
                            }
                        }
                        Ok(Cow::Owned(buf))
                    }
                }
            }
        }
    }
}

pub fn rank_dirs(dir: &Path) -> Result<Vec<PathBuf>> {
    let mut v: Vec<PathBuf> = std::fs::read_dir(dir)?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| {
            p.is_dir()
                && p.file_name()
                    .and_then(|n| n.to_str())
                    .is_some_and(|n| n.starts_with("tp_rank_"))
        })
        .collect();
    v.sort();
    Ok(v)
}

/// Open a directory produced by `split` (tp_rank_XX/ + tp_plan.json), or with explicit rules.
pub fn open_tp(dir: &Path, rules: Option<&Rules>) -> Result<Source> {
    let dirs = rank_dirs(dir)?;
    if dirs.is_empty() {
        bail!("{}: no tp_rank_* directories", dir.display());
    }
    let ranks: Vec<Checkpoint> = dirs
        .iter()
        .map(|d| Checkpoint::open(d))
        .collect::<Result<_>>()?;
    let tp = ranks.len();
    let plan_path = dir.join("tp_plan.json");
    let plan = if let Some(r) = rules {
        // foreign TP layout: infer the full shape from the shard shape and the rule
        let mut tensors = BTreeMap::new();
        for t in &ranks[0].tensors {
            let s = r.rule_split(&t.name)?;
            if let Some(s) = &s
                && s.dim >= t.shape.len()
            {
                bail!(
                    "{}: split dim {} out of range for shape {:?}",
                    t.name,
                    s.dim,
                    t.shape
                );
            }
            let s = s
                .map(|s| full_split_from_shard(&s, t.shape[s.dim], tp))
                .transpose()
                .with_context(|| format!("tensor {}", t.name))?;
            tensors.insert(t.name.clone(), s);
        }
        TpPlan { tp, tensors }
    } else if plan_path.is_file() {
        let p: TpPlan = serde_json::from_slice(&std::fs::read(&plan_path)?)
            .with_context(|| format!("parsing {}", plan_path.display()))?;
        if p.tp != tp {
            bail!("tp_plan.json says tp={} but found {tp} rank dirs", p.tp);
        }
        p
    } else {
        bail!("{}: no tp_plan.json; pass --rules", dir.display());
    };
    for t in &ranks[0].tensors {
        if let Some(Some(s)) = plan.tensors.get(&t.name) {
            check_split(s, &t.shape, t.dtype, tp, &t.name)?;
        }
    }
    Ok(Source::Tp { ranks, plan })
}

/// A rule resolved against a *shard* shape -> the split of the full tensor.
fn full_split_from_shard(s: &Split, shard: u64, tp: usize) -> Result<Split> {
    let mut f = s.clone();
    if s.pad_multiple.is_some() {
        f.orig_len = Some(shard * tp as u64); // vocab size unknown: keep the padded rows
        return Ok(f);
    }
    if s.sections.is_empty() {
        return Ok(f);
    }
    // per-rank rows of section i = heads_i*hd/tp, or hd when replicated -> solve for hd
    let tp64 = tp as u64;
    let mut units_num = 0u64; // sum of per-rank head counts, in units of 1/tp
    for sec in &s.sections {
        if sec.heads.is_multiple_of(tp64) {
            units_num += sec.heads;
        } else if sec.replicate_kv && tp64.is_multiple_of(sec.heads) {
            units_num += tp64; // one head per rank
        } else {
            bail!("{} heads cannot be split over tp={tp}", sec.heads);
        }
    }
    // shard = hd * units_num / tp
    if units_num == 0 {
        bail!("the rule's sections have no heads");
    }
    if !(shard * tp64).is_multiple_of(units_num) {
        bail!("shard length {shard} does not fit the rule's head layout");
    }
    let hd = shard * tp64 / units_num;
    for sec in &mut f.sections {
        sec.rows = sec.heads * hd;
    }
    Ok(f)
}

/// Split every tensor of `src` into `tp` rank files, reading each tensor once.
pub fn split(
    src: &Source,
    rules: &Rules,
    tp: usize,
    out: &Path,
    cast: &Option<CastSpec>,
) -> Result<()> {
    if tp == 0 {
        bail!("--tp must be >= 1");
    }
    let entries = src.entries();
    let mut plan = TpPlan {
        tp,
        tensors: BTreeMap::new(),
    };
    let mut splits: Vec<Option<Split>> = Vec::new();
    let mut decl = Vec::new();
    for e in &entries {
        let s = rules.lookup(&e.name, &e.shape)?;
        if let Some(s) = &s {
            for r in 0..tp {
                shard_segments(&e.shape, e.dtype.size(), s, tp, r, &e.name)
                    .with_context(|| format!("tensor {}", e.name))?;
            }
        }
        decl.push(OutTensor {
            name: e.name.clone(),
            dtype: CastSpec::target(cast, &e.name, e.dtype),
            shape: shard_shape(&e.shape, s.as_ref(), tp, &e.name)?,
        });
        plan.tensors.insert(e.name.clone(), s.clone());
        splits.push(s);
    }
    std::fs::create_dir_all(out)?;
    let mut md = BTreeMap::new();
    md.insert("format".to_string(), "pt".to_string());
    let mut writers = Vec::new();
    for r in 0..tp {
        let d = out.join(format!("tp_rank_{r:02}"));
        std::fs::create_dir_all(&d)?;
        let p = d.join("model.safetensors");
        let f = File::create(&p).with_context(|| format!("creating {}", p.display()))?;
        md.insert("tp_rank".into(), r.to_string());
        md.insert("tp_size".into(), tp.to_string());
        writers.push(StWriter::new(
            BufWriter::with_capacity(4 << 20, f),
            &decl,
            &md,
        )?);
    }
    for (i, e) in entries.iter().enumerate() {
        let full = src.read_full(i)?;
        let dst = decl[i].dtype;
        let es = e.dtype.size();
        for (r, w) in writers.iter_mut().enumerate() {
            let mut sink = |b: &[u8]| w.write_part(b);
            let mut cs = crate::cast::CastSink::new(e.dtype, dst, &mut sink);
            match &splits[i] {
                None => cs.write(&full)?,
                Some(s) => {
                    let zeros = vec![0u8; 1 << 16];
                    for g in shard_segments(&e.shape, es, s, tp, r, &e.name)? {
                        match g {
                            Seg::Data { off, len, .. } => cs.write(&full[off..off + len])?,
                            Seg::Pad { mut len } => {
                                // zero bytes are zero in every float format
                                while len > 0 {
                                    let n = len.min(zeros.len());
                                    cs.write(&zeros[..n])?;
                                    len -= n;
                                }
                            }
                        }
                    }
                }
            }
            cs.finish()?;
            w.end_tensor()?;
        }
        drop(full);
        src.release(i);
    }
    for w in writers {
        w.finish()?;
    }
    std::fs::write(
        out.join("tp_plan.json"),
        serde_json::to_string_pretty(&plan)? + "\n",
    )?;
    let n_split = splits.iter().filter(|s| s.is_some()).count();
    eprintln!(
        "split {} tensors ({} sharded, {} replicated) into {tp} TP ranks under {}",
        entries.len(),
        n_split,
        entries.len() - n_split,
        out.display()
    );
    Ok(())
}

/// Rules that reproduce a stored plan (for re-splitting a TP dir to another size).
/// Section head counts and padding multiples are kept; rows and padding follow the new tp.
pub fn rules_from_plan(plan: &TpPlan) -> Rules {
    let mut r = Rules {
        default: "error".into(),
        config: BTreeMap::new(),
        rules: vec![],
    };
    for (name, s) in &plan.tensors {
        let pattern = name.replace(['*', '?'], "?");
        r.rules.push(match s {
            None => RuleSpec {
                pattern,
                replicate: true,
                ..Default::default()
            },
            Some(s) => RuleSpec {
                pattern,
                dim: Some(s.dim),
                parts: s.sections.is_empty().then_some(s.parts),
                sections: (!s.sections.is_empty()).then(|| {
                    s.sections
                        .iter()
                        .map(|x| SectionSpec {
                            heads: (!x.replicate_kv).then_some(Num::Int(x.heads)),
                            kv_heads: x.replicate_kv.then_some(Num::Int(x.heads)),
                        })
                        .collect()
                }),
                pad_multiple: s.pad_multiple.map(Num::Int),
                ..Default::default()
            },
        });
    }
    r
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn ranges_fused() {
        // shape [2, 6], dim 1, parts 3 (q,k,v of width 2), tp 2: rank 0 gets cols {0,2,4}, rank 1 {1,3,5}
        let s = Split::simple(1, 3);
        let r0 = shard_ranges(&[2, 6], 1, &s, 2, 0).unwrap();
        assert_eq!(r0, vec![(0, 1), (2, 1), (4, 1), (6, 1), (8, 1), (10, 1)]);
        let r1 = shard_ranges(&[4, 2], 1, &Split::simple(0, 1), 2, 1).unwrap();
        assert_eq!(r1, vec![(4, 4)]);
        assert!(shard_ranges(&[3, 2], 1, &Split::simple(0, 1), 2, 0).is_err());
    }

    fn gqa_rules() -> Rules {
        serde_yaml::from_str(
            r#"
config: {h: 4, kv: 2}
rules:
  - {pattern: "k", dim: 0, kv_heads: kv}
  - {pattern: "qkv", dim: 0, sections: [{heads: h}, {kv_heads: kv}, {kv_heads: kv}]}
  - {pattern: "emb", dim: 0, pad_multiple: 2}
"#,
        )
        .unwrap()
    }

    #[test]
    fn kv_replication() {
        let r = gqa_rules();
        // k: 2 kv heads of dim 1 -> rows [kv0, kv1]; tp 4 -> ranks 0,1 get kv0, ranks 2,3 get kv1
        let s = r.lookup("k", &[2, 3]).unwrap().unwrap();
        let segs: Vec<Vec<Seg>> = (0..4)
            .map(|q| shard_segments(&[2, 3], 1, &s, 4, q, "k").unwrap())
            .collect();
        assert_eq!(
            segs[0],
            vec![Seg::Data {
                off: 0,
                len: 3,
                primary: true
            }]
        );
        assert_eq!(
            segs[1],
            vec![Seg::Data {
                off: 0,
                len: 3,
                primary: false
            }]
        );
        assert_eq!(
            segs[2],
            vec![Seg::Data {
                off: 3,
                len: 3,
                primary: true
            }]
        );
        assert_eq!(shard_len(2, &s, 4, "k").unwrap(), 1);
        // fused qkv with head_dim 1: rows q0..q3 k0 k1 v0 v1; tp 2: rank 1 = q2 q3 k1 v1
        let s = r.lookup("qkv", &[8, 1]).unwrap().unwrap();
        let r1 = shard_ranges(&[8, 1], 1, &s, 2, 1).unwrap();
        assert_eq!(r1, vec![(2, 2), (5, 1), (7, 1)]);
        // foreign merge: a tp=4 shard of qkv has 1 q + 1 k + 1 v row -> full rows 4 + 2 + 2
        let rs = r.rule_split("qkv").unwrap().unwrap();
        let f = full_split_from_shard(&rs, 3, 4).unwrap();
        assert_eq!(
            f.sections.iter().map(|x| x.rows).collect::<Vec<_>>(),
            vec![4, 2, 2]
        );
    }

    #[test]
    fn vocab_padding() {
        let r = gqa_rules();
        // 5 rows, pad multiple 2, tp 2 -> padded to 8, 4 per rank; rank 1 = rows 4 + 3 pad rows
        let s = r.lookup("emb", &[5, 2]).unwrap().unwrap();
        assert_eq!(s.orig_len, Some(5));
        assert_eq!(shard_len(5, &s, 2, "emb").unwrap(), 4);
        let r1 = shard_segments(&[5, 2], 1, &s, 2, 1, "emb").unwrap();
        assert_eq!(
            r1,
            vec![
                Seg::Data {
                    off: 8,
                    len: 2,
                    primary: true
                },
                Seg::Pad { len: 6 }
            ]
        );
        let r0 = shard_segments(&[5, 2], 1, &s, 2, 0, "emb").unwrap();
        assert_eq!(
            r0,
            vec![Seg::Data {
                off: 0,
                len: 8,
                primary: true
            }]
        );
    }
}
