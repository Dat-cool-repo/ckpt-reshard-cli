//! Generic reader for `torch.save` zip archives (`.pt`, `.bin`, Megatron `model_optim_rng.pt`,
//! DeepSpeed `*_states.pt`). The pickle goes through the restricted VM in `pickle.rs`. Nothing is
//! executed, and only the globals of the chosen [`Allow`] list are accepted.
//!
//! The pickle tree is walked and every tensor record is collected under its dotted path
//! (`model.decoder.layers.0.mlp.linear_fc1.weight`, `optimizer.state.3.exp_avg`, ...), together
//! with the byte range of its storage inside the archive and its view (offset, sizes, strides).
//! Small non-tensor leaves (ints, floats, strings) are collected as JSON for `inspect`.

use crate::dtype::{DType, numel};
use crate::pickle::{self, Allow, Node, Pickle, Value};
use crate::zipread;
use anyhow::{Context, Result, anyhow, bail};
use std::collections::HashMap;

/// One tensor found in the pickle tree.
#[derive(Clone, Debug)]
pub struct TsTensor {
    pub path: String,
    pub dtype: DType,
    /// absolute byte range of the storage inside the archive buffer
    pub storage_start: u64,
    pub storage_len: u64,
    /// view, in elements
    pub storage_offset: u64,
    pub sizes: Vec<u64>,
    pub strides: Vec<u64>,
    pub storage_key: String,
}

impl TsTensor {
    pub fn view<'a>(&self, file: &'a [u8]) -> crate::dcp::ChunkView<'a> {
        crate::dcp::ChunkView {
            storage: &file
                [self.storage_start as usize..(self.storage_start + self.storage_len) as usize],
            storage_offset: self.storage_offset,
            sizes: self.sizes.clone(),
            strides: self.strides.clone(),
            dtype: self.dtype,
        }
    }
    pub fn numel(&self) -> u64 {
        numel(&self.sizes)
    }
}

pub struct TorchSave {
    pub pickle: Pickle,
    /// storage key -> (start, len, crc32) inside the archive buffer
    storages: HashMap<String, (usize, usize, u32)>,
    pub entry_count: usize,
}

/// Is `b` a torch.save archive (zip)? Legacy (pre-1.6, non-zip) torch.save files are not supported.
pub fn is_torch_save(b: &[u8]) -> bool {
    zipread::is_zip(b)
}

impl TorchSave {
    /// Parse the archive's `data.pkl` (verifying its CRC with `--verify`) and index its storages.
    pub fn open(file: &[u8], allow: Allow) -> Result<TorchSave> {
        if !zipread::is_zip(file) {
            if file.first() == Some(&0x80) {
                bail!(
                    "legacy (pre torch 1.6, non-zip) torch.save format is not supported; re-save with a newer torch"
                );
            }
            bail!("not a torch.save zip archive");
        }
        let entries = zipread::entries(file)?;
        let pkl = entries
            .iter()
            .find(|e| e.name.ends_with("/data.pkl") || e.name == "data.pkl")
            .ok_or_else(|| anyhow!("torch.save archive has no data.pkl"))?;
        let prefix = &pkl.name[..pkl.name.len() - "data.pkl".len()];
        let pk = pickle::load(pkl.checked_data()?, allow)?;
        let mut storages = HashMap::new();
        let dprefix = format!("{prefix}data/");
        for e in &entries {
            if let Some(k) = e.name.strip_prefix(&dprefix) {
                storages.insert(k.to_string(), (e.offset, e.data.len(), e.crc32));
            }
        }
        Ok(TorchSave {
            pickle: pk,
            storages,
            entry_count: entries.len(),
        })
    }

    /// Check the CRC32 of a storage entry (used with `--verify`).
    pub fn verify_storage(&self, file: &[u8], key: &str) -> Result<()> {
        let &(s, l, crc) = self
            .storages
            .get(key)
            .ok_or_else(|| anyhow!("missing storage {key}"))?;
        let got = crc32fast::hash(&file[s..s + l]);
        if got != crc {
            bail!(
                "CRC32 mismatch in storage data/{key} ({l} bytes at offset {s}): central directory says {crc:#010x}, data hashes to {got:#010x} -- the file is corrupted"
            );
        }
        Ok(())
    }

    /// Decode a tensor record (`_rebuild_tensor_v2`, `_rebuild_parameter`, `_rebuild_from_type_v2`).
    /// Returns None if `v` is not a tensor.
    pub fn tensor_at(&self, v: &Value, path: &str) -> Result<Option<TsTensor>> {
        let pk = &self.pickle;
        let mut v = v.clone();
        for _ in 0..4 {
            match pk.class_of(&v) {
                Some(("torch._utils", "_rebuild_parameter")) => {
                    v = pk
                        .obj_args(&v)
                        .and_then(|a| a.first().cloned())
                        .ok_or_else(|| anyhow!("{path}: bad _rebuild_parameter"))?;
                }
                Some(("torch._tensor", "_rebuild_from_type_v2")) => {
                    // (func, type, args, state): only plain tensor rebuilds are accepted
                    let a = pk
                        .obj_args(&v)
                        .ok_or_else(|| anyhow!("{path}: bad _rebuild_from_type_v2"))?;
                    if a.len() < 3 {
                        bail!("{path}: bad _rebuild_from_type_v2 args");
                    }
                    match pk.class_of(&a[0]) {
                        Some(("torch._utils", "_rebuild_tensor_v2")) => {}
                        other => bail!("{path}: unsupported tensor subclass rebuild {other:?}"),
                    }
                    let inner = pk
                        .seq(&a[2])
                        .ok_or_else(|| anyhow!("{path}: bad rebuild args"))?
                        .to_vec();
                    return self.rebuild_v2(&inner, path).map(Some);
                }
                Some(("torch._utils", "_rebuild_tensor_v2" | "_rebuild_tensor")) => {
                    let a = pk
                        .obj_args(&v)
                        .ok_or_else(|| anyhow!("{path}: bad tensor record"))?
                        .to_vec();
                    return self.rebuild_v2(&a, path).map(Some);
                }
                _ => return Ok(None),
            }
        }
        Ok(None)
    }

    fn rebuild_v2(&self, args: &[Value], path: &str) -> Result<TsTensor> {
        let pk = &self.pickle;
        if args.len() < 4 {
            bail!("{path}: bad _rebuild_tensor args");
        }
        let Some(Node::Persistent(pid)) = pk.node(&args[0]) else {
            bail!("{path}: tensor storage is not a persistent id")
        };
        let pid = pk
            .seq(pid)
            .ok_or_else(|| anyhow!("{path}: bad persistent id"))?;
        if pid.len() < 5 || pk.as_str(&pid[0]) != Some("storage") {
            bail!("{path}: unsupported persistent id");
        }
        let dtype = match pk.class_of(&pid[1]) {
            Some(("torch", n)) => DType::from_storage_name(n)
                .or_else(|| DType::from_torch_name(n))
                .ok_or_else(|| anyhow!("{path}: unsupported storage type torch.{n}"))?,
            other => bail!("{path}: unexpected storage type {:?}", other),
        };
        let key = pk
            .as_str(&pid[2])
            .ok_or_else(|| anyhow!("{path}: bad storage key"))?
            .to_string();
        let storage_numel =
            pk.as_int(&pid[4])
                .filter(|&n| n >= 0)
                .ok_or_else(|| anyhow!("{path}: bad storage numel"))? as u64;
        let &(start, len, _) = self
            .storages
            .get(&key)
            .ok_or_else(|| anyhow!("{path}: storage data/{key} missing from archive"))?;
        let need = storage_numel
            .checked_mul(dtype.size() as u64)
            .ok_or_else(|| anyhow!("{path}: storage size overflow"))?;
        if (len as u64) < need {
            bail!("{path}: storage {key} truncated ({len} < {need} bytes)");
        }
        let storage_offset =
            pk.as_int(&args[1])
                .filter(|&n| n >= 0)
                .ok_or_else(|| anyhow!("{path}: bad storage offset"))? as u64;
        let to_u64 = |v: Vec<i64>| -> Result<Vec<u64>> {
            v.into_iter()
                .map(|x| u64::try_from(x).map_err(|_| anyhow!("{path}: negative size/stride")))
                .collect()
        };
        let sizes = to_u64(
            pk.int_list(&args[2])
                .ok_or_else(|| anyhow!("{path}: bad sizes"))?,
        )?;
        let strides = to_u64(
            pk.int_list(&args[3])
                .ok_or_else(|| anyhow!("{path}: bad strides"))?,
        )?;
        if sizes.len() != strides.len() {
            bail!("{path}: sizes/strides rank mismatch");
        }
        if numel(&sizes) > 0 {
            let mut max_idx = storage_offset;
            for (s, st) in sizes.iter().zip(&strides) {
                max_idx = (s - 1)
                    .checked_mul(*st)
                    .and_then(|x| x.checked_add(max_idx))
                    .ok_or_else(|| anyhow!("{path}: view overflow"))?;
            }
            if max_idx >= storage_numel {
                bail!("{path}: tensor view exceeds its storage");
            }
        }
        Ok(TsTensor {
            path: path.to_string(),
            dtype,
            storage_start: start as u64,
            storage_len: need,
            storage_offset,
            sizes,
            strides,
            storage_key: key,
        })
    }

    /// Look up a key in the root dict (a root that is a 1-element list/tuple wrapping a dict, as in
    /// Megatron's `common_state` sharded object, is unwrapped).
    pub fn root_get(&self, key: &str) -> Option<&Value> {
        let pk = &self.pickle;
        let mut root = &pk.root;
        if let Some(s) = pk.seq(root)
            && s.len() == 1
        {
            root = &s[0];
        }
        pk.dict_get(root, key)
    }

    /// Walk `v` and collect every tensor (dotted path) plus small scalar leaves.
    pub fn collect(&self, v: &Value, prefix: &str, out: &mut Collected) -> Result<()> {
        self.walk(v, prefix, out, 0)
    }

    fn walk(&self, v: &Value, path: &str, out: &mut Collected, depth: usize) -> Result<()> {
        if depth > 64 {
            bail!("{path}: pickle nested too deeply");
        }
        let pk = &self.pickle;
        if let Some(t) = self.tensor_at(v, path)? {
            out.tensors.push(t);
            return Ok(());
        }
        let join = |k: &str| {
            if path.is_empty() {
                k.to_string()
            } else {
                format!("{path}.{k}")
            }
        };
        match v {
            Value::Ref(_) => match pk.node(v) {
                Some(Node::Dict(d)) => {
                    for (k, val) in d {
                        let ks = key_str(pk, k);
                        self.walk(val, &join(&ks), out, depth + 1)?;
                    }
                }
                Some(Node::List(x)) | Some(Node::Tuple(x)) => {
                    if x.len() > 4096 && x.iter().all(|e| !matches!(e, Value::Ref(_))) {
                        out.scalars.push((
                            path.to_string(),
                            serde_json::json!(format!("<{} items>", x.len())),
                        ));
                        return Ok(());
                    }
                    for (i, val) in x.iter().enumerate() {
                        self.walk(val, &join(&i.to_string()), out, depth + 1)?;
                    }
                }
                Some(Node::Object { .. }) => {
                    // objects with a dict state (argparse.Namespace, dataclasses): walk their fields
                    if let Some(st) = pk.obj_state(v)
                        && pk.dict(st).is_some()
                    {
                        return self.walk(st, path, out, depth + 1);
                    }
                    out.scalars.push((path.to_string(), pk.to_json(v)));
                }
                _ => out.scalars.push((path.to_string(), pk.to_json(v))),
            },
            _ => out.scalars.push((path.to_string(), pk.to_json(v))),
        }
        Ok(())
    }
}

pub fn key_str(pk: &Pickle, k: &Value) -> String {
    match k {
        Value::Str(s) => s.to_string(),
        Value::Int(i) => i.to_string(),
        other => pk.to_json(other).to_string(),
    }
}

#[derive(Default)]
pub struct Collected {
    pub tensors: Vec<TsTensor>,
    pub scalars: Vec<(String, serde_json::Value)>,
}

/// Decode a small (non-tensor) torch.save blob to JSON (e.g. Megatron's `common_state` item).
pub fn blob_to_json(b: &[u8], allow: Allow) -> Result<serde_json::Value> {
    let ts = TorchSave::open(b, allow).context("decoding torch.save blob")?;
    Ok(ts.pickle.to_json(&ts.pickle.root))
}
