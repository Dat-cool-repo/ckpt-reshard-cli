//! PyTorch Distributed Checkpoint (DCP) reader.
//!
//! Layout of a DCP directory written by `torch.distributed.checkpoint.save` with the default
//! `FileSystemWriter`:
//!   .metadata          pickle of `Metadata(state_dict_metadata, planner_data, storage_data, ...)`
//!   __<rank>_<n>.distcp data files. Each tensor *chunk* (a shard of a DTensor / ShardedTensor /
//!                      FSDP param) is stored at (relative_path, offset, length):
//!                      - SerializationFormat.TORCH_SAVE (default): a complete `torch.save` zip
//!                        archive (data.pkl + data/<key>) per chunk;
//!                      - SerializationFormat.SAFETENSORS: one safetensors blob per file, chunk
//!                        tensors keyed by fqn, starting at `offset`.
//!                      Non-tensor values (BytesStorageMetadata) are `torch.save`d python objects.
//!
//! Everything pickled is decoded by the restricted, non-executing reader in `pickle.rs`.

use crate::dtype::{DType, numel};
use crate::pickle::{self, Allow, Node, Pickle, Value};
use crate::zipread;
use anyhow::{Context, Result, anyhow, bail};
use std::collections::HashMap;

#[derive(Clone, Debug)]
pub struct DcpChunkMeta {
    pub offsets: Vec<u64>,
    pub sizes: Vec<u64>,
}

#[derive(Clone, Debug)]
pub struct DcpTensorMeta {
    pub fqn: String,
    pub dtype: DType,
    pub size: Vec<u64>,
    pub chunks: Vec<DcpChunkMeta>,
}

#[derive(Clone, Debug)]
pub struct StorageInfo {
    pub relative_path: String,
    pub offset: u64,
    pub length: u64,
    pub transforms: Vec<String>,
}

#[derive(Debug, Default)]
pub struct DcpMeta {
    pub tensors: Vec<DcpTensorMeta>,
    pub bytes_items: Vec<String>,
    /// (fqn, chunk offsets or None for bytes) -> storage location
    pub storage: HashMap<(String, Option<Vec<u64>>), StorageInfo>,
    pub version: Option<String>,
}

fn to_u64s(v: Vec<i64>) -> Result<Vec<u64>> {
    v.into_iter()
        .map(|x| u64::try_from(x).map_err(|_| anyhow!("negative size/offset")))
        .collect()
}

pub fn parse_metadata(bytes: &[u8]) -> Result<DcpMeta> {
    let pk = pickle::load(bytes, Allow::Checkpoint).context("decoding DCP .metadata")?;
    let root = &pk.root;
    match pk.class_of(root) {
        Some(("torch.distributed.checkpoint.metadata", "Metadata")) => {}
        other => bail!(
            ".metadata root is {:?}, expected torch.distributed.checkpoint.metadata.Metadata",
            other
        ),
    }
    let mut out = DcpMeta {
        version: pk
            .field(root, "version")
            .and_then(|v| pk.as_str(v))
            .map(|s| s.to_string()),
        ..Default::default()
    };
    let sdm = pk
        .field(root, "state_dict_metadata")
        .ok_or_else(|| anyhow!("missing state_dict_metadata"))?;
    for (k, v) in pk
        .dict(sdm)
        .ok_or_else(|| anyhow!("state_dict_metadata is not a dict"))?
    {
        let fqn = pk
            .as_str(k)
            .ok_or_else(|| anyhow!("non-string fqn"))?
            .to_string();
        match pk.class_of(v) {
            Some((_, "BytesStorageMetadata")) => out.bytes_items.push(fqn),
            Some((_, "TensorStorageMetadata")) => out.tensors.push(parse_tensor_meta(&pk, fqn, v)?),
            other => bail!("{fqn}: unsupported storage metadata {:?}", other),
        }
    }
    if let Some(sd) = pk.field(root, "storage_data") {
        for (k, v) in pk
            .dict(sd)
            .ok_or_else(|| anyhow!("storage_data is not a dict"))?
        {
            let fqn = pk
                .field(k, "fqn")
                .and_then(|x| pk.as_str(x))
                .ok_or_else(|| anyhow!("MetadataIndex without fqn"))?
                .to_string();
            let off = match pk.field(k, "offset") {
                None | Some(Value::None) => None,
                Some(o) => Some(to_u64s(
                    pk.int_list(o).ok_or_else(|| anyhow!("bad offset"))?,
                )?),
            };
            let rp = pk
                .field(v, "relative_path")
                .and_then(|x| pk.as_str(x))
                .ok_or_else(|| anyhow!("{fqn}: storage info without relative_path"))?;
            if rp.contains("..") || rp.starts_with('/') || rp.starts_with('\\') || rp.contains(':')
            {
                bail!("{fqn}: suspicious relative_path {rp:?}");
            }
            let offset = pk
                .field(v, "offset")
                .and_then(|x| pk.as_int(x))
                .unwrap_or(0);
            let length = pk
                .field(v, "length")
                .and_then(|x| pk.as_int(x))
                .ok_or_else(|| anyhow!("{fqn}: no length"))?;
            let transforms = match pk.field(v, "transform_descriptors") {
                None | Some(Value::None) => vec![],
                Some(t) => pk
                    .seq(t)
                    .map(|s| {
                        s.iter()
                            .filter_map(|x| pk.as_str(x).map(|y| y.to_string()))
                            .collect()
                    })
                    .unwrap_or_default(),
            };
            out.storage.insert(
                (fqn, off),
                StorageInfo {
                    relative_path: rp.to_string(),
                    offset: offset as u64,
                    length: length as u64,
                    transforms,
                },
            );
        }
    } else {
        bail!("metadata has no storage_data (not written by FileSystemWriter?)");
    }
    Ok(out)
}

fn parse_tensor_meta(pk: &Pickle, fqn: String, v: &Value) -> Result<DcpTensorMeta> {
    let props = pk
        .field(v, "properties")
        .ok_or_else(|| anyhow!("{fqn}: no properties"))?;
    // TensorProperties state is a tuple (dtype, layout, requires_grad, mem_format, pin_memory)
    // in current torch; older versions used a dict state.
    let dtype_v = match pk.obj_state(props) {
        Some(st) => match pk.seq(st) {
            Some(t) if !t.is_empty() && pk.dict(&t[0]).is_none() => Some(&t[0]),
            _ => pk.field(props, "dtype"),
        },
        None => None,
    }
    .ok_or_else(|| anyhow!("{fqn}: cannot find dtype"))?;
    let dtype = match pk.class_of(dtype_v) {
        Some(("torch", name)) => DType::from_torch_name(name)
            .ok_or_else(|| anyhow!("{fqn}: unsupported dtype torch.{name}"))?,
        other => bail!("{fqn}: unexpected dtype value {:?}", other),
    };
    let size = to_u64s(
        pk.int_list(
            pk.field(v, "size")
                .ok_or_else(|| anyhow!("{fqn}: no size"))?,
        )
        .ok_or_else(|| anyhow!("{fqn}: bad size"))?,
    )?;
    let mut chunks = Vec::new();
    let cl = pk
        .field(v, "chunks")
        .and_then(|c| pk.seq(c))
        .ok_or_else(|| anyhow!("{fqn}: no chunks"))?;
    for c in cl {
        let offsets = to_u64s(
            pk.field(c, "offsets")
                .and_then(|x| pk.int_list(x))
                .ok_or_else(|| anyhow!("{fqn}: bad chunk"))?,
        )?;
        let sizes = to_u64s(
            pk.field(c, "sizes")
                .and_then(|x| pk.int_list(x))
                .ok_or_else(|| anyhow!("{fqn}: bad chunk"))?,
        )?;
        if offsets.len() != size.len() || sizes.len() != size.len() {
            bail!("{fqn}: chunk rank mismatch");
        }
        for d in 0..size.len() {
            if offsets[d] + sizes[d] > size[d] {
                bail!(
                    "{fqn}: chunk {:?}+{:?} out of bounds of {:?}",
                    offsets,
                    sizes,
                    size
                );
            }
        }
        chunks.push(DcpChunkMeta { offsets, sizes });
    }
    Ok(DcpTensorMeta {
        fqn,
        dtype,
        size,
        chunks,
    })
}

/// A decoded chunk: raw storage bytes + view parameters (in elements).
pub struct ChunkView<'a> {
    pub storage: &'a [u8],
    pub storage_offset: u64,
    pub sizes: Vec<u64>,
    pub strides: Vec<u64>,
    pub dtype: DType,
}

impl<'a> ChunkView<'a> {
    pub fn is_contiguous(&self) -> bool {
        let mut expect = 1u64;
        for d in (0..self.sizes.len()).rev() {
            if self.sizes[d] != 1 && self.strides[d] != expect {
                return false;
            }
            expect *= self.sizes[d];
        }
        true
    }
    /// Contiguous byte slice of the whole chunk, if the view is contiguous.
    pub fn contiguous_bytes(&self) -> Option<&'a [u8]> {
        if !self.is_contiguous() {
            return None;
        }
        let es = self.dtype.size() as u64;
        let s = self.storage_offset * es;
        let e = s + numel(&self.sizes) * es;
        self.storage.get(s as usize..e as usize)
    }
}

/// Decode the tensor chunk stored at `si` inside `file`.
pub fn read_chunk<'a>(file: &'a [u8], si: &StorageInfo, fqn: &str) -> Result<ChunkView<'a>> {
    if !si.transforms.is_empty() {
        bail!(
            "{fqn}: DCP stream transforms {:?} (e.g. compression) are not supported",
            si.transforms
        );
    }
    let start = si.offset as usize;
    if start > file.len() {
        bail!("{fqn}: offset beyond end of {}", si.relative_path);
    }
    let rest = &file[start..];
    if zipread::is_zip(rest) {
        let end = start
            .checked_add(si.length as usize)
            .filter(|&e| e <= file.len())
            .ok_or_else(|| anyhow!("{fqn}: chunk out of bounds"))?;
        return read_torch_save_tensor(&file[start..end])
            .with_context(|| format!("{fqn}: decoding torch.save chunk"));
    }
    // SerializationFormat.SAFETENSORS: a safetensors blob starting at `offset`.
    let h = crate::safetensors::parse_header(rest)
        .with_context(|| format!("{fqn}: chunk is neither a torch.save zip nor safetensors"))?;
    let e = h
        .entries
        .iter()
        .find(|e| e.name == fqn)
        .ok_or_else(|| anyhow!("{fqn}: not found in safetensors chunk file"))?;
    let n = e.shape.len();
    let mut strides = vec![1u64; n];
    for d in (0..n.saturating_sub(1)).rev() {
        strides[d] = strides[d + 1] * e.shape[d + 1];
    }
    Ok(ChunkView {
        storage: &rest[e.start as usize..e.end as usize],
        storage_offset: 0,
        sizes: e.shape.clone(),
        strides,
        dtype: e.dtype,
    })
}

/// Decode a `torch.save` archive holding a single tensor.
pub fn read_torch_save_tensor(zip: &[u8]) -> Result<ChunkView<'_>> {
    let entries = zipread::entries(zip)?;
    let pkl = entries
        .iter()
        .find(|e| e.name.ends_with("/data.pkl") || e.name == "data.pkl")
        .ok_or_else(|| anyhow!("no data.pkl"))?;
    let prefix = &pkl.name[..pkl.name.len() - "data.pkl".len()];
    let pk = pickle::load(pkl.checked_data()?, Allow::Checkpoint)?;
    let mut root = pk.root.clone();
    if let Some(("torch._utils", "_rebuild_parameter")) = pk.class_of(&root) {
        root = pk
            .obj_args(&root)
            .and_then(|a| a.first().cloned())
            .ok_or_else(|| anyhow!("bad _rebuild_parameter"))?;
    }
    match pk.class_of(&root) {
        Some(("torch._utils", "_rebuild_tensor_v2" | "_rebuild_tensor")) => {}
        other => bail!("expected a tensor record, found {:?}", other),
    }
    let args = pk
        .obj_args(&root)
        .ok_or_else(|| anyhow!("bad tensor record"))?;
    if args.len() < 4 {
        bail!("bad _rebuild_tensor args");
    }
    let Some(Node::Persistent(pid)) = pk.node(&args[0]) else {
        bail!("tensor storage is not a persistent id")
    };
    let pid = pk.seq(pid).ok_or_else(|| anyhow!("bad persistent id"))?;
    if pid.len() < 5 || pk.as_str(&pid[0]) != Some("storage") {
        bail!("unsupported persistent id");
    }
    let dtype = match pk.class_of(&pid[1]) {
        Some(("torch", n)) => DType::from_storage_name(n)
            .or_else(|| DType::from_torch_name(n))
            .ok_or_else(|| anyhow!("unsupported storage type torch.{n}"))?,
        other => bail!("unexpected storage type {:?}", other),
    };
    let key = pk
        .as_str(&pid[2])
        .ok_or_else(|| anyhow!("bad storage key"))?;
    let storage_numel = pk
        .as_int(&pid[4])
        .ok_or_else(|| anyhow!("bad storage numel"))? as u64;
    let data_name = format!("{prefix}data/{key}");
    let data = entries
        .iter()
        .find(|e| e.name == data_name)
        .ok_or_else(|| anyhow!("missing {data_name}"))?
        .checked_data()?;
    if (data.len() as u64) < storage_numel * dtype.size() as u64 {
        bail!("storage {key} truncated");
    }
    let storage_offset = pk
        .as_int(&args[1])
        .ok_or_else(|| anyhow!("bad storage offset"))? as u64;
    let sizes = to_u64s(pk.int_list(&args[2]).ok_or_else(|| anyhow!("bad sizes"))?)?;
    let strides = to_u64s(
        pk.int_list(&args[3])
            .ok_or_else(|| anyhow!("bad strides"))?,
    )?;
    if sizes.len() != strides.len() {
        bail!("sizes/strides rank mismatch");
    }
    // bounds check: max linear index must be inside storage
    if numel(&sizes) > 0 {
        let max_idx: u64 = storage_offset
            + sizes
                .iter()
                .zip(&strides)
                .map(|(s, st)| (s - 1) * st)
                .sum::<u64>();
        if max_idx >= storage_numel {
            bail!("tensor view exceeds its storage");
        }
    }
    Ok(ChunkView {
        storage: &data[..(storage_numel as usize) * dtype.size()],
        storage_offset,
        sizes,
        strides,
        dtype,
    })
}

/// Decode a non-tensor (BytesStorageMetadata) item into JSON for display.
pub fn read_bytes_item(file: &[u8], si: &StorageInfo, allow: Allow) -> Result<serde_json::Value> {
    let s = si.offset as usize;
    let e = s + si.length as usize;
    let b = file
        .get(s..e)
        .ok_or_else(|| anyhow!("bytes item out of bounds"))?;
    if !zipread::is_zip(b) {
        return Ok(serde_json::json!(format!("<{} raw bytes>", b.len())));
    }
    let entries = zipread::entries(b)?;
    let pkl = entries
        .iter()
        .find(|e| e.name.ends_with("data.pkl"))
        .ok_or_else(|| anyhow!("no data.pkl"))?;
    let pk = pickle::load(pkl.checked_data()?, allow)?;
    Ok(pk.to_json(&pk.root))
}
