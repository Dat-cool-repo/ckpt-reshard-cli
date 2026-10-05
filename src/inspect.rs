//! `ckpt inspect`: format detection, tensor list, per-shard summary.

use crate::ckpt::{Checkpoint, Format, Loc, Place};
use crate::dcp;
use crate::dtype::{human_bytes, human_count};
use crate::pickle::Allow;
use crate::writer::glob_match;
use anyhow::Result;
use serde_json::json;
use std::collections::BTreeSet;

pub struct InspectOpts {
    pub filter: Vec<String>,
    pub tensors: bool,
    pub chunks: bool,
}

/// Describe how a tensor is split: e.g. "4 chunks, grid 2x2".
fn grid(t: &crate::ckpt::TensorInfo) -> Option<(usize, Vec<usize>)> {
    let nd = t.shape.len();
    let offs: Vec<Vec<u64>> = match &t.loc {
        Loc::Dcp { chunks } => chunks.iter().map(|c| c.meta.offsets.clone()).collect(),
        Loc::Views { parts } => parts
            .iter()
            .filter_map(|p| match &p.place {
                Place::Block(o) => Some(o.clone()),
                Place::Linear(_) => None,
            })
            .collect(),
        _ => return None,
    };
    let g: Vec<usize> = (0..nd)
        .map(|d| offs.iter().map(|o| o[d]).collect::<BTreeSet<_>>().len())
        .collect();
    Some((offs.len(), g))
}

fn bytes_allow(ck: &Checkpoint) -> Allow {
    if ck.format == Format::MegatronDist {
        Allow::Megatron
    } else {
        Allow::Checkpoint
    }
}

pub fn to_json(ck: &Checkpoint, o: &InspectOpts) -> Result<serde_json::Value> {
    let keep = |n: &str| o.filter.is_empty() || o.filter.iter().any(|p| glob_match(p, n));
    let mut tensors = Vec::new();
    for t in ck.tensors.iter().filter(|t| keep(&t.name)) {
        let mut j = json!({
            "name": t.name, "dtype": t.dtype.st_name(), "shape": t.shape,
            "numel": t.numel(), "bytes": t.nbytes(),
        });
        match &t.loc {
            Loc::Contig { file, start, end } => {
                j["file"] = json!(ck.files[*file].path.file_name().unwrap().to_string_lossy());
                j["data_offsets"] = json!([start, end]);
            }
            Loc::Dcp { chunks } => {
                let (_, g) = grid(t).unwrap();
                j["grid"] = json!(g);
                j["chunks"] = chunks
                    .iter()
                    .map(|c| {
                        json!({"offsets": c.meta.offsets, "sizes": c.meta.sizes,
                               "file": c.info.relative_path, "offset": c.info.offset, "length": c.info.length})
                    })
                    .collect();
            }
            Loc::Views { parts } => {
                if let Some((_, g)) = grid(t) {
                    j["grid"] = json!(g);
                }
                j["parts"] = parts
                    .iter()
                    .map(|p| {
                        let f = ck.files[p.file]
                            .path
                            .strip_prefix(&ck.root)
                            .map(|x| x.display().to_string())
                            .unwrap_or_default();
                        let place = match &p.place {
                            Place::Block(o) => json!({"offsets": o}),
                            Place::Linear(o) => json!({"flat_offset": o}),
                        };
                        json!({"file": f, "sizes": p.sizes, "strides": p.strides,
                               "storage_offset": p.storage_offset, "place": place})
                    })
                    .collect();
            }
            Loc::Derived { pieces } => {
                j["from"] = pieces
                    .iter()
                    .map(|p| json!({"tensor": ck.base[p.base].name, "start": p.start, "size": p.size}))
                    .collect();
            }
        }
        tensors.push(j);
    }
    let mut bytes_items = serde_json::Map::new();
    for b in ck.bytes_items.iter().filter(|b| keep(&b.name)) {
        let v = dcp::read_bytes_item(&ck.files[b.file].map, &b.info, bytes_allow(ck))
            .unwrap_or_else(|e| json!(format!("<undecodable: {e}>")));
        bytes_items.insert(b.name.clone(), v);
    }
    let shards: Vec<_> = ck
        .shard_summary()
        .into_iter()
        .map(|(f, n, b, p)| json!({"file": f, "entries": n, "bytes": b, "params": p}))
        .collect();
    let mut dtypes = std::collections::BTreeMap::<String, u64>::new();
    for t in &ck.tensors {
        *dtypes.entry(t.dtype.st_name().to_string()).or_default() += t.numel();
    }
    let mut out = json!({
        "path": ck.path.display().to_string(),
        "format": ck.format.name(),
        "num_tensors": ck.tensors.len(),
        "total_params": ck.total_params(),
        "total_bytes": ck.total_bytes(),
        "params_by_dtype": dtypes,
        "files": shards,
        "metadata": ck.metadata,
        "non_tensor_items": bytes_items,
        "info": ck.info,
    });
    if !ck.scalars.is_empty() {
        let m: serde_json::Map<String, serde_json::Value> = ck
            .scalars
            .iter()
            .filter(|(k, _)| keep(k))
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect();
        out["scalars"] = serde_json::Value::Object(m);
    }
    if let Some(c) = &ck.hf_config {
        out["hf_config"] = c.clone();
    }
    if let Some(v) = &ck.dcp_version {
        out["dcp_version"] = json!(v);
    }
    if o.tensors {
        out["tensors"] = json!(tensors);
    }
    Ok(out)
}

pub fn print_human(ck: &Checkpoint, o: &InspectOpts) -> Result<()> {
    let keep = |n: &str| o.filter.is_empty() || o.filter.iter().any(|p| glob_match(p, n));
    println!("path:    {}", ck.path.display());
    print!("format:  {}", ck.format.name());
    if let Some(v) = &ck.dcp_version {
        print!(" (DCP metadata version {v})");
    }
    println!();
    println!(
        "tensors: {}   params: {} ({})   bytes: {} ({})",
        ck.tensors.len(),
        human_count(ck.total_params()),
        ck.total_params(),
        human_bytes(ck.total_bytes()),
        ck.total_bytes()
    );
    let mut dtypes = std::collections::BTreeMap::<&str, u64>::new();
    for t in &ck.tensors {
        *dtypes.entry(t.dtype.st_name()).or_default() += t.numel();
    }
    println!(
        "dtypes:  {}",
        dtypes
            .iter()
            .map(|(k, v)| format!("{k}={}", human_count(*v)))
            .collect::<Vec<_>>()
            .join(", ")
    );
    if !ck.metadata.is_empty() {
        println!("metadata: {}", serde_json::to_string(&ck.metadata)?);
    }
    for (k, v) in &ck.info {
        let s = v.to_string();
        if k == "args" {
            println!("{k}:");
            if let Some(m) = v.as_object() {
                for (ak, av) in m {
                    println!("  {ak:<40} {av}");
                }
            }
        } else {
            println!("{k}: {}", s.trim_matches('"'));
        }
    }
    println!("files ({}):", ck.files.len());
    let entry_word = match ck.format {
        Format::Dcp | Format::MegatronDist => "chunks",
        Format::Megatron | Format::DeepSpeed | Format::TorchSave => "parts",
        _ => "tensors",
    };
    for (f, n, b, p) in ck.shard_summary() {
        println!(
            "  {f:<40} {entry_word} {n:>6}   params {:>10}   {:>12}",
            human_count(p),
            human_bytes(b)
        );
    }
    if !ck.bytes_items.is_empty() {
        println!("non-tensor items ({}):", ck.bytes_items.len());
        for b in ck.bytes_items.iter().filter(|b| keep(&b.name)) {
            let v = dcp::read_bytes_item(&ck.files[b.file].map, &b.info, bytes_allow(ck))
                .map(|v| {
                    let s = v.to_string();
                    if s.chars().count() > 100 {
                        format!("{}...", s.chars().take(100).collect::<String>())
                    } else {
                        s
                    }
                })
                .unwrap_or_else(|e| format!("<undecodable: {e}>"));
            println!("  {:<60} = {v}", b.name);
        }
    }
    if o.tensors {
        println!("tensors:");
        for t in ck.tensors.iter().filter(|t| keep(&t.name)) {
            let shape = format!("{:?}", t.shape);
            let loc = match &t.loc {
                Loc::Contig { file, .. } => {
                    if ck.files.len() > 1 {
                        ck.files[*file]
                            .path
                            .file_name()
                            .unwrap()
                            .to_string_lossy()
                            .to_string()
                    } else {
                        String::new()
                    }
                }
                Loc::Derived { pieces } => {
                    let b = &ck.base[pieces[0].base];
                    if pieces.len() == 1 && b.name == t.name {
                        String::new()
                    } else {
                        format!(
                            "<- {}{}",
                            b.name,
                            if pieces.len() > 1 {
                                format!(" ({} pieces)", pieces.len())
                            } else {
                                String::new()
                            }
                        )
                    }
                }
                Loc::Views { parts } if grid(t).is_none_or(|(n, _)| n != parts.len()) => {
                    format!("{} flat part(s)", parts.len())
                }
                Loc::Dcp { .. } | Loc::Views { .. } => {
                    let (n, g) = grid(t).unwrap();
                    let gs = g
                        .iter()
                        .map(|x| x.to_string())
                        .collect::<Vec<_>>()
                        .join("x");
                    let word = if matches!(t.loc, Loc::Dcp { .. }) {
                        "chunk"
                    } else {
                        "part"
                    };
                    if n == 1 && matches!(t.loc, Loc::Views { .. }) {
                        String::new()
                    } else {
                        format!(
                            "{n} {word}(s), grid {}",
                            if gs.is_empty() { "scalar".into() } else { gs }
                        )
                    }
                }
            };
            println!(
                "  {:<60} {:<6} {:<18} {:>10}  {loc}",
                t.name,
                t.dtype.st_name(),
                shape,
                human_count(t.numel())
            );
            if o.chunks
                && let Loc::Dcp { chunks } = &t.loc
            {
                for c in chunks {
                    println!(
                        "      offsets {:?} sizes {:?} -> {} @{} (+{})",
                        c.meta.offsets,
                        c.meta.sizes,
                        c.info.relative_path,
                        c.info.offset,
                        c.info.length
                    );
                }
            }
        }
    }
    Ok(())
}
