//! Raw bytes as a safetensors file: header parse, then the file opened and every tensor read; also
//! the bytes as a safetensors-format DCP chunk.
#![no_main]
#[path = "common.rs"]
mod common;
use ckpt::ckpt::{Checkpoint, Format};
use libfuzzer_sys::fuzz_target;
use std::path::Path;

fuzz_target!(|data: &[u8]| {
    if data.is_empty() {
        return;
    }
    if let Ok(h) = ckpt::safetensors::parse_header(data) {
        for e in &h.entries {
            assert!(e.start <= e.end && e.end <= data.len() as u64);
        }
        if let Some(e) = h.entries.first() {
            let si = ckpt::dcp::StorageInfo {
                relative_path: "x".into(),
                offset: 0,
                length: data.len() as u64,
                transforms: vec![],
            };
            let _ = ckpt::dcp::read_chunk(data, &si, &e.name);
        }
    }
    let m = common::mapped(data, "fuzz.safetensors");
    if let Ok(ck) = Checkpoint::open_safetensors_maps(Path::new("."), vec![m], Format::Safetensors)
    {
        common::exercise(&ck);
    }
});
