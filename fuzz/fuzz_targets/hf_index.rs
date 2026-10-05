//! Raw bytes as `model.safetensors.index.json` next to the six committed HF shards.
#![no_main]
#[path = "common.rs"]
mod common;
use common::{FixtureDir, fixture};
use libfuzzer_sys::fuzz_target;
use std::sync::OnceLock;

const FILES: &[(&str, &[u8])] = &[
    fixture!("hf_sharded/model.safetensors.index.json"),
    fixture!("hf_sharded/config.json"),
    fixture!("hf_sharded/model-00001-of-00006.safetensors"),
    fixture!("hf_sharded/model-00002-of-00006.safetensors"),
    fixture!("hf_sharded/model-00003-of-00006.safetensors"),
    fixture!("hf_sharded/model-00004-of-00006.safetensors"),
    fixture!("hf_sharded/model-00005-of-00006.safetensors"),
    fixture!("hf_sharded/model-00006-of-00006.safetensors"),
];

fuzz_target!(|data: &[u8]| {
    static DIR: OnceLock<FixtureDir> = OnceLock::new();
    let dir = DIR.get_or_init(|| FixtureDir::new("hf_index", FILES));
    let (rel, orig) = FILES[0];
    dir.with_file(rel, orig, data, |root| {
        if let Ok(ck) = ckpt::ckpt::Checkpoint::open(&root.join("hf_sharded")) {
            common::exercise(&ck);
        }
    });
});
