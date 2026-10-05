//! A DCP checkpoint directory with one file (the `.metadata` or a `__R_0.distcp` data file) replaced
//! by the fuzz input. The first byte picks the fixture and the file: FSDP2 with torch_save chunks,
//! FSDP2 with safetensors chunks, a 2-D sharded bf16 DCP, and a Megatron `torch_dist` checkpoint
//! (DCP plus layer unstacking and, with the high bit set, the HF mapping).
#![no_main]
#[path = "common.rs"]
mod common;
use ckpt::ckpt::{Checkpoint, OpenOpts};
use common::{FixtureDir, fixture};
use libfuzzer_sys::fuzz_target;
use std::sync::OnceLock;

const FILES: &[(&str, &[u8])] = &[
    fixture!("dcp_fsdp/.metadata"),
    fixture!("dcp_fsdp/__0_0.distcp"),
    fixture!("dcp_fsdp/__1_0.distcp"),
    fixture!("dcp_fsdp/__2_0.distcp"),
    fixture!("dcp_fsdp/__3_0.distcp"),
    fixture!("dcp_fsdp_st/.metadata"),
    fixture!("dcp_fsdp_st/__0_0.distcp"),
    fixture!("dcp_fsdp_st/__1_0.distcp"),
    fixture!("dcp_fsdp_st/__2_0.distcp"),
    fixture!("dcp_fsdp_st/__3_0.distcp"),
    fixture!("dcp_2d/.metadata"),
    fixture!("dcp_2d/__0_0.distcp"),
    fixture!("dcp_2d/__1_0.distcp"),
    fixture!("dcp_2d/__2_0.distcp"),
    fixture!("dcp_2d/__3_0.distcp"),
    fixture!("megatron/torch_dist/iter_0000010/.metadata"),
    fixture!("megatron/torch_dist/iter_0000010/__0_0.distcp"),
    fixture!("megatron/torch_dist/iter_0000010/__1_0.distcp"),
    fixture!("megatron/torch_dist/iter_0000010/__2_0.distcp"),
    fixture!("megatron/torch_dist/iter_0000010/__3_0.distcp"),
    fixture!("megatron/torch_dist/iter_0000010/metadata.json"),
    fixture!("megatron/torch_dist/latest_checkpointed_iteration.txt"),
];

fuzz_target!(|data: &[u8]| {
    static DIR: OnceLock<FixtureDir> = OnceLock::new();
    let dir = DIR.get_or_init(|| FixtureDir::new("dcp_dir", FILES));
    let Some((&sel, body)) = data.split_first() else {
        return;
    };
    let i = (sel & 0x7f) as usize % 20; // one of the .metadata / .distcp files
    let (rel, orig) = FILES[i];
    let ckdir = rel.rsplit_once('/').unwrap().0;
    let hf = i >= 15 && sel & 0x80 != 0;
    dir.with_file(rel, orig, body, |root| {
        let o = OpenOpts {
            megatron_hf: hf.then(|| "auto".to_string()),
            ..Default::default()
        };
        if let Ok(ck) = Checkpoint::open_with(&root.join(ckdir), &o) {
            common::exercise(&ck);
        }
    });
});
