//! A Megatron-LM legacy TP2/PP2 checkpoint with one `model_optim_rng.pt` rank file replaced by the
//! fuzz input: the restricted pickle path with the Megatron allowlist, the TP/PP merge, `--vocab-size`
//! unpadding (bit 6 of the first byte) and the HF Llama/Qwen2 mapping (bit 7).
#![no_main]
#[path = "common.rs"]
mod common;
use ckpt::ckpt::{Checkpoint, OpenOpts};
use common::{FixtureDir, fixture};
use libfuzzer_sys::fuzz_target;
use std::sync::OnceLock;

const FILES: &[(&str, &[u8])] = &[
    fixture!("megatron/legacy/iter_0000010/mp_rank_00_000/model_optim_rng.pt"),
    fixture!("megatron/legacy/iter_0000010/mp_rank_00_001/model_optim_rng.pt"),
    fixture!("megatron/legacy/iter_0000010/mp_rank_01_000/model_optim_rng.pt"),
    fixture!("megatron/legacy/iter_0000010/mp_rank_01_001/model_optim_rng.pt"),
    fixture!("megatron/legacy/latest_checkpointed_iteration.txt"),
];

fuzz_target!(|data: &[u8]| {
    static DIR: OnceLock<FixtureDir> = OnceLock::new();
    let dir = DIR.get_or_init(|| FixtureDir::new("megatron", FILES));
    let Some((&sel, body)) = data.split_first() else {
        return;
    };
    let (rel, orig) = FILES[(sel & 0x3f) as usize % 4];
    dir.with_file(rel, orig, body, |root| {
        let o = OpenOpts {
            megatron_hf: (sel & 0x80 != 0).then(|| "auto".to_string()),
            vocab_size: (sel & 0x40 != 0).then_some(50),
            ..Default::default()
        };
        if let Ok(ck) = Checkpoint::open_with(&root.join("megatron/legacy"), &o) {
            common::exercise(&ck);
        }
    });
});
