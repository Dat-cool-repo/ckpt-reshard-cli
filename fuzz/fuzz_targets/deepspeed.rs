//! A DeepSpeed ZeRO-2 or ZeRO-3 checkpoint (3 ranks) with one `*_model_states.pt` /
//! `*_optim_states.pt` file replaced by the fuzz input: the DeepSpeed allowlist and the fp32
//! reconstruction.
#![no_main]
#[path = "common.rs"]
mod common;
use ckpt::ckpt::Checkpoint;
use common::{FixtureDir, fixture};
use libfuzzer_sys::fuzz_target;
use std::sync::OnceLock;

const FILES: &[(&str, &[u8])] = &[
    fixture!("deepspeed/zero2/global_step1/mp_rank_00_model_states.pt"),
    fixture!("deepspeed/zero2/global_step1/bf16_zero_pp_rank_0_mp_rank_00_optim_states.pt"),
    fixture!("deepspeed/zero2/global_step1/bf16_zero_pp_rank_1_mp_rank_00_optim_states.pt"),
    fixture!("deepspeed/zero2/global_step1/bf16_zero_pp_rank_2_mp_rank_00_optim_states.pt"),
    fixture!("deepspeed/zero3/global_step1/zero_pp_rank_0_mp_rank_00_model_states.pt"),
    fixture!("deepspeed/zero3/global_step1/zero_pp_rank_1_mp_rank_00_model_states.pt"),
    fixture!("deepspeed/zero3/global_step1/zero_pp_rank_2_mp_rank_00_model_states.pt"),
    fixture!("deepspeed/zero3/global_step1/bf16_zero_pp_rank_0_mp_rank_00_optim_states.pt"),
    fixture!("deepspeed/zero3/global_step1/bf16_zero_pp_rank_1_mp_rank_00_optim_states.pt"),
    fixture!("deepspeed/zero3/global_step1/bf16_zero_pp_rank_2_mp_rank_00_optim_states.pt"),
    fixture!("deepspeed/zero2/latest"),
    fixture!("deepspeed/zero3/latest"),
];

fuzz_target!(|data: &[u8]| {
    static DIR: OnceLock<FixtureDir> = OnceLock::new();
    let dir = DIR.get_or_init(|| FixtureDir::new("deepspeed", FILES));
    let Some((&sel, body)) = data.split_first() else {
        return;
    };
    let i = sel as usize % 10;
    let (rel, orig) = FILES[i];
    let stage = if i < 4 { "zero2" } else { "zero3" };
    dir.with_file(rel, orig, body, |root| {
        if let Ok(ck) = Checkpoint::open(&root.join("deepspeed").join(stage)) {
            common::exercise(&ck);
        }
    });
});
