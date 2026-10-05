//! Raw bytes into the restricted pickle VM, with each allowlist, then render the result as JSON
//! (the walker `inspect` uses for non-tensor values).
#![no_main]
#[path = "common.rs"]
mod common;
use ckpt::pickle::{Allow, load};
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    let Some((&sel, body)) = data.split_first() else {
        return;
    };
    let allow = match sel % 3 {
        0 => Allow::Checkpoint,
        1 => Allow::Megatron,
        _ => Allow::DeepSpeed,
    };
    if let Ok(pk) = load(body, allow) {
        let _ = pk.to_json(&pk.root);
    }
});
