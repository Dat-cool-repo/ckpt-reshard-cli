//! Raw bytes as a DCP `.metadata` file.
#![no_main]
#[path = "common.rs"]
mod common;
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    let _ = ckpt::dcp::parse_metadata(data);
});
