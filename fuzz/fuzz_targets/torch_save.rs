//! Raw bytes as a `torch.save` zip archive: the zip reader, the single-tensor chunk decoder used by
//! DCP, and a whole `.pt` state dict opened, read and streamed tensor by tensor.
#![no_main]
#[path = "common.rs"]
mod common;
use ckpt::ckpt::Checkpoint;
use ckpt::pickle::Allow;
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    let Some((&sel, body)) = data.split_first() else {
        return;
    };
    if body.is_empty() {
        return;
    }
    if let Ok(es) = ckpt::zipread::entries(body) {
        for e in &es {
            let _ = e.verify();
        }
    }
    if let Ok(v) = ckpt::dcp::read_torch_save_tensor(body) {
        // views may overlap their storage (stride 0, as_strided), so cap the test allocation
        let n = ckpt::dtype::numel(&v.sizes) as usize * v.dtype.size();
        if n <= 1 << 24 {
            let mut buf = vec![0u8; n];
            let z = vec![0; v.sizes.len()];
            let sizes = v.sizes.clone();
            ckpt::ckpt::copy_chunk(&mut buf, &sizes, &z, &v).expect("copy of a validated view");
        }
    }
    let allow = match sel % 3 {
        0 => Allow::Checkpoint,
        1 => Allow::Megatron,
        _ => Allow::DeepSpeed,
    };
    if let Ok(ck) = Checkpoint::open_torch_save_map(common::mapped(body, "fuzz.pt"), allow) {
        common::exercise(&ck);
    }
});
