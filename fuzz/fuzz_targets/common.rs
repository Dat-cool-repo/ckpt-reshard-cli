//! Shared helpers for the fuzz targets (included with `#[path = "common.rs"] mod common;`).
#![allow(dead_code)]

use ckpt::ckpt::{Checkpoint, MappedFile};
use std::path::{Path, PathBuf};
use std::sync::Once;

#[allow(unused_macros)]
macro_rules! fixture {
    ($p:literal) => {
        (
            $p,
            include_bytes!(concat!(env!("CARGO_MANIFEST_DIR"), "/../tests/fixtures/tiny/", $p))
                as &[u8],
        )
    };
}
#[allow(unused_imports)]
pub(crate) use fixture;

/// Abort (a crash libFuzzer records with its input) on any single allocation over 256 MiB. The
/// fuzz inputs and fixtures are at most a few hundred KiB, so a bigger allocation means a declared
/// size was trusted. This works with or without a sanitizer (`-malloc_limit_mb` needs one).
struct LimitAlloc;
const ALLOC_LIMIT: usize = 256 << 20;
unsafe impl std::alloc::GlobalAlloc for LimitAlloc {
    unsafe fn alloc(&self, l: std::alloc::Layout) -> *mut u8 {
        check(l.size());
        unsafe { std::alloc::System.alloc(l) }
    }
    unsafe fn alloc_zeroed(&self, l: std::alloc::Layout) -> *mut u8 {
        check(l.size());
        unsafe { std::alloc::System.alloc_zeroed(l) }
    }
    unsafe fn dealloc(&self, p: *mut u8, l: std::alloc::Layout) {
        unsafe { std::alloc::System.dealloc(p, l) }
    }
    unsafe fn realloc(&self, p: *mut u8, l: std::alloc::Layout, n: usize) -> *mut u8 {
        check(n);
        unsafe { std::alloc::System.realloc(p, l, n) }
    }
}
fn check(n: usize) {
    if n > ALLOC_LIMIT {
        // no formatting machinery here: it could allocate
        let _ = std::io::Write::write_all(
            &mut std::io::stderr(),
            b"\n==ckpt-fuzz== allocation over 256 MiB requested\n",
        );
        // set CKPT_FUZZ_BT=1 when replaying a crash to see where it came from
        if std::env::var_os("CKPT_FUZZ_BT").is_some() {
            eprintln!("{n} bytes\n{}", std::backtrace::Backtrace::force_capture());
        }
        std::process::abort();
    }
}
#[global_allocator]
static GLOBAL: LimitAlloc = LimitAlloc;

/// One worker thread, so a fuzz process does not spawn a thread per core.
pub fn init() {
    static ONCE: Once = Once::new();
    ONCE.call_once(|| {
        let _ = rayon::ThreadPoolBuilder::new().num_threads(1).build_global();
    });
}

/// `bytes` as a read-only anonymous mapping (what the readers get from a real file).
/// `bytes` must not be empty (a zero-length anonymous mapping is not portable).
pub fn mapped(bytes: &[u8], name: &str) -> MappedFile {
    assert!(!bytes.is_empty());
    let mut m = memmap2::MmapMut::map_anon(bytes.len()).expect("map_anon");
    m.copy_from_slice(bytes);
    MappedFile {
        path: PathBuf::from(name),
        map: m.make_read_only().unwrap(),
    }
}

/// Everything a user can make `ckpt` do with an opened checkpoint, minus writing files: inspect
/// (with chunk grids and decoded non-tensor items), CRC verification, reading and streaming every
/// tensor, a diff against itself and casting every tensor to bf16.
pub fn exercise(ck: &Checkpoint) {
    init();
    // `Checkpoint::open*` validates; the map-based openers used by some targets do not
    if ck.validate().is_err() {
        return;
    }
    let o = ckpt::inspect::InspectOpts {
        filter: vec![],
        tensors: true,
        chunks: true,
    };
    let _ = ckpt::inspect::to_json(ck, &o);
    let _ = ck.verify_all();
    let _ = ck.shard_summary();
    for t in &ck.tensors {
        if let Ok(b) = ck.read(t) {
            assert_eq!(b.len() as u64, t.nbytes(), "read() size of {}", t.name);
            if ckpt::cast::is_float(t.dtype) {
                let _ = ckpt::cast::convert(t.dtype, ckpt::dtype::DType::BF16, &b);
            }
        }
        let mut n = 0u64;
        if ck
            .stream(t, &mut |b| {
                n += b.len() as u64;
                Ok(())
            })
            .is_ok()
        {
            assert_eq!(n, t.nbytes(), "stream() size of {}", t.name);
        }
        if !t.shape.is_empty() && t.shape[0] > 1 {
            let mut start = vec![0; t.shape.len()];
            start[0] = 1;
            let mut size = t.shape.clone();
            size[0] -= 1;
            let _ = ck.read_box(t, &start, &size);
        }
    }
    let _ = ckpt::diff::diff(ck, ck, &ckpt::diff::DiffOpts::default());
}

/// A private copy of a fixture directory in the temp dir. `with_file` swaps one file for the fuzz
/// input, runs the closure, then restores the original, so every run starts from the fixture.
pub struct FixtureDir {
    pub root: PathBuf,
}

impl FixtureDir {
    pub fn new(tag: &str, files: &[(&str, &[u8])]) -> FixtureDir {
        let root = std::env::temp_dir().join(format!("ckpt-fuzz-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        for (rel, data) in files {
            let p = root.join(rel);
            std::fs::create_dir_all(p.parent().unwrap()).unwrap();
            std::fs::write(&p, data).unwrap();
        }
        FixtureDir { root }
    }

    pub fn with_file<R>(&self, rel: &str, original: &[u8], data: &[u8], f: impl FnOnce(&Path) -> R) -> R {
        let p = self.root.join(rel);
        std::fs::write(&p, data).unwrap();
        let r = f(&self.root);
        std::fs::write(&p, original).unwrap();
        r
    }
}
