//! Hand-crafted hostile checkpoints. Every one must be refused with exit code 2 and a clear message,
//! without a panic, a hang or a large allocation. The files are built in temp dirs at test time.
//!
//! Inputs found by fuzzing (`fuzz/`) are replayed by `fuzz_regressions` at the end.

use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use std::time::{Duration, Instant};

fn ckpt(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_ckpt"))
        .args(args)
        .output()
        .expect("run ckpt")
}

/// `ckpt args` must fail cleanly: exit code 2, `needle` in the error, no panic, within 20 s.
fn refused(args: &[&str], needle: &str) {
    let t = Instant::now();
    let o = ckpt(args);
    let err = String::from_utf8_lossy(&o.stderr);
    assert_eq!(o.status.code(), Some(2), "ckpt {args:?}: {err}");
    assert!(
        err.contains(needle),
        "ckpt {args:?}: expected {needle:?} in:\n{err}"
    );
    assert!(
        !err.contains("internal error") && !err.contains("panicked"),
        "ckpt {args:?} panicked:\n{err}"
    );
    assert!(
        t.elapsed() < Duration::from_secs(20),
        "ckpt {args:?} too slow"
    );
}

fn s(p: &Path) -> &str {
    p.to_str().unwrap()
}

// ------------------------------------------------------------------ builders

/// A tiny protocol-2 pickle emitter.
struct P(Vec<u8>);
impl P {
    fn new() -> P {
        P(vec![0x80, 2])
    }
    fn op(&mut self, b: &[u8]) -> &mut Self {
        self.0.extend_from_slice(b);
        self
    }
    fn global(&mut self, m: &str, n: &str) -> &mut Self {
        self.op(format!("c{m}\n{n}\n").as_bytes())
    }
    fn int(&mut self, v: i64) -> &mut Self {
        self.op(format!("I{v}\n").as_bytes())
    }
    fn str(&mut self, v: &str) -> &mut Self {
        self.op(b"X")
            .op(&(v.len() as u32).to_le_bytes())
            .op(v.as_bytes())
    }
    /// `module.name()` with an empty-args REDUCE, then a dict state that `fields` fills via BUILD
    fn obj(&mut self, m: &str, n: &str, fields: impl FnOnce(&mut P)) -> &mut Self {
        self.global(m, n).op(b")R}(");
        fields(self);
        self.op(b"ub")
    }
    fn size(&mut self, dims: &[i64]) -> &mut Self {
        self.global("torch", "Size").op(b"((");
        for &d in dims {
            self.int(d);
        }
        self.op(b"ttR")
    }
    fn stop(&mut self) -> Vec<u8> {
        self.op(b".");
        std::mem::take(&mut self.0)
    }
}

const DCP: &str = "torch.distributed.checkpoint.metadata";

/// A DCP `.metadata` with one f32 tensor `w` of `size`, one chunk `(offs, sizes)` stored in
/// `__0_0.distcp` at `(offset, length)`.
fn dcp_metadata(size: &[i64], offs: &[i64], sizes: &[i64], offset: i64, length: i64) -> Vec<u8> {
    let mut p = P::new();
    p.obj(DCP, "Metadata", |p| {
        p.str("state_dict_metadata").op(b"}(").str("w");
        p.obj(DCP, "TensorStorageMetadata", |p| {
            p.str("properties").obj(DCP, "TensorProperties", |p| {
                p.str("dtype").global("torch", "float32");
            });
            p.str("size").size(size);
            p.str("chunks").op(b"](");
            p.obj(DCP, "ChunkStorageMetadata", |p| {
                p.str("offsets").size(offs).str("sizes").size(sizes);
            });
            p.op(b"e");
        });
        p.op(b"u");
        p.str("storage_data").op(b"}(");
        p.obj(DCP, "MetadataIndex", |p| {
            p.str("fqn").str("w").str("offset").size(offs);
        });
        p.obj(
            "torch.distributed.checkpoint.filesystem",
            "_StorageInfo",
            |p| {
                p.str("relative_path").str("__0_0.distcp");
                p.str("offset").int(offset).str("length").int(length);
            },
        );
        p.op(b"u");
    });
    p.stop()
}

fn dcp_dir(meta: &[u8], data: &[u8]) -> tempfile::TempDir {
    let td = tempfile::tempdir().unwrap();
    std::fs::write(td.path().join(".metadata"), meta).unwrap();
    std::fs::write(td.path().join("__0_0.distcp"), data).unwrap();
    td
}

/// A stored (uncompressed) zip. `cd` lists the central-directory records as
/// (name, index of the local entry it points to, compression method).
fn zip_raw(locals: &[(&str, &[u8])], cd: &[(&str, usize, u16)]) -> Vec<u8> {
    let mut z = Vec::new();
    let mut offs = Vec::new();
    for (name, data) in locals {
        offs.push(z.len() as u32);
        let crc = crc32fast::hash(data);
        z.extend_from_slice(&0x0403_4b50u32.to_le_bytes());
        z.extend_from_slice(&[20, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
        z.extend_from_slice(&crc.to_le_bytes());
        z.extend_from_slice(&(data.len() as u32).to_le_bytes());
        z.extend_from_slice(&(data.len() as u32).to_le_bytes());
        z.extend_from_slice(&(name.len() as u16).to_le_bytes());
        z.extend_from_slice(&0u16.to_le_bytes());
        z.extend_from_slice(name.as_bytes());
        z.extend_from_slice(data);
    }
    let cd_start = z.len();
    for (name, li, method) in cd {
        let data = locals[*li].1;
        z.extend_from_slice(&0x0201_4b50u32.to_le_bytes());
        z.extend_from_slice(&[20, 0, 20, 0, 0, 0]);
        z.extend_from_slice(&method.to_le_bytes());
        z.extend_from_slice(&[0, 0, 0, 0]);
        z.extend_from_slice(&crc32fast::hash(data).to_le_bytes());
        z.extend_from_slice(&(data.len() as u32).to_le_bytes());
        z.extend_from_slice(&(data.len() as u32).to_le_bytes());
        z.extend_from_slice(&(name.len() as u16).to_le_bytes());
        z.extend_from_slice(&[0u8; 12]);
        z.extend_from_slice(&offs[*li].to_le_bytes());
        z.extend_from_slice(name.as_bytes());
    }
    let cd_len = z.len() - cd_start;
    z.extend_from_slice(&0x0605_4b50u32.to_le_bytes());
    z.extend_from_slice(&[0, 0, 0, 0]);
    z.extend_from_slice(&(cd.len() as u16).to_le_bytes());
    z.extend_from_slice(&(cd.len() as u16).to_le_bytes());
    z.extend_from_slice(&(cd_len as u32).to_le_bytes());
    z.extend_from_slice(&(cd_start as u32).to_le_bytes());
    z.extend_from_slice(&[0, 0]);
    z
}

fn zip(entries: &[(&str, &[u8])]) -> Vec<u8> {
    let cd: Vec<(&str, usize, u16)> = entries
        .iter()
        .enumerate()
        .map(|(i, (n, _))| (*n, i, 0))
        .collect();
    zip_raw(entries, &cd)
}

/// A torch.save archive whose data.pkl is `pkl` plus the given storages.
fn torch_save(pkl: &[u8], storages: &[(&str, &[u8])]) -> Vec<u8> {
    let mut e: Vec<(String, &[u8])> = vec![("archive/data.pkl".into(), pkl)];
    for (k, d) in storages {
        e.push((format!("archive/data/{k}"), d));
    }
    let refs: Vec<(&str, &[u8])> = e.iter().map(|(n, d)| (n.as_str(), *d)).collect();
    zip(&refs)
}

/// data.pkl of `{"w": _rebuild_tensor_v2(storage(FloatStorage, "0", cpu, numel), 0, sizes, strides)}`
fn tensor_pkl(numel: i64, sizes: &[i64], strides: &[i64]) -> Vec<u8> {
    tensor_rec(numel, sizes, strides, true)
}

/// The same tensor record, as a dict entry or as the pickle root (a DCP chunk record).
fn tensor_rec(numel: i64, sizes: &[i64], strides: &[i64], in_dict: bool) -> Vec<u8> {
    let mut p = P::new();
    if in_dict {
        p.op(b"}(").str("w");
    }
    p.global("torch._utils", "_rebuild_tensor_v2").op(b"((");
    p.str("storage")
        .global("torch", "FloatStorage")
        .str("0")
        .str("cpu")
        .int(numel)
        .op(b"tQ");
    p.int(0).op(b"(");
    for &d in sizes {
        p.int(d);
    }
    p.op(b"t(");
    for &d in strides {
        p.int(d);
    }
    p.op(b"t").op(b"\x89").op(b"}").op(b"tR"); // requires_grad=False, backward_hooks={}
    if in_dict {
        p.op(b"u");
    }
    p.stop()
}

fn safetensors(header: &str, data_len: usize) -> Vec<u8> {
    let mut v = (header.len() as u64).to_le_bytes().to_vec();
    v.extend_from_slice(header.as_bytes());
    v.resize(v.len() + data_len, 0);
    v
}

fn write(dir: &Path, name: &str, data: &[u8]) -> PathBuf {
    let p = dir.join(name);
    std::fs::write(&p, data).unwrap();
    p
}

// ------------------------------------------------------------------ the builders themselves work

#[test]
fn builders_make_valid_files() {
    let td = tempfile::tempdir().unwrap();
    // 2x3 f32 tensor
    let pt = torch_save(&tensor_pkl(6, &[2, 3], &[3, 1]), &[("0", &[0u8; 24])]);
    let p = write(td.path(), "ok.pt", &pt);
    let o = ckpt(&["inspect", "--json", s(&p)]);
    assert!(o.status.success(), "{}", String::from_utf8_lossy(&o.stderr));
    assert!(String::from_utf8_lossy(&o.stdout).contains("\"shape\": ["));
    // DCP: one 4x4 f32 chunk, its torch_save record is not needed for inspect
    let d = dcp_dir(&dcp_metadata(&[4, 4], &[0, 0], &[4, 4], 0, 64), &[0u8; 64]);
    let o = ckpt(&["inspect", "--json", s(d.path())]);
    assert!(o.status.success(), "{}", String::from_utf8_lossy(&o.stderr));
    let st = safetensors(
        r#"{"a":{"dtype":"F32","shape":[2],"data_offsets":[0,8]}}"#,
        8,
    );
    let p = write(td.path(), "ok.safetensors", &st);
    assert!(ckpt(&["inspect", s(&p)]).status.success());
}

// ------------------------------------------------------------------ safetensors

#[test]
fn safetensors_hostile_headers() {
    let td = tempfile::tempdir().unwrap();
    let cases: &[(&str, usize, &str)] = &[
        // shape product overflows u64
        (
            r#"{"a":{"dtype":"F32","shape":[4294967296,4294967296,16],"data_offsets":[0,0]}}"#,
            0,
            "absurdly large",
        ),
        // huge but representable shape, a few bytes of data
        (
            r#"{"a":{"dtype":"F32","shape":[1000000,1000000],"data_offsets":[0,8]}}"#,
            8,
            "does not match shape",
        ),
        // zero dim hiding an overflowing stride
        (
            r#"{"a":{"dtype":"F32","shape":[0,4294967296,4294967296],"data_offsets":[0,0]}}"#,
            0,
            "absurdly large",
        ),
        // negative and overflowing offsets
        (
            r#"{"a":{"dtype":"F32","shape":[1],"data_offsets":[-4,0]}}"#,
            8,
            "out of bounds",
        ),
        (
            r#"{"a":{"dtype":"F32","shape":[1],"data_offsets":[18446744073709551612,18446744073709551616]}}"#,
            8,
            "out of bounds",
        ),
        (
            r#"{"a":{"dtype":"F32","shape":[1],"data_offsets":[4,12]}}"#,
            8,
            "out of bounds",
        ),
        // overlapping tensors
        (
            r#"{"a":{"dtype":"F32","shape":[2],"data_offsets":[0,8]},"b":{"dtype":"F32","shape":[1],"data_offsets":[4,8]}}"#,
            8,
            "overlap",
        ),
        // too many dims
        (
            r#"{"a":{"dtype":"U8","shape":[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],"data_offsets":[0,1]}}"#,
            1,
            "rank",
        ),
    ];
    for (i, (h, n, needle)) in cases.iter().enumerate() {
        let p = write(td.path(), &format!("h{i}.safetensors"), &safetensors(h, *n));
        refused(&["inspect", s(&p)], needle);
        let out = td.path().join(format!("o{i}.safetensors"));
        refused(&["convert", s(&p), "-o", s(&out)], needle);
    }
    // header length larger than the file / absurd
    let mut v = u64::MAX.to_le_bytes().to_vec();
    v.extend_from_slice(b"{}");
    let p = write(td.path(), "len.safetensors", &v);
    refused(&["inspect", s(&p)], "invalid safetensors header length");
}

// ------------------------------------------------------------------ DCP

#[test]
fn dcp_hostile_metadata() {
    // absurd declared shape
    let d = dcp_dir(
        &dcp_metadata(&[1 << 31, 1 << 31], &[0, 0], &[1 << 31, 1 << 31], 0, 64),
        &[0u8; 64],
    );
    refused(&["inspect", s(d.path())], "absurdly large");
    // plausible shape, 4 GB declared, 64 bytes on disk: refused before any allocation
    let d = dcp_dir(
        &dcp_metadata(&[1 << 20, 1 << 10], &[0, 0], &[1 << 20, 1 << 10], 0, 64),
        &[0u8; 64],
    );
    refused(
        &["inspect", s(d.path())],
        "files holding it have only 64 bytes",
    );
    let out = d.path().join("o.safetensors");
    refused(&["convert", s(d.path()), "-o", s(&out)], "only 64 bytes");
    // negative storage offset / length, overflowing offset + length, range past the file
    for (off, len, needle) in [
        (-8, 64, "invalid storage range"),
        (0, -1, "invalid storage range"),
        (i64::MAX - 4, 64, "invalid storage range"),
        (32, 64, "outside __0_0.distcp"),
    ] {
        let d = dcp_dir(
            &dcp_metadata(&[4, 4], &[0, 0], &[4, 4], off, len),
            &[0u8; 64],
        );
        refused(&["inspect", s(d.path())], needle);
    }
    // chunk offsets that overflow / lie outside the tensor, negative sizes
    let d = dcp_dir(
        &dcp_metadata(&[4, 4], &[i64::MAX, 0], &[4, 4], 0, 64),
        &[0u8; 64],
    );
    refused(&["inspect", s(d.path())], "out of bounds");
    let d = dcp_dir(&dcp_metadata(&[4, 4], &[0, 0], &[-4, 4], 0, 64), &[0u8; 64]);
    refused(&["inspect", s(d.path())], "negative size");
    // path traversal in relative_path
    let meta = dcp_metadata(&[4, 4], &[0, 0], &[4, 4], 0, 64);
    let i = meta.windows(12).position(|w| w == b"__0_0.distcp").unwrap();
    let mut m2 = meta.clone();
    m2.splice(i..i + 12, b"../0_0.distcp".iter().copied());
    m2[i - 4] = 13; // the BINUNICODE length prefix
    let d = dcp_dir(&m2, &[0u8; 64]);
    refused(&["inspect", s(d.path())], "suspicious relative_path");
}

#[test]
fn dcp_chunk_record_lies_about_its_storage() {
    // the .metadata is sane (4x4 f32 = 64 bytes); the chunk's own torch.save record claims a
    // 2^40-element view of a 16-element storage
    let rec = torch_save(
        &tensor_rec(16, &[1 << 40], &[1], false),
        &[("0", &[0u8; 64])],
    );
    let len = rec.len() as i64;
    let d = dcp_dir(&dcp_metadata(&[4, 4], &[0, 0], &[4, 4], 0, len), &rec);
    let out = d.path().join("o.safetensors");
    refused(
        &["convert", s(d.path()), "-o", s(&out)],
        "exceeds its storage",
    );
}

// ------------------------------------------------------------------ pickles

#[test]
fn deeply_nested_pickle() {
    let td = tempfile::tempdir().unwrap();
    // 100k nested MARKs: over the MARK-stack limit
    let mut p = P::new();
    p.op(&vec![b'('; 100_000]);
    let pk = p.op(b"N").stop();
    let d = dcp_dir(&pk, &[]);
    refused(&["inspect", s(d.path())], "nested too deeply");
    // a list nested 200k deep (built iteratively in the arena): parses, but walking it is cut off
    let mut p = P::new();
    p.op(b"}(").str("w");
    for _ in 0..200_000 {
        p.op(b"]");
    }
    for _ in 0..199_999 {
        p.op(b"a");
    }
    let pk = p.op(b"u").stop();
    let f = write(td.path(), "deep.pt", &torch_save(&pk, &[]));
    refused(&["inspect", s(&f)], "nested too deeply");
}

#[test]
fn memo_reference_abuse() {
    let td = tempfile::tempdir().unwrap();
    // a 100k-item list in memo slot 0, then `set(memo[0])` 1000 times: 10^8 copied values
    // (several GB) without the copy budget
    let mut p = P::new();
    p.op(b"]q\x00(");
    for _ in 0..100_000 {
        p.op(b"K\x01");
    }
    p.op(b"e0");
    for _ in 0..1000 {
        p.global("builtins", "set").op(b"h\x00\x85R0");
    }
    let pk = p.op(b"N").stop();
    let f = write(td.path(), "memo.pt", &torch_save(&pk, &[]));
    refused(&["inspect", s(&f)], "memo-reference abuse");
    // a DAG of shared references: L_i = [L_{i-1}, L_{i-1}], 60 levels = 2^60 paths
    let mut p = P::new();
    p.op(b"}(").str("w").op(b"]q\x000");
    for i in 1..=60u8 {
        p.op(&[b'(', b'h', i - 1, b'h', i - 1, b'l', b'q', i, b'0']);
    }
    let pk = p.op(b"h\x3cu").stop();
    let f = write(td.path(), "dag.pt", &torch_save(&pk, &[]));
    refused(&["inspect", s(&f)], "shared-reference bomb");
    // the same lists also used as dict keys: rendering a key must not create huge tensor names
    let mut p = P::new();
    p.op(b"}(").str("w").op(b"]q\x00");
    for i in 1..=60u8 {
        p.op(&[b'(', b'h', i - 1, b'h', i - 1, b'l', b'q', i]);
    }
    let pk = p.op(b"u").stop();
    let f = write(td.path(), "dagkeys.pt", &torch_save(&pk, &[]));
    refused(&["inspect", s(&f)], "shared-reference bomb");
    // a 20-level DAG over one tensor record: 2^20 tensor entries from a few hundred bytes
    let mut p = P::new();
    let t = tensor_rec(1, &[1], &[1], false);
    p.op(b"}(").str("w").op(&t[2..t.len() - 1]).op(b"q\x000");
    for i in 1..=20u8 {
        p.op(&[b'(', b'h', i - 1, b'h', i - 1, b'l', b'q', i, b'0']);
    }
    let pk = p.op(b"h\x14u").stop();
    let f = write(td.path(), "dagt.pt", &torch_save(&pk, &[("0", &[0u8; 4])]));
    refused(&["inspect", s(&f)], "shared-reference bomb");
}

#[test]
fn torch_save_hostile_tensor_records() {
    let td = tempfile::tempdir().unwrap();
    for (i, (numel, sizes, strides, needle)) in [
        (16, vec![1i64 << 40], vec![1i64], "exceeds its storage"),
        (16, vec![4, -4], vec![4, 1], "negative size"),
        (16, vec![1 << 62, 1 << 62], vec![1, 1], "absurdly large"),
        (16, vec![3, 3], vec![1 << 62, 1 << 62], "overflow"),
        (1 << 40, vec![4], vec![1], "truncated"),
        (-1, vec![4], vec![1], "bad storage numel"),
    ]
    .into_iter()
    .enumerate()
    {
        let pt = torch_save(&tensor_pkl(numel, &sizes, &strides), &[("0", &[0u8; 64])]);
        let f = write(td.path(), &format!("t{i}.pt"), &pt);
        refused(&["inspect", s(&f)], needle);
    }
}

// ------------------------------------------------------------------ zip

#[test]
fn zip_hostile_archives() {
    let td = tempfile::tempdir().unwrap();
    let pkl = tensor_pkl(16, &[4, 4], &[4, 1]);
    let st = [0u8; 64];
    // two central-directory records pointing at the same local entry
    let z = zip_raw(
        &[("archive/data.pkl", &pkl), ("archive/data/0", &st)],
        &[
            ("archive/data.pkl", 0, 0),
            ("archive/data/0", 1, 0),
            ("archive/data/1", 1, 0),
        ],
    );
    let f = write(td.path(), "overlap.pt", &z);
    refused(&["inspect", s(&f)], "overlap");
    // duplicate names
    let z = zip_raw(
        &[
            ("archive/data.pkl", &pkl),
            ("archive/data/0", &st),
            ("x", b""),
        ],
        &[
            ("archive/data.pkl", 0, 0),
            ("archive/data/0", 1, 0),
            ("archive/data/0", 2, 0),
        ],
    );
    let f = write(td.path(), "dup.pt", &z);
    refused(&["inspect", s(&f)], "duplicate entry name");
    // compressed (deflate) entry: refused, nothing is ever inflated
    let z = zip_raw(
        &[("archive/data.pkl", &pkl), ("archive/data/0", &st)],
        &[("archive/data.pkl", 0, 0), ("archive/data/0", 1, 8)],
    );
    let f = write(td.path(), "deflate.pt", &z);
    refused(&["inspect", s(&f)], "compressed");
    // zip64 end-of-central-directory locator pointing at offset 2^64-1
    let mut z = b"PK\x03\x04".to_vec();
    z.extend_from_slice(&[0u8; 40]);
    z.extend_from_slice(&0x0706_4b50u32.to_le_bytes());
    z.extend_from_slice(&0u32.to_le_bytes());
    z.extend_from_slice(&u64::MAX.to_le_bytes());
    z.extend_from_slice(&1u32.to_le_bytes());
    z.extend_from_slice(&0x0605_4b50u32.to_le_bytes());
    z.extend_from_slice(&[0, 0, 0, 0, 0xff, 0xff, 0xff, 0xff]);
    z.extend_from_slice(&[0xff; 8]);
    z.extend_from_slice(&[0, 0]);
    let f = write(td.path(), "zip64.pt", &z);
    refused(&["inspect", s(&f)], "zip: truncated");
    // central directory offset past the end of the file
    let mut z = zip(&[("archive/data.pkl", &pkl)]);
    let n = z.len();
    z[n - 6..n - 2].copy_from_slice(&0xfff0_0000u32.to_le_bytes());
    let f = write(td.path(), "cdoff.pt", &z);
    refused(&["inspect", s(&f)], "zip: truncated");
}

// ------------------------------------------------------------------ HF index

#[test]
fn hf_index_hostile() {
    let td = tempfile::tempdir().unwrap();
    let st = safetensors(
        r#"{"a":{"dtype":"F32","shape":[2],"data_offsets":[0,8]}}"#,
        8,
    );
    write(td.path(), "model-1.safetensors", &st);
    for (i, (idx, needle)) in [
        (
            r#"{"weight_map":{"a":"../model-1.safetensors"}}"#,
            "suspicious shard file name",
        ),
        (
            r#"{"weight_map":{"a":"/etc/passwd"}}"#,
            "suspicious shard file name",
        ),
        (
            r#"{"weight_map":{"a":"C:model-1.safetensors"}}"#,
            "suspicious shard file name",
        ),
        (r#"{"weight_map":{"a":7}}"#, "bad weight_map value"),
        (r#"{"weight_map":[]}"#, "no weight_map"),
        (
            r#"{"weight_map":{"b":"model-1.safetensors"}}"#,
            "no shard contains it",
        ),
        (
            r#"{"weight_map":{"a":"model-1.safetensors""#,
            "parsing index json",
        ),
    ]
    .into_iter()
    .enumerate()
    {
        let d = td.path().join(format!("d{i}"));
        std::fs::create_dir(&d).unwrap();
        write(&d, "model-1.safetensors", &st);
        write(&d, "model.safetensors.index.json", idx.as_bytes());
        refused(&["inspect", s(&d)], needle);
    }
}

// ------------------------------------------------------------------ fuzzer findings

/// Every input in `tests/fuzz_regressions/<target>/` (crashes found by `cargo fuzz`, minimized) must
/// be refused cleanly by the CLI. The target name says how to place the input.
#[test]
fn fuzz_regressions() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fuzz_regressions");
    let Ok(targets) = std::fs::read_dir(&root) else {
        return;
    };
    let fix = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/tiny");
    let mut n = 0;
    for t in targets {
        let t = t.unwrap().path();
        let target = t.file_name().unwrap().to_str().unwrap().to_string();
        for f in std::fs::read_dir(&t).unwrap() {
            let f = f.unwrap().path();
            let data = std::fs::read(&f).unwrap();
            let td = tempfile::tempdir().unwrap();
            let path = place(&target, &data, &fix, td.path());
            let Some(path) = path else { continue };
            let o = ckpt(&["inspect", "--json", s(&path)]);
            let err = String::from_utf8_lossy(&o.stderr);
            assert!(
                matches!(o.status.code(), Some(0) | Some(2)),
                "{}: exit {:?}\n{err}",
                f.display(),
                o.status.code()
            );
            assert!(
                !err.contains("internal error") && !err.contains("panicked"),
                "{} panicked:\n{err}",
                f.display()
            );
            n += 1;
        }
    }
    eprintln!("replayed {n} fuzz regression inputs");
}

fn copy_dir(src: &Path, dst: &Path) {
    std::fs::create_dir_all(dst).unwrap();
    for e in std::fs::read_dir(src).unwrap() {
        let p = e.unwrap().path();
        let q = dst.join(p.file_name().unwrap());
        if p.is_dir() {
            copy_dir(&p, &q);
        } else {
            std::fs::copy(&p, &q).unwrap();
        }
    }
}

/// Lay out a fuzz input the way its target does; returns the path to inspect.
fn place(target: &str, data: &[u8], fix: &Path, td: &Path) -> Option<PathBuf> {
    let (sel, body) = data.split_first().map(|(a, b)| (*a, b)).unwrap_or((0, &[]));
    match target {
        "pickle" | "dcp_metadata" => {
            let body = if target == "pickle" { body } else { data };
            std::fs::write(td.join(".metadata"), body).unwrap();
            Some(td.to_path_buf())
        }
        "torch_save" => Some(write(td, "x.pt", body)),
        "safetensors" => Some(write(td, "x.safetensors", data)),
        "hf_index" => {
            copy_dir(&fix.join("hf_sharded"), td);
            write(td, "model.safetensors.index.json", data);
            Some(td.to_path_buf())
        }
        "dcp_dir" => {
            let dirs = [
                "dcp_fsdp",
                "dcp_fsdp_st",
                "dcp_2d",
                "megatron/torch_dist/iter_0000010",
            ];
            let i = (sel & 0x7f) as usize % 20;
            let d = dirs[i / 5];
            copy_dir(&fix.join(d), td);
            let file = if i.is_multiple_of(5) {
                ".metadata".to_string()
            } else {
                format!("__{}_0.distcp", i % 5 - 1)
            };
            write(td, &file, body);
            Some(td.to_path_buf())
        }
        "megatron" => {
            copy_dir(&fix.join("megatron/legacy"), td);
            let r = (sel & 0x3f) as usize % 4;
            let f = format!(
                "iter_0000010/mp_rank_0{}_00{}/model_optim_rng.pt",
                r / 2,
                r % 2
            );
            write(td, &f, body);
            Some(td.to_path_buf())
        }
        "deepspeed" => {
            let i = sel as usize % 10;
            let stage = if i < 4 { "zero2" } else { "zero3" };
            copy_dir(&fix.join("deepspeed").join(stage), td);
            let g = td.join("global_step1");
            let name = match i {
                0 => "mp_rank_00_model_states.pt".to_string(),
                1..=3 => format!("bf16_zero_pp_rank_{}_mp_rank_00_optim_states.pt", i - 1),
                4..=6 => format!("zero_pp_rank_{}_mp_rank_00_model_states.pt", i - 4),
                _ => format!("bf16_zero_pp_rank_{}_mp_rank_00_optim_states.pt", i - 7),
            };
            write(&g, &name, body);
            Some(td.to_path_buf())
        }
        _ => None,
    }
}
