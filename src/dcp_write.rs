//! Write a PyTorch Distributed Checkpoint (FileSystemWriter layout, torch_save chunks) without torch.
//!
//! Every tensor with dim0 >= ranks is split into `ranks` row slabs (torch.chunk semantics, like
//! FSDP); slab r goes to `__r_0.distcp`. Smaller tensors and scalars go to rank 0. DCP reshards on
//! load, so the result loads into any world size (and via `dcp_to_torch_save`).
//! The pickles are emitted opcode by opcode (protocol 2), mirroring what torch itself writes.

use crate::ckpt::Checkpoint;
use crate::dtype::{DType, numel};
use crate::writer::Sel;
use anyhow::{Context, Result, bail};
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

// ------------------------------------------------------------------ pickle emitter

struct P(Vec<u8>);

impl P {
    fn new() -> P {
        P(vec![0x80, 2])
    }
    fn global(&mut self, m: &str, n: &str) {
        self.0.push(b'c');
        self.0.extend_from_slice(m.as_bytes());
        self.0.push(b'\n');
        self.0.extend_from_slice(n.as_bytes());
        self.0.push(b'\n');
    }
    fn s(&mut self, s: &str) {
        self.0.push(b'X');
        self.0.extend_from_slice(&(s.len() as u32).to_le_bytes());
        self.0.extend_from_slice(s.as_bytes());
    }
    fn int(&mut self, i: i64) {
        if (0..256).contains(&i) {
            self.0.extend_from_slice(&[b'K', i as u8]);
        } else if (0..65536).contains(&i) {
            self.0.push(b'M');
            self.0.extend_from_slice(&(i as u16).to_le_bytes());
        } else if i32::try_from(i).is_ok() {
            self.0.push(b'J');
            self.0.extend_from_slice(&(i as i32).to_le_bytes());
        } else {
            self.0.extend_from_slice(&[0x8a, 8]);
            self.0.extend_from_slice(&i.to_le_bytes());
        }
    }
    fn op(&mut self, o: u8) {
        self.0.push(o);
    }
    fn boolean(&mut self, b: bool) {
        self.0.push(if b { 0x88 } else { 0x89 });
    }
    fn int_tuple(&mut self, v: &[u64]) {
        self.op(b'(');
        for &x in v {
            self.int(x as i64);
        }
        self.op(b't');
    }
    fn size(&mut self, v: &[u64]) {
        self.global("torch", "Size");
        self.int_tuple(v);
        self.op(0x85); // TUPLE1
        self.op(b'R');
    }
    /// `cls()` via NEWOBJ, leaving the instance on the stack; caller emits state + BUILD.
    fn new_obj(&mut self, m: &str, n: &str) {
        self.global(m, n);
        self.op(b')');
        self.op(0x81);
    }
}

const MD: &str = "torch.distributed.checkpoint.metadata";

struct ChunkRec {
    fqn: String,
    offsets: Vec<u64>,
    sizes: Vec<u64>,
    file: String,
    offset: u64,
    length: u64,
}

struct TensorRec {
    fqn: String,
    dtype: DType,
    shape: Vec<u64>,
    chunks: Vec<usize>, // indexes into chunk records
}

fn metadata_pickle(tensors: &[TensorRec], chunks: &[ChunkRec]) -> Vec<u8> {
    let mut p = P::new();
    p.new_obj(MD, "Metadata");
    p.op(b'}');
    p.op(b'(');
    p.s("state_dict_metadata");
    p.op(b'}');
    p.op(b'(');
    for t in tensors {
        p.s(&t.fqn);
        p.new_obj(MD, "TensorStorageMetadata");
        p.op(b'}');
        p.op(b'(');
        p.s("properties");
        p.new_obj(MD, "TensorProperties");
        p.op(b'(');
        p.global("torch", t.dtype.torch_name());
        p.global("torch.serialization", "_get_layout");
        p.s("torch.strided");
        p.op(0x85);
        p.op(b'R');
        p.boolean(false);
        p.global(MD, "_MEM_FORMAT_ENCODING");
        p.int(0);
        p.op(0x85);
        p.op(b'R');
        p.boolean(false);
        p.op(b't');
        p.op(b'b');
        p.s("size");
        p.size(&t.shape);
        p.s("chunks");
        p.op(b']');
        p.op(b'(');
        for &ci in &t.chunks {
            let c = &chunks[ci];
            p.new_obj(MD, "ChunkStorageMetadata");
            p.op(b'}');
            p.op(b'(');
            p.s("offsets");
            p.size(&c.offsets);
            p.s("sizes");
            p.size(&c.sizes);
            p.op(b'u');
            p.op(b'b');
        }
        p.op(b'e');
        p.op(b'u');
        p.op(b'b');
    }
    p.op(b'u');
    p.s("planner_data");
    p.op(b'}');
    p.op(b'(');
    for t in tensors {
        p.s(&t.fqn);
        p.s(&t.fqn);
        p.op(0x85);
    }
    p.op(b'u');
    p.s("storage_data");
    p.op(b'}');
    p.op(b'(');
    for (i, c) in chunks.iter().enumerate() {
        let _ = i;
        p.new_obj(MD, "MetadataIndex");
        p.op(b'}');
        p.op(b'(');
        p.s("fqn");
        p.s(&c.fqn);
        p.s("offset");
        p.size(&c.offsets);
        p.s("index");
        p.int(0);
        p.op(b'u');
        p.op(b'b');
        p.new_obj("torch.distributed.checkpoint.filesystem", "_StorageInfo");
        p.op(b'}');
        p.op(b'(');
        p.s("relative_path");
        p.s(&c.file);
        p.s("offset");
        p.int(c.offset as i64);
        p.s("length");
        p.int(c.length as i64);
        p.op(b'u');
        p.op(b'b');
    }
    p.op(b'u');
    p.s("storage_meta");
    p.op(b'N');
    p.s("version");
    p.s("1.0.0");
    p.op(b'u');
    p.op(b'b');
    p.op(b'.');
    p.0
}

fn tensor_pickle(dtype: DType, sizes: &[u64]) -> Vec<u8> {
    let mut p = P::new();
    p.global("torch._utils", "_rebuild_tensor_v2");
    p.op(b'(');
    p.op(b'(');
    p.s("storage");
    p.global("torch", dtype.storage_name());
    p.s("0");
    p.s("cpu");
    p.int(numel(sizes) as i64);
    p.op(b't');
    p.op(b'Q');
    p.int(0);
    p.int_tuple(sizes);
    let mut strides = vec![1u64; sizes.len()];
    for d in (0..sizes.len().saturating_sub(1)).rev() {
        strides[d] = strides[d + 1] * sizes[d + 1];
    }
    p.int_tuple(&strides);
    p.boolean(false);
    p.global("collections", "OrderedDict");
    p.op(b')');
    p.op(b'R');
    p.op(b't');
    p.op(b'R');
    p.op(b'.');
    p.0
}

// ------------------------------------------------------------------ zip writer (stored, 64-byte aligned)

struct ZipOut<'a, W: Write> {
    w: &'a mut W,
    pos: u64,
    start: u64,
    central: Vec<u8>,
    count: u64,
    any_zip64: bool,
    /// write zip64 records even when nothing overflows
    force: bool,
}

const U32_MAX: u64 = 0xffff_ffff;

/// `CKPT_FORCE_ZIP64=1` writes zip64 records even for small entries (testing aid: lets the
/// zip64 code path be checked against torch/python without multi-GiB files).
fn force_zip64() -> bool {
    static F: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *F.get_or_init(|| std::env::var_os("CKPT_FORCE_ZIP64").is_some_and(|v| v != "0"))
}

impl<'a, W: Write> ZipOut<'a, W> {
    fn new(w: &'a mut W, pos: u64) -> Self {
        ZipOut {
            w,
            pos,
            start: pos,
            central: Vec::new(),
            count: 0,
            any_zip64: false,
            force: force_zip64(),
        }
    }
    /// Add a stored entry. Sizes or offsets >= 4 GiB switch to ZIP64 records (PKWARE APPNOTE 4.5.3):
    /// the local header carries both 64-bit sizes, the central record the overflowing fields.
    fn entry(&mut self, name: &str, data: &[u8]) -> Result<()> {
        let crc = crc32fast::hash(data);
        let rel = self.pos - self.start;
        let len = data.len() as u64;
        let big_size = self.force || len >= U32_MAX;
        let big_off = self.force || rel >= U32_MAX;
        let z64 = big_size || big_off;
        self.any_zip64 |= z64;
        let ver: u16 = if z64 { 45 } else { 20 };
        let local_z64_len = if big_size { 20 } else { 0 };
        let base = 30 + name.len() as u64 + local_z64_len + 4;
        let pad = (64 - (rel + base) % 64) % 64;
        let mut h = Vec::with_capacity(96 + name.len() + pad as usize);
        h.extend_from_slice(&0x0403_4b50u32.to_le_bytes());
        h.extend_from_slice(&ver.to_le_bytes()); // version needed
        h.extend_from_slice(&0u16.to_le_bytes()); // flags
        h.extend_from_slice(&0u16.to_le_bytes()); // stored
        h.extend_from_slice(&0u32.to_le_bytes()); // time+date
        h.extend_from_slice(&crc.to_le_bytes());
        let l32 = if big_size { U32_MAX as u32 } else { len as u32 };
        h.extend_from_slice(&l32.to_le_bytes()); // compressed
        h.extend_from_slice(&l32.to_le_bytes()); // uncompressed
        h.extend_from_slice(&(name.len() as u16).to_le_bytes());
        h.extend_from_slice(&((local_z64_len + 4 + pad) as u16).to_le_bytes());
        h.extend_from_slice(name.as_bytes());
        if big_size {
            h.extend_from_slice(&0x0001u16.to_le_bytes());
            h.extend_from_slice(&16u16.to_le_bytes());
            h.extend_from_slice(&len.to_le_bytes()); // uncompressed
            h.extend_from_slice(&len.to_le_bytes()); // compressed
        }
        h.extend_from_slice(b"FB");
        h.extend_from_slice(&(pad as u16).to_le_bytes());
        h.extend(std::iter::repeat_n(b'Z', pad as usize));
        self.w.write_all(&h)?;
        self.w.write_all(data)?;
        // central directory record; the zip64 extra holds only the fields that are 0xffffffff,
        // in the order uncompressed, compressed, local header offset
        let mut x = Vec::new();
        if big_size {
            x.extend_from_slice(&len.to_le_bytes());
            x.extend_from_slice(&len.to_le_bytes());
        }
        if big_off {
            x.extend_from_slice(&rel.to_le_bytes());
        }
        let c = &mut self.central;
        c.extend_from_slice(&0x0201_4b50u32.to_le_bytes());
        c.extend_from_slice(&ver.to_le_bytes()); // made by
        c.extend_from_slice(&ver.to_le_bytes()); // needed
        c.extend_from_slice(&0u16.to_le_bytes());
        c.extend_from_slice(&0u16.to_le_bytes());
        c.extend_from_slice(&0u32.to_le_bytes());
        c.extend_from_slice(&crc.to_le_bytes());
        c.extend_from_slice(&l32.to_le_bytes());
        c.extend_from_slice(&l32.to_le_bytes());
        c.extend_from_slice(&(name.len() as u16).to_le_bytes());
        let xl = if x.is_empty() { 0 } else { 4 + x.len() };
        c.extend_from_slice(&(xl as u16).to_le_bytes()); // extra
        c.extend_from_slice(&0u16.to_le_bytes()); // comment
        c.extend_from_slice(&0u16.to_le_bytes()); // disk
        c.extend_from_slice(&0u16.to_le_bytes()); // int attr
        c.extend_from_slice(&0u32.to_le_bytes()); // ext attr
        let o32 = if big_off { U32_MAX as u32 } else { rel as u32 };
        c.extend_from_slice(&o32.to_le_bytes());
        c.extend_from_slice(name.as_bytes());
        if !x.is_empty() {
            c.extend_from_slice(&0x0001u16.to_le_bytes());
            c.extend_from_slice(&(x.len() as u16).to_le_bytes());
            c.extend_from_slice(&x);
        }
        self.pos += h.len() as u64 + len;
        self.count += 1;
        Ok(())
    }
    /// Finish the archive; returns the new absolute stream position.
    fn finish(self) -> Result<u64> {
        let cd_off = self.pos - self.start;
        let cd_len = self.central.len() as u64;
        self.w.write_all(&self.central)?;
        let mut e = Vec::new();
        let z64 = self.any_zip64 || cd_off >= U32_MAX || cd_len >= U32_MAX || self.count >= 0xffff;
        if z64 {
            let z64_off = cd_off + cd_len;
            e.extend_from_slice(&0x0606_4b50u32.to_le_bytes()); // zip64 end of central directory
            e.extend_from_slice(&44u64.to_le_bytes()); // size of the rest of this record
            e.extend_from_slice(&45u16.to_le_bytes());
            e.extend_from_slice(&45u16.to_le_bytes());
            e.extend_from_slice(&0u32.to_le_bytes()); // this disk
            e.extend_from_slice(&0u32.to_le_bytes()); // cd disk
            e.extend_from_slice(&self.count.to_le_bytes());
            e.extend_from_slice(&self.count.to_le_bytes());
            e.extend_from_slice(&cd_len.to_le_bytes());
            e.extend_from_slice(&cd_off.to_le_bytes());
            e.extend_from_slice(&0x0706_4b50u32.to_le_bytes()); // zip64 EOCD locator
            e.extend_from_slice(&0u32.to_le_bytes());
            e.extend_from_slice(&z64_off.to_le_bytes());
            e.extend_from_slice(&1u32.to_le_bytes());
        }
        let c16 = if z64 { 0xffff } else { self.count as u16 };
        e.extend_from_slice(&0x0605_4b50u32.to_le_bytes());
        e.extend_from_slice(&0u16.to_le_bytes());
        e.extend_from_slice(&0u16.to_le_bytes());
        e.extend_from_slice(&c16.to_le_bytes());
        e.extend_from_slice(&c16.to_le_bytes());
        let cl32 = if cd_len >= U32_MAX {
            U32_MAX as u32
        } else {
            cd_len as u32
        };
        e.extend_from_slice(&cl32.to_le_bytes());
        let co32 = if z64 { U32_MAX as u32 } else { cd_off as u32 };
        e.extend_from_slice(&co32.to_le_bytes());
        e.extend_from_slice(&0u16.to_le_bytes());
        self.w.write_all(&e)?;
        Ok(self.pos + cd_len + e.len() as u64)
    }
}

/// Write one tensor chunk as a torch.save archive; returns its byte length.
fn write_chunk<W: Write>(
    w: &mut W,
    pos: u64,
    dtype: DType,
    sizes: &[u64],
    data: &[u8],
) -> Result<u64> {
    let mut z = ZipOut::new(w, pos);
    z.entry("archive/data.pkl", &tensor_pickle(dtype, sizes))?;
    z.entry("archive/.format_version", b"1")?;
    z.entry("archive/.storage_alignment", b"64")?;
    z.entry("archive/byteorder", b"little")?;
    z.entry("archive/data/0", data)?;
    z.entry("archive/version", b"3\n")?;
    let end = z.finish()?;
    Ok(end - pos)
}

/// Row-slab partition of dim0 with torch.chunk semantics (ceil-sized pieces).
fn slabs(d0: u64, ranks: usize) -> Vec<(u64, u64)> {
    let per = d0.div_ceil(ranks as u64).max(1);
    let mut v = Vec::new();
    let mut s = 0;
    while s < d0 {
        let e = (s + per).min(d0);
        v.push((s, e - s));
        s = e;
    }
    v
}

pub fn write_dcp(ck: &Checkpoint, sel: &[Sel], out: &Path, ranks: usize) -> Result<()> {
    if ranks == 0 {
        bail!("--dcp-ranks must be >= 1");
    }
    std::fs::create_dir_all(out)?;
    let mut files: Vec<BufWriter<File>> = Vec::new();
    let mut pos = vec![0u64; ranks];
    for r in 0..ranks {
        let p = out.join(format!("__{r}_0.distcp"));
        files.push(BufWriter::with_capacity(
            8 << 20,
            File::create(&p).with_context(|| format!("creating {}", p.display()))?,
        ));
    }
    let mut tensors = Vec::new();
    let mut chunks: Vec<ChunkRec> = Vec::new();
    for s in sel {
        let t = &ck.tensors[s.idx];
        let full = ck.read(t)?;
        let out_dtype = s.dtype;
        let row_bytes = if t.shape.is_empty() {
            t.nbytes()
        } else {
            numel(&t.shape[1..]) * t.dtype.size() as u64
        };
        let parts: Vec<(usize, u64, u64)> =
            if t.shape.is_empty() || t.shape[0] < ranks as u64 || numel(&t.shape) == 0 {
                vec![(0, 0, if t.shape.is_empty() { 1 } else { t.shape[0] })]
            } else {
                slabs(t.shape[0], ranks)
                    .into_iter()
                    .enumerate()
                    .map(|(r, (a, n))| (r, a, n))
                    .collect()
            };
        let mut rec = TensorRec {
            fqn: s.name.clone(),
            dtype: out_dtype,
            shape: t.shape.clone(),
            chunks: vec![],
        };
        for (r, start, n) in parts {
            let (offsets, sizes, data) = if t.shape.is_empty() {
                (vec![], vec![], &full[..])
            } else {
                let mut off = vec![0u64; t.shape.len()];
                off[0] = start;
                let mut sz = t.shape.clone();
                sz[0] = n;
                (
                    off,
                    sz,
                    &full[(start * row_bytes) as usize..((start + n) * row_bytes) as usize],
                )
            };
            let converted;
            let data = if out_dtype != t.dtype {
                converted = crate::cast::convert(t.dtype, out_dtype, data)?;
                &converted[..]
            } else {
                data
            };
            let len = write_chunk(&mut files[r], pos[r], out_dtype, &sizes, data)?;
            chunks.push(ChunkRec {
                fqn: s.name.clone(),
                offsets,
                sizes,
                file: format!("__{r}_0.distcp"),
                offset: pos[r],
                length: len,
            });
            pos[r] += len;
            rec.chunks.push(chunks.len() - 1);
        }
        tensors.push(rec);
        drop(full);
        ck.release(t);
    }
    for f in files {
        f.into_inner().map_err(|e| e.into_error())?.sync_all().ok();
    }
    std::fs::write(out.join(".metadata"), metadata_pickle(&tensors, &chunks))?;
    let total: u64 = sel
        .iter()
        .map(|s| ck.tensors[s.idx].numel() * s.dtype.size() as u64)
        .sum();
    eprintln!(
        "wrote DCP checkpoint: {} tensors, {} chunks, {} across {ranks} file(s) in {}",
        tensors.len(),
        chunks.len(),
        crate::dtype::human_bytes(total),
        out.display()
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn slab_split() {
        assert_eq!(slabs(10, 4), vec![(0, 3), (3, 3), (6, 3), (9, 1)]);
        assert_eq!(slabs(8, 4), vec![(0, 2), (2, 2), (4, 2), (6, 2)]);
    }
    #[test]
    fn zip64_records_roundtrip() {
        // force the zip64 layout on a small archive and read it back with our reader
        let data: Vec<u8> = (0..40u8).collect();
        let mut buf = vec![0u8; 7];
        let mut z = ZipOut::new(&mut buf, 7);
        z.force = true;
        z.entry("archive/data/0", &data).unwrap();
        let end = z.finish().unwrap();
        let arch = &buf[7..end as usize];
        let e = crate::zipread::entries(arch).unwrap();
        assert_eq!(e[0].data, &data[..]);
        assert_eq!(e[0].offset % 64, 0); // data stays 64-byte aligned within the archive
        e[0].verify().unwrap();
    }

    #[test]
    fn chunk_roundtrip() {
        let data: Vec<u8> = (0..24u8).collect();
        let mut buf = Vec::new();
        buf.extend_from_slice(b"prefix-bytes"); // chunks need not start at file offset 0
        let pos = buf.len() as u64;
        let len = write_chunk(&mut buf, pos, DType::F32, &[2, 3], &data).unwrap();
        let z = &buf[pos as usize..(pos + len) as usize];
        let v = crate::dcp::read_torch_save_tensor(z).unwrap();
        assert_eq!(v.sizes, vec![2, 3]);
        assert_eq!(v.dtype, DType::F32);
        assert_eq!(v.contiguous_bytes().unwrap(), &data[..]);
    }
}
