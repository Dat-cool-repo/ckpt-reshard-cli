//! Unified, read-only view over a checkpoint in any supported format.
//! Files are memory-mapped; tensor data is only touched when a tensor is read or streamed.

use crate::dcp::{self, ChunkView, DcpChunkMeta, StorageInfo};
use crate::dtype::{DType, numel};
use crate::pickle::Allow;
use crate::safetensors::parse_header;
use anyhow::{Context, Result, anyhow, bail};
use memmap2::Mmap;
use std::borrow::Cow;
use std::collections::{BTreeMap, HashMap};
use std::fs::File;
use std::io::Write;
use std::path::{Path, PathBuf};

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum Format {
    Safetensors,
    HfSharded,
    Dcp,
    /// a single `torch.save` file (`.pt` / `.bin` state dict)
    TorchSave,
    /// Megatron-LM legacy `mp_rank_XX[_YYY]/model_optim_rng.pt` (TP/PP merged on read)
    Megatron,
    /// Megatron-LM `torch_dist` (DCP-based) checkpoint
    MegatronDist,
    /// DeepSpeed ZeRO checkpoint (fp32 weights reconstructed like zero_to_fp32.py)
    DeepSpeed,
}

impl Format {
    pub fn name(self) -> &'static str {
        match self {
            Format::Safetensors => "safetensors",
            Format::HfSharded => "hf-sharded-safetensors",
            Format::Dcp => "pytorch-dcp",
            Format::TorchSave => "torch-save",
            Format::Megatron => "megatron-legacy",
            Format::MegatronDist => "megatron-torch-dist",
            Format::DeepSpeed => "deepspeed-zero",
        }
    }
    /// formats whose files are DCP (chunked) underneath
    pub fn is_dcp(self) -> bool {
        matches!(self, Format::Dcp | Format::MegatronDist)
    }
}

#[derive(Clone, Debug)]
pub struct DcpChunk {
    pub meta: DcpChunkMeta,
    pub file: usize,
    pub info: StorageInfo,
}

/// Where a part of a tensor goes inside the full tensor.
#[derive(Clone, Debug)]
pub enum Place {
    /// a block at these offsets (same rank as the tensor)
    Block(Vec<u64>),
    /// a run of elements starting at this row-major element offset (flattened partitions)
    Linear(u64),
}

/// A strided view into a `torch.save` storage, placed into the full tensor.
#[derive(Clone, Debug)]
pub struct ViewPart {
    pub file: usize,
    pub storage_start: u64,
    pub storage_len: u64,
    pub storage_offset: u64,
    pub sizes: Vec<u64>,
    pub strides: Vec<u64>,
    pub place: Place,
}

impl ViewPart {
    pub fn from_ts(file: usize, t: &crate::torchsave::TsTensor, place: Place) -> ViewPart {
        ViewPart {
            file,
            storage_start: t.storage_start,
            storage_len: t.storage_len,
            storage_offset: t.storage_offset,
            sizes: t.sizes.clone(),
            strides: t.strides.clone(),
            place,
        }
    }
    /// Restrict the view to `[start, start+len)` along `dim` (placement is left as is).
    pub fn narrow(&self, dim: usize, start: u64, len: u64) -> ViewPart {
        let mut v = self.clone();
        v.storage_offset += start * self.strides[dim];
        v.sizes[dim] = len;
        v
    }
}

/// A box of a base tensor copied into the output tensor at `out_offset`.
/// `start`/`size` have the base tensor's rank. Leading dims beyond the output rank must have size 1;
/// they are dropped, which lets a layer of a stacked `[L, ...]` tensor become its own tensor.
#[derive(Clone, Debug)]
pub struct Piece {
    pub base: usize,
    pub start: Vec<u64>,
    pub size: Vec<u64>,
    pub out_offset: Vec<u64>,
}

#[derive(Clone, Debug)]
pub enum Loc {
    /// contiguous bytes inside files[file]
    Contig {
        file: usize,
        start: u64,
        end: u64,
    },
    Dcp {
        chunks: Vec<DcpChunk>,
    },
    /// assembled from strided views into torch.save storages (Megatron TP merge, DeepSpeed, .pt)
    Views {
        parts: Vec<ViewPart>,
    },
    /// assembled from boxes of `Checkpoint::base` tensors (name mapping, unstacking, unpadding)
    Derived {
        pieces: Vec<Piece>,
    },
}

#[derive(Clone, Debug)]
pub struct TensorInfo {
    pub name: String,
    pub dtype: DType,
    pub shape: Vec<u64>,
    pub loc: Loc,
}

impl TensorInfo {
    pub fn numel(&self) -> u64 {
        numel(&self.shape)
    }
    pub fn nbytes(&self) -> u64 {
        self.numel() * self.dtype.size() as u64
    }
}

pub struct MappedFile {
    pub path: PathBuf,
    pub map: Mmap,
}

#[derive(Clone, Debug)]
pub struct BytesItem {
    pub name: String,
    pub file: usize,
    pub info: StorageInfo,
}

pub struct Checkpoint {
    pub format: Format,
    /// path as given by the user
    pub path: PathBuf,
    /// directory containing the checkpoint files
    pub root: PathBuf,
    pub files: Vec<MappedFile>,
    pub tensors: Vec<TensorInfo>,
    pub bytes_items: Vec<BytesItem>,
    /// safetensors __metadata__ of the (first) file
    pub metadata: BTreeMap<String, String>,
    pub dcp_version: Option<String>,
    /// raw tensors referenced by `Loc::Derived` pieces (not listed as outputs)
    pub base: Vec<TensorInfo>,
    /// format-specific details shown by `inspect` (Megatron parallel layout, ZeRO stage, ...)
    pub info: serde_json::Map<String, serde_json::Value>,
    /// non-tensor leaves of torch.save pickles (iteration, args.*, ...)
    pub scalars: Vec<(String, serde_json::Value)>,
    /// HF config.json to write next to an HF export (Megatron -> HF conversion)
    pub hf_config: Option<serde_json::Value>,
}

/// Options that change how a checkpoint is presented.
#[derive(Clone, Debug, Default)]
pub struct OpenOpts {
    /// Megatron: map names/layouts to this HF architecture ("auto", "llama", "qwen2")
    pub megatron_hf: Option<String>,
    /// Megatron: logical vocab size (unpad embeddings / output layer); default args.vocab_size
    pub vocab_size: Option<u64>,
    /// show a format's files as stored (Megatron torch_dist: plain DCP with stacked layers)
    pub raw: bool,
}

fn mmap(path: &Path) -> Result<Mmap> {
    let f = File::open(path).with_context(|| format!("opening {}", path.display()))?;
    // SAFETY: read-only mapping; we assume the checkpoint is not modified concurrently.
    unsafe { Mmap::map(&f) }.with_context(|| format!("mmap {}", path.display()))
}

pub fn mmap_file(path: &Path) -> Result<MappedFile> {
    Ok(MappedFile {
        map: mmap(path)?,
        path: path.to_path_buf(),
    })
}

impl Checkpoint {
    pub fn new(format: Format, root: &Path) -> Checkpoint {
        Checkpoint {
            format,
            path: root.to_path_buf(),
            root: root.to_path_buf(),
            files: vec![],
            tensors: vec![],
            bytes_items: vec![],
            metadata: BTreeMap::new(),
            dcp_version: None,
            base: vec![],
            info: serde_json::Map::new(),
            scalars: vec![],
            hf_config: None,
        }
    }

    pub fn open(path: &Path) -> Result<Checkpoint> {
        Self::open_with(path, &OpenOpts::default())
    }

    pub fn open_with(path: &Path, o: &OpenOpts) -> Result<Checkpoint> {
        let mut ck = Self::open_inner(path, o, 0)?;
        ck.path = path.to_path_buf();
        Ok(ck)
    }

    fn open_inner(path: &Path, o: &OpenOpts, depth: usize) -> Result<Checkpoint> {
        if depth > 3 {
            bail!("{}: too many levels of tracker files", path.display());
        }
        let md = std::fs::metadata(path)
            .with_context(|| format!("{} does not exist", path.display()))?;
        if md.is_file() {
            let name = path.file_name().and_then(|s| s.to_str()).unwrap_or("");
            let parent = path.parent().unwrap_or(Path::new("."));
            if name.ends_with(".index.json") {
                return Self::open_hf_index(path);
            }
            if name == ".metadata" {
                return Self::open_inner(parent, o, depth + 1);
            }
            if name.ends_with("_model_states.pt") || name.ends_with("_optim_states.pt") {
                if o.raw {
                    return Self::open_torch_save(path, Allow::DeepSpeed);
                }
                return crate::deepspeed::open(parent);
            }
            if is_torch_save_file(path)? {
                let allow = if name == "model_optim_rng.pt" || name == "common.pt" {
                    Allow::Megatron
                } else {
                    Allow::Checkpoint
                };
                return Self::open_torch_save(path, allow);
            }
            return Self::open_safetensors_files(
                parent,
                vec![path.to_path_buf()],
                Format::Safetensors,
            );
        }
        // directory
        if path.join(".metadata").is_file() {
            let ck = Self::open_dcp(path)?;
            if crate::megatron::is_torch_dist(path) && !o.raw {
                return crate::megatron::from_torch_dist(ck, o);
            }
            return Ok(ck);
        }
        if let Some(sub) = crate::megatron::tracker_target(path)? {
            return Self::open_inner(&sub, o, depth + 1);
        }
        if crate::megatron::is_legacy_dir(path)? {
            return crate::megatron::open_legacy(path, o);
        }
        if let Some(sub) = crate::deepspeed::latest_target(path)? {
            return Self::open_inner(&sub, o, depth + 1);
        }
        if crate::deepspeed::is_ds_dir(path)? {
            return crate::deepspeed::open(path);
        }
        let mut idx = Vec::new();
        let mut sts = Vec::new();
        for e in std::fs::read_dir(path)? {
            let p = e?.path();
            let n = p
                .file_name()
                .and_then(|s| s.to_str())
                .unwrap_or("")
                .to_string();
            if n.ends_with(".safetensors.index.json") {
                idx.push(p);
            } else if n.ends_with(".safetensors") {
                sts.push(p);
            }
        }
        idx.sort();
        sts.sort();
        if let Some(i) = idx.first() {
            if idx.len() > 1 {
                eprintln!("warning: several index files, using {}", i.display());
            }
            return Self::open_hf_index(i);
        }
        match sts.len() {
            0 => bail!(
                "{}: no .metadata, *.safetensors.index.json, *.safetensors, Megatron mp_rank_* or DeepSpeed *_model_states.pt found",
                path.display()
            ),
            1 => Self::open_safetensors_files(path, sts, Format::Safetensors),
            _ => Self::open_safetensors_files(path, sts, Format::HfSharded),
        }
    }

    fn open_hf_index(index: &Path) -> Result<Checkpoint> {
        let dir = index.parent().unwrap_or(Path::new(".")).to_path_buf();
        let js: serde_json::Value =
            serde_json::from_slice(&std::fs::read(index)?).context("parsing index json")?;
        let wm = js["weight_map"]
            .as_object()
            .ok_or_else(|| anyhow!("index has no weight_map"))?;
        let mut files: Vec<String> = Vec::new();
        for v in wm.values() {
            let f = v.as_str().ok_or_else(|| anyhow!("bad weight_map value"))?;
            if f.contains('/') || f.contains('\\') || f.contains("..") {
                bail!("suspicious shard file name {f:?} in index");
            }
            if !files.iter().any(|x| x == f) {
                files.push(f.to_string());
            }
        }
        files.sort();
        let paths = files.iter().map(|f| dir.join(f)).collect();
        let ck = Self::open_safetensors_files(&dir, paths, Format::HfSharded)?;
        // consistency check between index and shard headers
        let mut where_: HashMap<&str, usize> = HashMap::new();
        for t in &ck.tensors {
            if let Loc::Contig { file, .. } = t.loc {
                where_.insert(&t.name, file);
            }
        }
        for (k, v) in wm {
            match where_.get(k.as_str()) {
                None => bail!("index lists {k} but no shard contains it"),
                Some(&fi) => {
                    let actual = ck.files[fi].path.file_name().unwrap().to_string_lossy();
                    if actual != v.as_str().unwrap() {
                        bail!("index says {k} is in {} but it is in {}", v, actual);
                    }
                }
            }
        }
        if where_.len() != wm.len() {
            eprintln!(
                "warning: shards contain {} tensors not listed in the index",
                where_.len() - wm.len()
            );
        }
        Ok(ck)
    }

    fn open_safetensors_files(
        root: &Path,
        paths: Vec<PathBuf>,
        format: Format,
    ) -> Result<Checkpoint> {
        let mut files = Vec::new();
        let mut tensors = Vec::new();
        let mut metadata = BTreeMap::new();
        let mut seen = HashMap::new();
        for (fi, p) in paths.into_iter().enumerate() {
            let map = mmap(&p)?;
            let h = parse_header(&map).with_context(|| format!("reading {}", p.display()))?;
            if fi == 0 {
                metadata = h.metadata.clone();
            }
            for e in h.entries {
                if seen.insert(e.name.clone(), fi).is_some() {
                    bail!("tensor {} appears in more than one shard", e.name);
                }
                tensors.push(TensorInfo {
                    name: e.name,
                    dtype: e.dtype,
                    shape: e.shape,
                    loc: Loc::Contig {
                        file: fi,
                        start: e.start,
                        end: e.end,
                    },
                });
            }
            files.push(MappedFile { path: p, map });
        }
        let mut ck = Checkpoint::new(format, root);
        ck.files = files;
        ck.tensors = tensors;
        ck.metadata = metadata;
        Ok(ck)
    }

    /// A single torch.save file: every tensor in the pickle tree under its dotted path.
    pub fn open_torch_save(path: &Path, allow: Allow) -> Result<Checkpoint> {
        let mf = mmap_file(path)?;
        let ts = crate::torchsave::TorchSave::open(&mf.map, allow)
            .with_context(|| format!("reading {}", path.display()))?;
        let mut col = crate::torchsave::Collected::default();
        ts.collect(&ts.pickle.root, "", &mut col)?;
        let mut ck = Checkpoint::new(Format::TorchSave, path.parent().unwrap_or(Path::new(".")));
        let mut seen = std::collections::HashSet::new();
        for t in col.tensors {
            if !seen.insert(t.path.clone()) {
                bail!("duplicate tensor path {}", t.path);
            }
            if crate::zipread::verify_enabled() {
                ts.verify_storage(&mf.map, &t.storage_key)?;
            }
            let zeros = vec![0; t.sizes.len()];
            ck.tensors.push(TensorInfo {
                name: t.path.clone(),
                dtype: t.dtype,
                shape: t.sizes.clone(),
                loc: Loc::Views {
                    parts: vec![ViewPart::from_ts(0, &t, Place::Block(zeros))],
                },
            });
        }
        ck.scalars = col.scalars;
        ck.files.push(mf);
        Ok(ck)
    }

    fn open_dcp(dir: &Path) -> Result<Checkpoint> {
        let raw = std::fs::read(dir.join(".metadata")).context("reading .metadata")?;
        let meta = dcp::parse_metadata(&raw)?;
        let mut file_idx: HashMap<String, usize> = HashMap::new();
        let mut files = Vec::new();
        let mut get_file = |rp: &str, files: &mut Vec<MappedFile>| -> Result<usize> {
            if let Some(&i) = file_idx.get(rp) {
                return Ok(i);
            }
            let p = dir.join(rp);
            files.push(MappedFile {
                map: mmap(&p)?,
                path: p,
            });
            file_idx.insert(rp.to_string(), files.len() - 1);
            Ok(files.len() - 1)
        };
        let mut tensors = Vec::new();
        for t in meta.tensors {
            let mut chunks = Vec::new();
            for c in t.chunks {
                let info = meta
                    .storage
                    .get(&(t.fqn.clone(), Some(c.offsets.clone())))
                    .ok_or_else(|| {
                        anyhow!("{}: no storage entry for chunk at {:?}", t.fqn, c.offsets)
                    })?
                    .clone();
                let file = get_file(&info.relative_path, &mut files)?;
                chunks.push(DcpChunk {
                    meta: c,
                    file,
                    info,
                });
            }
            tensors.push(TensorInfo {
                name: t.fqn,
                dtype: t.dtype,
                shape: t.size,
                loc: Loc::Dcp { chunks },
            });
        }
        let mut bytes_items = Vec::new();
        for b in meta.bytes_items {
            if let Some(info) = meta.storage.get(&(b.clone(), None)) {
                let file = get_file(&info.relative_path, &mut files)?;
                bytes_items.push(BytesItem {
                    name: b,
                    file,
                    info: info.clone(),
                });
            }
        }
        let mut ck = Checkpoint::new(Format::Dcp, dir);
        ck.files = files;
        ck.tensors = tensors;
        ck.bytes_items = bytes_items;
        ck.dcp_version = meta.version;
        Ok(ck)
    }

    pub fn find(&self, name: &str) -> Option<&TensorInfo> {
        self.tensors.iter().find(|t| t.name == name)
    }

    pub fn total_bytes(&self) -> u64 {
        self.tensors.iter().map(|t| t.nbytes()).sum()
    }
    pub fn total_params(&self) -> u64 {
        self.tensors.iter().map(|t| t.numel()).sum()
    }

    /// Return the full tensor as contiguous row-major little-endian bytes.
    /// Borrowed (zero-copy) for safetensors and contiguous single views; assembled otherwise.
    pub fn read(&self, t: &TensorInfo) -> Result<Cow<'_, [u8]>> {
        match &t.loc {
            Loc::Contig { file, start, end } => Ok(Cow::Borrowed(
                &self.files[*file].map[*start as usize..*end as usize],
            )),
            Loc::Dcp { chunks } => {
                if let Some(b) = self.single_contig_chunk(t, chunks)? {
                    return Ok(Cow::Borrowed(b));
                }
                self.check_coverage(t, chunks)?;
                let mut buf = vec![0u8; t.nbytes() as usize];
                for c in chunks {
                    let v = dcp::read_chunk(&self.files[c.file].map, &c.info, &t.name)?;
                    check_chunk(t, c, &v)?;
                    copy_chunk(&mut buf, &t.shape, &c.meta.offsets, &v)?;
                }
                Ok(Cow::Owned(buf))
            }
            Loc::Views { parts } => {
                if parts.len() == 1
                    && let Place::Block(off) = &parts[0].place
                    && off.iter().all(|&o| o == 0)
                    && parts[0].sizes == t.shape
                {
                    let v = self.part_view(&parts[0], t.dtype)?;
                    if let Some(b) = v.contiguous_bytes() {
                        return Ok(Cow::Borrowed(b));
                    }
                }
                let covered: u64 = parts.iter().map(|p| numel(&p.sizes)).sum();
                if covered != t.numel() {
                    bail!(
                        "{}: parts cover {covered} of {} elements",
                        t.name,
                        t.numel()
                    );
                }
                let mut buf = vec![0u8; t.nbytes() as usize];
                for p in parts {
                    self.place_part(&mut buf, t, p)?;
                }
                Ok(Cow::Owned(buf))
            }
            Loc::Derived { pieces } => {
                let covered: u64 = pieces.iter().map(|p| numel(&p.size)).sum();
                if covered != t.numel() {
                    bail!(
                        "{}: pieces cover {covered} of {} elements",
                        t.name,
                        t.numel()
                    );
                }
                if pieces.len() == 1 && pieces[0].start.iter().all(|&x| x == 0) {
                    let b = &self.base[pieces[0].base];
                    if b.shape == pieces[0].size && b.shape == t.shape {
                        return self.read(b);
                    }
                }
                let mut buf = vec![0u8; t.nbytes() as usize];
                for p in pieces {
                    let b = &self.base[p.base];
                    if p.size.len() < t.shape.len() {
                        bail!("{}: piece rank below tensor rank", t.name);
                    }
                    let drop = p.size.len() - t.shape.len();
                    if p.size[..drop].iter().any(|&x| x != 1) {
                        bail!("{}: dropped dims of a piece must have size 1", t.name);
                    }
                    let boxb = self.read_box(b, &p.start, &p.size)?;
                    let v = contiguous_view(&boxb, &p.size[drop..], t.dtype);
                    copy_chunk(&mut buf, &t.shape, &p.out_offset, &v)?;
                }
                Ok(Cow::Owned(buf))
            }
        }
    }

    fn part_view<'a>(&'a self, p: &ViewPart, dtype: DType) -> Result<ChunkView<'a>> {
        let m = &self.files[p.file].map;
        let s = p.storage_start as usize;
        let e = s
            .checked_add(p.storage_len as usize)
            .filter(|&e| e <= m.len())
            .ok_or_else(|| anyhow!("storage out of bounds"))?;
        // the view (in elements of `dtype`) must stay inside the storage bytes
        if numel(&p.sizes) > 0 {
            let mut max = p.storage_offset;
            for (sz, st) in p.sizes.iter().zip(&p.strides) {
                max = (sz - 1)
                    .checked_mul(*st)
                    .and_then(|x| x.checked_add(max))
                    .ok_or_else(|| anyhow!("view overflow"))?;
            }
            if (max + 1) * dtype.size() as u64 > p.storage_len {
                bail!(
                    "view exceeds its storage ({} bytes) as {dtype}",
                    p.storage_len
                );
            }
        }
        Ok(ChunkView {
            storage: &m[s..e],
            storage_offset: p.storage_offset,
            sizes: p.sizes.clone(),
            strides: p.strides.clone(),
            dtype,
        })
    }

    fn place_part(&self, buf: &mut [u8], t: &TensorInfo, p: &ViewPart) -> Result<()> {
        let v = self.part_view(p, t.dtype)?;
        match &p.place {
            Place::Block(off) => {
                if off.len() != t.shape.len() || p.sizes.len() != t.shape.len() {
                    bail!("{}: part rank mismatch", t.name);
                }
                if (0..t.shape.len()).any(|d| off[d] + p.sizes[d] > t.shape[d]) {
                    bail!("{}: part out of bounds", t.name);
                }
                copy_chunk(buf, &t.shape, off, &v)
            }
            Place::Linear(off) => {
                let es = t.dtype.size();
                let n = numel(&p.sizes) as usize;
                let s = *off as usize * es;
                let dst = buf
                    .get_mut(s..s + n * es)
                    .ok_or_else(|| anyhow!("{}: flat part out of bounds", t.name))?;
                if let Some(b) = v.contiguous_bytes() {
                    dst.copy_from_slice(b);
                    Ok(())
                } else {
                    let z = vec![0; p.sizes.len()];
                    copy_chunk(dst, &p.sizes, &z, &v)
                }
            }
        }
    }

    /// Read the box `[start, start+size)` of tensor `t` as contiguous row-major bytes, touching only
    /// the chunks/parts that intersect it.
    pub fn read_box(&self, t: &TensorInfo, start: &[u64], size: &[u64]) -> Result<Cow<'_, [u8]>> {
        let n = t.shape.len();
        if start.len() != n || size.len() != n {
            bail!("{}: box rank mismatch", t.name);
        }
        for d in 0..n {
            if start[d] + size[d] > t.shape[d] {
                bail!(
                    "{}: box {:?}+{:?} out of bounds of {:?}",
                    t.name,
                    start,
                    size,
                    t.shape
                );
            }
        }
        if start.iter().all(|&s| s == 0) && size == &t.shape[..] {
            return self.read(t);
        }
        let es = t.dtype.size();
        let mut buf = vec![0u8; numel(size) as usize * es];
        match &t.loc {
            Loc::Contig {
                file,
                start: s,
                end,
            } => {
                let st = &self.files[*file].map[*s as usize..*end as usize];
                copy_box(
                    &mut buf,
                    start,
                    size,
                    &[(vec![0; n], contiguous_view(st, &t.shape, t.dtype))],
                )?;
            }
            Loc::Dcp { chunks } => {
                self.check_coverage(t, chunks)?;
                let mut blocks = Vec::new();
                for c in chunks {
                    if !intersects(&c.meta.offsets, &c.meta.sizes, start, size) {
                        continue;
                    }
                    let v = dcp::read_chunk(&self.files[c.file].map, &c.info, &t.name)?;
                    check_chunk(t, c, &v)?;
                    blocks.push((c.meta.offsets.clone(), v));
                }
                copy_box(&mut buf, start, size, &blocks)?;
            }
            Loc::Views { parts } if parts.iter().all(|p| matches!(p.place, Place::Block(_))) => {
                let mut blocks = Vec::new();
                for p in parts {
                    let Place::Block(off) = &p.place else {
                        unreachable!()
                    };
                    if !intersects(off, &p.sizes, start, size) {
                        continue;
                    }
                    blocks.push((off.clone(), self.part_view(p, t.dtype)?));
                }
                copy_box(&mut buf, start, size, &blocks)?;
            }
            _ => {
                let full = self.read(t)?;
                copy_box(
                    &mut buf,
                    start,
                    size,
                    &[(vec![0; n], contiguous_view(&full, &t.shape, t.dtype))],
                )?;
            }
        }
        Ok(Cow::Owned(buf))
    }

    fn single_contig_chunk<'a>(
        &'a self,
        t: &TensorInfo,
        chunks: &[DcpChunk],
    ) -> Result<Option<&'a [u8]>> {
        if chunks.len() == 1 && chunks[0].meta.sizes == t.shape {
            let c = &chunks[0];
            let v = dcp::read_chunk(&self.files[c.file].map, &c.info, &t.name)?;
            check_chunk(t, c, &v)?;
            return Ok(v.contiguous_bytes());
        }
        Ok(None)
    }

    fn check_coverage(&self, t: &TensorInfo, chunks: &[DcpChunk]) -> Result<()> {
        let total: u64 = chunks.iter().map(|c| numel(&c.meta.sizes)).sum();
        if total != t.numel() {
            bail!(
                "{}: chunks cover {} of {} elements (incomplete or overlapping shards)",
                t.name,
                total,
                t.numel()
            );
        }
        Ok(())
    }

    /// Stream the full tensor (row-major) into `w` without materializing it when possible
    /// (safetensors: direct copy; DCP / torch.save dim-0 slabs, e.g. FSDP or TP rows: chunk by chunk).
    pub fn stream(&self, t: &TensorInfo, w: &mut dyn FnMut(&[u8]) -> Result<()>) -> Result<()> {
        if let Loc::Views { parts } = &t.loc
            && parts.len() > 1
            && let Some(order) = view_slab_order(t, parts)
        {
            for i in order {
                let v = self.part_view(&parts[i], t.dtype)?;
                match v.contiguous_bytes() {
                    Some(b) => w(b)?,
                    None => {
                        let mut buf = vec![0u8; (numel(&v.sizes) as usize) * v.dtype.size()];
                        copy_chunk(&mut buf, &v.sizes.clone(), &vec![0; v.sizes.len()], &v)?;
                        w(&buf)?
                    }
                }
            }
            self.release(t);
            return Ok(());
        }
        if let Loc::Dcp { chunks } = &t.loc
            && let Some(order) = slab_order(t, chunks)
        {
            self.check_coverage(t, chunks)?;
            for i in order {
                let c = &chunks[i];
                let v = dcp::read_chunk(&self.files[c.file].map, &c.info, &t.name)?;
                check_chunk(t, c, &v)?;
                match v.contiguous_bytes() {
                    Some(b) => w(b)?,
                    None => {
                        let mut buf = vec![0u8; (numel(&v.sizes) as usize) * v.dtype.size()];
                        copy_chunk(&mut buf, &v.sizes.clone(), &vec![0; v.sizes.len()], &v)?;
                        w(&buf)?
                    }
                }
            }
            self.release(t);
            return Ok(());
        }
        let b = self.read(t)?;
        // write in 64 MiB pieces so callers can account progress
        for piece in b.chunks(64 << 20) {
            w(piece)?;
        }
        if b.is_empty() {
            w(&[])?;
        }
        drop(b);
        self.release(t);
        Ok(())
    }

    /// Tell the kernel we are done with this tensor's source pages (drops them from our RSS;
    /// they stay in the page cache). Keeps resident memory flat while streaming huge files.
    pub fn release(&self, t: &TensorInfo) {
        let mut ranges: Vec<(usize, u64, u64)> = Vec::new();
        match &t.loc {
            Loc::Contig { file, start, end } => ranges.push((*file, *start, *end)),
            Loc::Dcp { chunks } => {
                for c in chunks {
                    ranges.push((c.file, c.info.offset, c.info.offset + c.info.length));
                }
            }
            Loc::Views { parts } => {
                for p in parts {
                    ranges.push((p.file, p.storage_start, p.storage_start + p.storage_len));
                }
            }
            Loc::Derived { pieces } => {
                for p in pieces {
                    if let Some(b) = self.base.get(p.base)
                        && !matches!(b.loc, Loc::Derived { .. })
                    {
                        self.release(b);
                    }
                }
            }
        }
        for (f, s, e) in ranges {
            let m = &self.files[f].map;
            let e = e.min(m.len() as u64);
            if e > s {
                // SAFETY: read-only shared file mapping; DONTNEED only drops resident pages,
                // later accesses re-fault identical contents from the file.
                let _ = unsafe {
                    m.unchecked_advise_range(
                        memmap2::UncheckedAdvice::DontNeed,
                        s as usize,
                        (e - s) as usize,
                    )
                };
            }
        }
    }

    pub fn stream_to(&self, t: &TensorInfo, out: &mut dyn Write) -> Result<()> {
        self.stream(t, &mut |b| Ok(out.write_all(b)?))
    }

    /// Per-file summary: (path, tensors-or-chunks, bytes, params)
    pub fn shard_summary(&self) -> Vec<(String, u64, u64, u64)> {
        let mut v: Vec<(String, u64, u64, u64)> = self
            .files
            .iter()
            .map(|f| {
                let p = f
                    .path
                    .strip_prefix(&self.root)
                    .map(|p| p.display().to_string())
                    .unwrap_or_else(|_| f.path.file_name().unwrap().to_string_lossy().to_string());
                (p, 0, 0, 0)
            })
            .collect();
        for t in self.tensors.iter().chain(&self.base) {
            match &t.loc {
                Loc::Contig { file, .. } => {
                    v[*file].1 += 1;
                    v[*file].2 += t.nbytes();
                    v[*file].3 += t.numel();
                }
                Loc::Dcp { chunks } => {
                    for c in chunks {
                        let n = numel(&c.meta.sizes);
                        v[c.file].1 += 1;
                        v[c.file].2 += n * t.dtype.size() as u64;
                        v[c.file].3 += n;
                    }
                }
                Loc::Views { parts } => {
                    for p in parts {
                        let n = numel(&p.sizes);
                        v[p.file].1 += 1;
                        v[p.file].2 += n * t.dtype.size() as u64;
                        v[p.file].3 += n;
                    }
                }
                Loc::Derived { .. } => {}
            }
        }
        v
    }

    /// With `--verify`: check the CRC32 of every entry of every zip archive this checkpoint uses
    /// (DCP torch_save chunks, torch.save files). Returns (archives, entries) checked.
    pub fn verify_all(&self) -> Result<(usize, usize)> {
        let mut archives = 0;
        let mut entries = 0;
        let mut seen = std::collections::HashSet::new();
        let mut check = |file: usize, start: u64, len: Option<u64>| -> Result<()> {
            if !seen.insert((file, start)) {
                return Ok(());
            }
            let m = &self.files[file].map;
            let s = start as usize;
            let e = match len {
                Some(l) => s + l as usize,
                None => m.len(),
            };
            let b = m
                .get(s..e)
                .ok_or_else(|| anyhow!("archive out of bounds"))?;
            if !crate::zipread::is_zip(b) {
                return Ok(());
            }
            let es = crate::zipread::entries(b)
                .with_context(|| format!("{} @{start}", self.files[file].path.display()))?;
            for z in &es {
                z.verify().with_context(|| {
                    format!(
                        "{} (archive at offset {start})",
                        self.files[file].path.display()
                    )
                })?;
            }
            archives += 1;
            entries += es.len();
            Ok(())
        };
        for t in self.tensors.iter().chain(&self.base) {
            match &t.loc {
                Loc::Dcp { chunks } => {
                    for c in chunks {
                        check(c.file, c.info.offset, Some(c.info.length))?;
                    }
                }
                Loc::Views { parts } => {
                    for p in parts {
                        check(p.file, 0, None)?;
                    }
                }
                _ => {}
            }
        }
        for b in &self.bytes_items {
            check(b.file, b.info.offset, Some(b.info.length))?;
        }
        Ok((archives, entries))
    }
}

fn check_chunk(t: &TensorInfo, c: &DcpChunk, v: &dcp::ChunkView) -> Result<()> {
    if v.dtype != t.dtype {
        bail!(
            "{}: chunk dtype {} != metadata dtype {}",
            t.name,
            v.dtype,
            t.dtype
        );
    }
    if v.sizes != c.meta.sizes {
        bail!(
            "{}: chunk shape {:?} != metadata chunk shape {:?}",
            t.name,
            v.sizes,
            c.meta.sizes
        );
    }
    Ok(())
}

/// If every chunk is a full-width slab along dim 0 (FSDP / row sharding), return them in order.
fn slab_order(t: &TensorInfo, chunks: &[DcpChunk]) -> Option<Vec<usize>> {
    if t.shape.is_empty() {
        return if chunks.len() == 1 {
            Some(vec![0])
        } else {
            None
        };
    }
    for c in chunks {
        if c.meta.offsets[1..].iter().any(|&o| o != 0) || c.meta.sizes[1..] != t.shape[1..] {
            return None;
        }
    }
    let mut order: Vec<usize> = (0..chunks.len()).collect();
    order.sort_by_key(|&i| chunks[i].meta.offsets[0]);
    let mut next = 0;
    for &i in &order {
        if chunks[i].meta.offsets[0] != next {
            return None;
        }
        next += chunks[i].meta.sizes[0];
    }
    if next != t.shape[0] {
        return None;
    }
    Some(order)
}

/// Like `slab_order` for torch.save view parts placed as dim-0 blocks.
fn view_slab_order(t: &TensorInfo, parts: &[ViewPart]) -> Option<Vec<usize>> {
    if t.shape.is_empty() {
        return None;
    }
    let mut keyed = Vec::new();
    for (i, p) in parts.iter().enumerate() {
        let Place::Block(off) = &p.place else {
            return None;
        };
        if off.len() != t.shape.len()
            || off[1..].iter().any(|&o| o != 0)
            || p.sizes[1..] != t.shape[1..]
        {
            return None;
        }
        keyed.push((off[0], i));
    }
    keyed.sort();
    let mut next = 0;
    for &(o, i) in &keyed {
        if o != next {
            return None;
        }
        next += parts[i].sizes[0];
    }
    (next == t.shape[0]).then(|| keyed.into_iter().map(|(_, i)| i).collect())
}

fn intersects(off: &[u64], sz: &[u64], start: &[u64], size: &[u64]) -> bool {
    (0..off.len()).all(|d| off[d] < start[d] + size[d] && start[d] < off[d] + sz[d])
}

/// Row-major contiguous view over `b` with shape `shape`.
pub fn contiguous_view<'a>(b: &'a [u8], shape: &[u64], dtype: DType) -> ChunkView<'a> {
    let n = shape.len();
    let mut strides = vec![1u64; n];
    for d in (0..n.saturating_sub(1)).rev() {
        strides[d] = strides[d + 1] * shape[d + 1];
    }
    ChunkView {
        storage: b,
        storage_offset: 0,
        sizes: shape.to_vec(),
        strides,
        dtype,
    }
}

/// Copy the intersection of each (offsets, view) block with the box into `buf` (box-shaped).
fn copy_box(
    buf: &mut [u8],
    start: &[u64],
    size: &[u64],
    blocks: &[(Vec<u64>, ChunkView)],
) -> Result<()> {
    let n = start.len();
    for (off, v) in blocks {
        if n == 0 {
            copy_chunk(buf, size, &[], v)?;
            continue;
        }
        let mut sub = ChunkView {
            storage: v.storage,
            storage_offset: v.storage_offset,
            sizes: vec![0; n],
            strides: v.strides.clone(),
            dtype: v.dtype,
        };
        let mut dst_off = vec![0u64; n];
        let mut empty = false;
        for d in 0..n {
            let lo = off[d].max(start[d]);
            let hi = (off[d] + v.sizes[d]).min(start[d] + size[d]);
            if lo >= hi {
                empty = true;
                break;
            }
            sub.storage_offset += (lo - off[d]) * v.strides[d];
            sub.sizes[d] = hi - lo;
            dst_off[d] = lo - start[d];
        }
        if !empty {
            copy_chunk(buf, size, &dst_off, &sub)?;
        }
    }
    Ok(())
}

/// Is `path` a torch.save archive (by magic, for .pt/.pth/.bin files)?
fn is_torch_save_file(path: &Path) -> Result<bool> {
    let name = path.file_name().and_then(|s| s.to_str()).unwrap_or("");
    if !(name.ends_with(".pt") || name.ends_with(".pth") || name.ends_with(".bin")) {
        return Ok(false);
    }
    let mut f = File::open(path).with_context(|| format!("opening {}", path.display()))?;
    let mut magic = [0u8; 4];
    use std::io::Read;
    let n = f.read(&mut magic)?;
    Ok(n == 4 && (crate::zipread::is_zip(&magic) || magic[0] == 0x80))
}

/// Copy a (possibly strided) chunk view into a row-major buffer of `full_shape` at `offsets`.
pub fn copy_chunk(
    dst: &mut [u8],
    full_shape: &[u64],
    offsets: &[u64],
    v: &dcp::ChunkView,
) -> Result<()> {
    let es = v.dtype.size();
    let n = full_shape.len();
    if n == 0 {
        let s = v.storage_offset as usize * es;
        dst[..es].copy_from_slice(&v.storage[s..s + es]);
        return Ok(());
    }
    if numel(&v.sizes) == 0 {
        return Ok(());
    }
    // destination strides (elements)
    let mut dstr = vec![1u64; n];
    for d in (0..n - 1).rev() {
        dstr[d] = dstr[d + 1] * full_shape[d + 1];
    }
    // merge trailing dims into one contiguous run where both sides are contiguous
    let mut k = n - 1; // dims k..n form the inner run
    let inner_contig = v.strides[n - 1] == 1 || v.sizes[n - 1] == 1;
    let mut run = v.sizes[n - 1];
    if inner_contig {
        // absorb dim k-1 when dims k..n are full-width in dst and src continues the run
        while k > 0
            && (k..n).all(|d| v.sizes[d] == full_shape[d])
            && (v.strides[k - 1] == run || v.sizes[k - 1] == 1)
        {
            run *= v.sizes[k - 1];
            k -= 1;
        }
    }
    let outer: Vec<u64> = v.sizes[..k].to_vec();
    let mut idx = vec![0u64; k];
    loop {
        let mut so = v.storage_offset;
        let mut d_o = 0u64;
        for d in 0..k {
            so += idx[d] * v.strides[d];
            d_o += (offsets[d] + idx[d]) * dstr[d];
        }
        for d in k..n {
            d_o += offsets[d] * dstr[d];
        }
        if inner_contig {
            let s = so as usize * es;
            let ds = d_o as usize * es;
            let len = run as usize * es;
            dst[ds..ds + len].copy_from_slice(&v.storage[s..s + len]);
        } else {
            // elementwise along last dim (k == n-1 here)
            for j in 0..v.sizes[n - 1] {
                let s = (so + j * v.strides[n - 1]) as usize * es;
                let ds = (d_o + j) as usize * es;
                dst[ds..ds + es].copy_from_slice(&v.storage[s..s + es]);
            }
        }
        // increment outer index
        let mut d = k;
        loop {
            if d == 0 {
                return Ok(());
            }
            d -= 1;
            idx[d] += 1;
            if idx[d] < outer[d] {
                break;
            }
            idx[d] = 0;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn view(storage: &[u8], sizes: Vec<u64>, strides: Vec<u64>) -> dcp::ChunkView<'_> {
        dcp::ChunkView {
            storage,
            storage_offset: 0,
            sizes,
            strides,
            dtype: DType::U8,
        }
    }
    #[test]
    fn copy_blocks() {
        // full 4x6, chunk 2x3 at (2,3)
        let mut dst = vec![0u8; 24];
        let src: Vec<u8> = (1..=6).collect();
        copy_chunk(
            &mut dst,
            &[4, 6],
            &[2, 3],
            &view(&src, vec![2, 3], vec![3, 1]),
        )
        .unwrap();
        assert_eq!(&dst[15..18], &[1, 2, 3]);
        assert_eq!(&dst[21..24], &[4, 5, 6]);
        // full-width slab 2x6 at row 1
        let mut dst = vec![0u8; 24];
        let src: Vec<u8> = (1..=12).collect();
        copy_chunk(
            &mut dst,
            &[4, 6],
            &[1, 0],
            &view(&src, vec![2, 6], vec![6, 1]),
        )
        .unwrap();
        assert_eq!(&dst[6..18], &src[..]);
        // transposed source (strides 1,3) 3x2 at (0,0) of full 3x2
        let mut dst = vec![0u8; 6];
        let src: Vec<u8> = vec![1, 2, 3, 4, 5, 6]; // storage of a 2x3 tensor; view is its transpose
        copy_chunk(
            &mut dst,
            &[3, 2],
            &[0, 0],
            &view(&src, vec![3, 2], vec![1, 3]),
        )
        .unwrap();
        assert_eq!(dst, vec![1, 4, 2, 5, 3, 6]);
        // 3-d chunk full in last two dims -> single run
        let mut dst = vec![0u8; 2 * 2 * 3];
        let src: Vec<u8> = (1..=6).collect();
        copy_chunk(
            &mut dst,
            &[2, 2, 3],
            &[1, 0, 0],
            &view(&src, vec![1, 2, 3], vec![6, 3, 1]),
        )
        .unwrap();
        assert_eq!(&dst[6..], &src[..]);
    }

    #[test]
    fn box_of_blocks() {
        // 4x4 tensor made of two 2x4 row blocks; read box rows 1..3, cols 1..3
        let a: Vec<u8> = (0..8).collect();
        let b: Vec<u8> = (8..16).collect();
        let blocks = vec![
            (vec![0, 0], view(&a, vec![2, 4], vec![4, 1])),
            (vec![2, 0], view(&b, vec![2, 4], vec![4, 1])),
        ];
        let mut buf = vec![0u8; 4];
        copy_box(&mut buf, &[1, 1], &[2, 2], &blocks).unwrap();
        assert_eq!(buf, vec![5, 6, 9, 10]);
    }
}
