//! Minimal safetensors header parser and streaming writer.

use crate::dtype::{DType, numel};
use anyhow::{Context, Result, bail};
use std::collections::BTreeMap;
use std::io::Write;

#[derive(Clone, Debug)]
pub struct StEntry {
    pub name: String,
    pub dtype: DType,
    pub shape: Vec<u64>,
    /// absolute byte range inside the buffer the header was parsed from
    pub start: u64,
    pub end: u64,
}

#[derive(Clone, Debug, Default)]
pub struct StHeader {
    pub entries: Vec<StEntry>,
    pub metadata: BTreeMap<String, String>,
    pub header_len: u64,
}

const MAX_HEADER: u64 = 512 * 1024 * 1024;

/// Parse a safetensors header from `buf` (which starts at the safetensors blob).
/// Entries are returned in file (offset) order with absolute offsets relative to `buf`.
pub fn parse_header(buf: &[u8]) -> Result<StHeader> {
    if buf.len() < 8 {
        bail!("file too small for safetensors");
    }
    let n = u64::from_le_bytes(buf[..8].try_into().unwrap());
    if n > MAX_HEADER || 8 + n > buf.len() as u64 {
        bail!("invalid safetensors header length {n}");
    }
    let hdr: serde_json::Map<String, serde_json::Value> =
        serde_json::from_slice(&buf[8..8 + n as usize])
            .context("safetensors header is not valid JSON")?;
    let base = 8 + n;
    let data_len = buf.len() as u64 - base;
    let mut out = StHeader {
        header_len: base,
        ..Default::default()
    };
    for (k, v) in hdr {
        if k == "__metadata__" {
            if let Some(m) = v.as_object() {
                for (mk, mv) in m {
                    out.metadata.insert(
                        mk.clone(),
                        mv.as_str()
                            .map(|s| s.to_string())
                            .unwrap_or_else(|| mv.to_string()),
                    );
                }
            }
            continue;
        }
        let dtype = DType::from_st_name(v["dtype"].as_str().unwrap_or(""))
            .with_context(|| format!("tensor {k}"))?;
        let shape: Vec<u64> = v["shape"]
            .as_array()
            .with_context(|| format!("tensor {k}: missing shape"))?
            .iter()
            .map(|x| x.as_u64().context("bad shape"))
            .collect::<Result<_>>()?;
        let offs = v["data_offsets"]
            .as_array()
            .with_context(|| format!("tensor {k}: missing data_offsets"))?;
        if offs.len() != 2 {
            bail!("tensor {k}: bad data_offsets");
        }
        let (s, e) = (
            offs[0].as_u64().unwrap_or(u64::MAX),
            offs[1].as_u64().unwrap_or(u64::MAX),
        );
        if s > e || e > data_len {
            bail!(
                "tensor {k}: data_offsets [{s},{e}] out of bounds (data section {data_len} bytes)"
            );
        }
        if e - s != numel(&shape) * dtype.size() as u64 {
            bail!(
                "tensor {k}: byte length {} does not match shape {:?} x {}",
                e - s,
                shape,
                dtype
            );
        }
        out.entries.push(StEntry {
            name: k,
            dtype,
            shape,
            start: base + s,
            end: base + e,
        });
    }
    out.entries.sort_by_key(|e| e.start);
    Ok(out)
}

/// A tensor to be written: header info only; data is streamed later in the same order.
#[derive(Clone, Debug)]
pub struct OutTensor {
    pub name: String,
    pub dtype: DType,
    pub shape: Vec<u64>,
}

impl OutTensor {
    pub fn nbytes(&self) -> u64 {
        numel(&self.shape) * self.dtype.size() as u64
    }
}

/// Build the header bytes (8-byte length prefix + JSON padded with spaces to 8-byte alignment).
pub fn build_header(tensors: &[OutTensor], metadata: &BTreeMap<String, String>) -> Result<Vec<u8>> {
    let mut m = serde_json::Map::new();
    if !metadata.is_empty() {
        let mm: serde_json::Map<String, serde_json::Value> = metadata
            .iter()
            .map(|(k, v)| (k.clone(), serde_json::Value::String(v.clone())))
            .collect();
        m.insert("__metadata__".into(), serde_json::Value::Object(mm));
    }
    let mut off = 0u64;
    for t in tensors {
        let n = t.nbytes();
        if m.contains_key(&t.name) {
            bail!("duplicate tensor name {}", t.name);
        }
        m.insert(
            t.name.clone(),
            serde_json::json!({"dtype": t.dtype.st_name(), "shape": t.shape, "data_offsets": [off, off + n]}),
        );
        off += n;
    }
    let mut js = serde_json::to_vec(&serde_json::Value::Object(m))?;
    while (js.len() + 8) % 8 != 0 {
        js.push(b' ');
    }
    let mut out = Vec::with_capacity(js.len() + 8);
    out.extend_from_slice(&(js.len() as u64).to_le_bytes());
    out.extend_from_slice(&js);
    Ok(out)
}

/// Streaming writer: header first, then each tensor's bytes in declaration order.
pub struct StWriter<W: Write> {
    w: W,
    expected: Vec<u64>,
    idx: usize,
    written_in_cur: u64,
}

impl<W: Write> StWriter<W> {
    pub fn new(
        mut w: W,
        tensors: &[OutTensor],
        metadata: &BTreeMap<String, String>,
    ) -> Result<Self> {
        let h = build_header(tensors, metadata)?;
        w.write_all(&h)?;
        Ok(StWriter {
            w,
            expected: tensors.iter().map(|t| t.nbytes()).collect(),
            idx: 0,
            written_in_cur: 0,
        })
    }
    /// Write (part of) the current tensor's bytes.
    pub fn write_part(&mut self, b: &[u8]) -> Result<()> {
        if self.idx >= self.expected.len() {
            bail!("more tensor data than declared");
        }
        self.written_in_cur += b.len() as u64;
        if self.written_in_cur > self.expected[self.idx] {
            bail!("tensor {} overflow", self.idx);
        }
        self.w.write_all(b)?;
        Ok(())
    }
    /// Mark the current tensor complete.
    pub fn end_tensor(&mut self) -> Result<()> {
        if self.written_in_cur != self.expected[self.idx] {
            bail!(
                "tensor {}: wrote {} of {} bytes",
                self.idx,
                self.written_in_cur,
                self.expected[self.idx]
            );
        }
        self.idx += 1;
        self.written_in_cur = 0;
        Ok(())
    }
    pub fn finish(mut self) -> Result<W> {
        if self.idx != self.expected.len() {
            bail!(
                "only {} of {} tensors written",
                self.idx,
                self.expected.len()
            );
        }
        self.w.flush()?;
        Ok(self.w)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn roundtrip_header() {
        let ts = vec![
            OutTensor {
                name: "a".into(),
                dtype: DType::F32,
                shape: vec![2, 3],
            },
            OutTensor {
                name: "b".into(),
                dtype: DType::BF16,
                shape: vec![],
            },
        ];
        let mut md = BTreeMap::new();
        md.insert("format".into(), "pt".into());
        let mut w = StWriter::new(Vec::new(), &ts, &md).unwrap();
        w.write_part(&[0u8; 24]).unwrap();
        w.end_tensor().unwrap();
        w.write_part(&[1u8, 2]).unwrap();
        w.end_tensor().unwrap();
        let buf = w.finish().unwrap();
        assert_eq!(u64::from_le_bytes(buf[..8].try_into().unwrap()) % 8, 0);
        let h = parse_header(&buf).unwrap();
        assert_eq!(h.entries.len(), 2);
        assert_eq!(h.entries[1].name, "b");
        assert_eq!(
            &buf[h.entries[1].start as usize..h.entries[1].end as usize],
            &[1, 2]
        );
        assert_eq!(h.metadata["format"], "pt");
    }
}
