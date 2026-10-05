//! Minimal read-only ZIP reader for `torch.save` archives (stored entries only, ZIP64 aware).
//! Works on an in-memory / mmapped byte slice and returns sub-slices, never copies.
//!
//! CRC32: every entry records the CRC from the central directory. With `--verify`
//! (`set_verify(true)`), every entry whose data is used is checked against it
//! (`ZipEntry::checked_data`), so silent corruption is reported instead of read.

use anyhow::{Result, bail};
use std::sync::atomic::{AtomicBool, Ordering};

static VERIFY: AtomicBool = AtomicBool::new(false);

/// Enable CRC32 verification of every zip entry that is read.
pub fn set_verify(on: bool) {
    VERIFY.store(on, Ordering::Relaxed);
}

pub fn verify_enabled() -> bool {
    VERIFY.load(Ordering::Relaxed)
}

pub struct ZipEntry<'a> {
    pub name: String,
    pub data: &'a [u8],
    /// byte offset of `data` inside the buffer given to `entries`
    pub offset: usize,
    /// CRC32 recorded in the central directory
    pub crc32: u32,
}

impl<'a> ZipEntry<'a> {
    /// Check the CRC32 of this entry's data against the central directory.
    pub fn verify(&self) -> Result<()> {
        let got = crc32fast::hash(self.data);
        if got != self.crc32 {
            bail!(
                "CRC32 mismatch in zip entry {:?} ({} bytes at offset {}): central directory says {:#010x}, data hashes to {:#010x} -- the file is corrupted",
                self.name,
                self.data.len(),
                self.offset,
                self.crc32,
                got
            );
        }
        Ok(())
    }
    /// The entry's data, CRC-checked first when `--verify` is on.
    pub fn checked_data(&self) -> Result<&'a [u8]> {
        if verify_enabled() {
            self.verify()?;
        }
        Ok(self.data)
    }
}

fn rd16(b: &[u8], o: usize) -> Result<u16> {
    b.get(o..o + 2)
        .map(|s| u16::from_le_bytes(s.try_into().unwrap()))
        .ok_or_else(|| anyhow::anyhow!("zip: truncated"))
}
fn rd32(b: &[u8], o: usize) -> Result<u32> {
    b.get(o..o + 4)
        .map(|s| u32::from_le_bytes(s.try_into().unwrap()))
        .ok_or_else(|| anyhow::anyhow!("zip: truncated"))
}
fn rd64(b: &[u8], o: usize) -> Result<u64> {
    b.get(o..o + 8)
        .map(|s| u64::from_le_bytes(s.try_into().unwrap()))
        .ok_or_else(|| anyhow::anyhow!("zip: truncated"))
}

pub fn is_zip(b: &[u8]) -> bool {
    b.len() >= 4 && &b[..4] == b"PK\x03\x04"
}

/// List all entries of a zip archive contained in `b` (offsets relative to `b`).
pub fn entries(b: &[u8]) -> Result<Vec<ZipEntry<'_>>> {
    // locate end-of-central-directory record
    let min = b.len().saturating_sub(22 + 65535);
    let mut eocd = None;
    let mut i = b.len().saturating_sub(22);
    loop {
        if rd32(b, i).ok() == Some(0x0605_4b50) {
            eocd = Some(i);
            break;
        }
        if i == min || i == 0 {
            break;
        }
        i -= 1;
    }
    let Some(eocd) = eocd else {
        bail!("zip: end of central directory not found")
    };
    let mut count = rd16(b, eocd + 10)? as u64;
    let mut cd_off = rd32(b, eocd + 16)? as u64;
    if (count == 0xffff || cd_off == 0xffff_ffff)
        && eocd >= 20
        && rd32(b, eocd - 20)? == 0x0706_4b50
    {
        let z64 = rd64(b, eocd - 20 + 8)? as usize;
        if rd32(b, z64)? != 0x0606_4b50 {
            bail!("zip: bad zip64 EOCD");
        }
        count = rd64(b, z64 + 32)?;
        cd_off = rd64(b, z64 + 48)?;
    }
    let mut out = Vec::new();
    let mut p = cd_off as usize;
    for _ in 0..count {
        if rd32(b, p)? != 0x0201_4b50 {
            bail!("zip: bad central directory entry");
        }
        let method = rd16(b, p + 10)?;
        let crc32 = rd32(b, p + 16)?;
        let mut csize = rd32(b, p + 20)? as u64;
        let mut usize_ = rd32(b, p + 24)? as u64;
        let nlen = rd16(b, p + 28)? as usize;
        let xlen = rd16(b, p + 30)? as usize;
        let clen = rd16(b, p + 32)? as usize;
        let mut lho = rd32(b, p + 42)? as u64;
        let name = String::from_utf8_lossy(
            b.get(p + 46..p + 46 + nlen)
                .ok_or_else(|| anyhow::anyhow!("zip: truncated"))?,
        )
        .to_string();
        // zip64 extra field
        let mut x = p + 46 + nlen;
        let xend = x + xlen;
        while x + 4 <= xend {
            let id = rd16(b, x)?;
            let sz = rd16(b, x + 2)? as usize;
            if id == 0x0001 {
                let mut q = x + 4;
                if usize_ == 0xffff_ffff {
                    usize_ = rd64(b, q)?;
                    q += 8;
                }
                if csize == 0xffff_ffff {
                    csize = rd64(b, q)?;
                    q += 8;
                }
                if lho == 0xffff_ffff {
                    lho = rd64(b, q)?;
                }
            }
            x += 4 + sz;
        }
        if method != 0 {
            bail!(
                "zip: entry {name} is compressed (method {method}); only stored entries are supported"
            );
        }
        if csize != usize_ {
            bail!("zip: entry {name} size mismatch");
        }
        let l = lho as usize;
        if rd32(b, l)? != 0x0403_4b50 {
            bail!("zip: bad local header for {name}");
        }
        let lnlen = rd16(b, l + 26)? as usize;
        let lxlen = rd16(b, l + 28)? as usize;
        let start = l + 30 + lnlen + lxlen;
        let end = start
            .checked_add(usize_ as usize)
            .ok_or_else(|| anyhow::anyhow!("zip: overflow"))?;
        if end > b.len() {
            bail!("zip: entry {name} out of bounds");
        }
        out.push(ZipEntry {
            name,
            data: &b[start..end],
            offset: start,
            crc32,
        });
        p += 46 + nlen + xlen + clen;
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn crc_check() {
        // zip with one stored entry "a" = b"hello", made by python zipfile (ZIP_STORED)
        let mut z = Vec::new();
        let data = b"hello";
        let crc = crc32fast::hash(data);
        z.extend_from_slice(&0x0403_4b50u32.to_le_bytes());
        z.extend_from_slice(&[20, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
        z.extend_from_slice(&crc.to_le_bytes());
        z.extend_from_slice(&5u32.to_le_bytes());
        z.extend_from_slice(&5u32.to_le_bytes());
        z.extend_from_slice(&1u16.to_le_bytes());
        z.extend_from_slice(&0u16.to_le_bytes());
        z.push(b'a');
        z.extend_from_slice(data);
        let cd = z.len();
        z.extend_from_slice(&0x0201_4b50u32.to_le_bytes());
        z.extend_from_slice(&[20, 0, 20, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
        z.extend_from_slice(&crc.to_le_bytes());
        z.extend_from_slice(&5u32.to_le_bytes());
        z.extend_from_slice(&5u32.to_le_bytes());
        z.extend_from_slice(&1u16.to_le_bytes());
        z.extend_from_slice(&[0u8; 12]);
        z.extend_from_slice(&0u32.to_le_bytes());
        z.push(b'a');
        let cdlen = z.len() - cd;
        z.extend_from_slice(&0x0605_4b50u32.to_le_bytes());
        z.extend_from_slice(&[0, 0, 0, 0, 1, 0, 1, 0]);
        z.extend_from_slice(&(cdlen as u32).to_le_bytes());
        z.extend_from_slice(&(cd as u32).to_le_bytes());
        z.extend_from_slice(&[0, 0]);
        let e = entries(&z).unwrap();
        assert_eq!(e[0].data, b"hello");
        e[0].verify().unwrap();
        let pos = e[0].offset;
        z[pos] ^= 1; // flip a bit in the data
        let e = entries(&z).unwrap();
        let err = format!("{:#}", e[0].verify().unwrap_err());
        assert!(err.contains("CRC32 mismatch"), "{err}");
    }
}
