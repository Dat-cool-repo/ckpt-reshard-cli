//! Element types shared by all formats, with safetensors / torch name mappings.

use anyhow::{Result, bail};
use std::fmt;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DType {
    Bool,
    U8,
    I8,
    F8E5M2,
    F8E4M3,
    I16,
    U16,
    F16,
    BF16,
    I32,
    U32,
    F32,
    I64,
    U64,
    F64,
    C64,
}

impl DType {
    pub fn size(self) -> usize {
        use DType::*;
        match self {
            Bool | U8 | I8 | F8E5M2 | F8E4M3 => 1,
            I16 | U16 | F16 | BF16 => 2,
            I32 | U32 | F32 => 4,
            I64 | U64 | F64 | C64 => 8,
        }
    }

    /// safetensors header name
    pub fn st_name(self) -> &'static str {
        use DType::*;
        match self {
            Bool => "BOOL",
            U8 => "U8",
            I8 => "I8",
            F8E5M2 => "F8_E5M2",
            F8E4M3 => "F8_E4M3",
            I16 => "I16",
            U16 => "U16",
            F16 => "F16",
            BF16 => "BF16",
            I32 => "I32",
            U32 => "U32",
            F32 => "F32",
            I64 => "I64",
            U64 => "U64",
            F64 => "F64",
            C64 => "C64",
        }
    }

    pub fn from_st_name(s: &str) -> Result<DType> {
        use DType::*;
        Ok(match s {
            "BOOL" => Bool,
            "U8" => U8,
            "I8" => I8,
            "F8_E5M2" => F8E5M2,
            "F8_E4M3" => F8E4M3,
            "I16" => I16,
            "U16" => U16,
            "F16" => F16,
            "BF16" => BF16,
            "I32" => I32,
            "U32" => U32,
            "F32" => F32,
            "I64" => I64,
            "U64" => U64,
            "F64" => F64,
            "C64" => C64,
            _ => bail!("unsupported safetensors dtype {s:?}"),
        })
    }

    /// `torch.<name>` attribute names (as they appear as pickle globals).
    pub fn from_torch_name(s: &str) -> Option<DType> {
        use DType::*;
        Some(match s {
            "bool" => Bool,
            "uint8" => U8,
            "int8" => I8,
            "float8_e5m2" => F8E5M2,
            "float8_e4m3fn" => F8E4M3,
            "int16" | "short" => I16,
            "uint16" => U16,
            "float16" | "half" => F16,
            "bfloat16" => BF16,
            "int32" | "int" => I32,
            "uint32" => U32,
            "float32" | "float" => F32,
            "int64" | "long" => I64,
            "uint64" => U64,
            "float64" | "double" => F64,
            "complex64" | "cfloat" => C64,
            _ => return None,
        })
    }

    pub fn torch_name(self) -> &'static str {
        use DType::*;
        match self {
            Bool => "bool",
            U8 => "uint8",
            I8 => "int8",
            F8E5M2 => "float8_e5m2",
            F8E4M3 => "float8_e4m3fn",
            I16 => "int16",
            U16 => "uint16",
            F16 => "float16",
            BF16 => "bfloat16",
            I32 => "int32",
            U32 => "uint32",
            F32 => "float32",
            I64 => "int64",
            U64 => "uint64",
            F64 => "float64",
            C64 => "complex64",
        }
    }

    /// Legacy typed-storage class names used by `torch.save` (`torch.FloatStorage`, ...).
    pub fn from_storage_name(s: &str) -> Option<DType> {
        use DType::*;
        Some(match s {
            "BoolStorage" => Bool,
            "ByteStorage" => U8,
            "CharStorage" => I8,
            "Float8_e5m2Storage" => F8E5M2,
            "Float8_e4m3fnStorage" => F8E4M3,
            "ShortStorage" => I16,
            "UInt16Storage" => U16,
            "HalfStorage" => F16,
            "BFloat16Storage" => BF16,
            "IntStorage" => I32,
            "UInt32Storage" => U32,
            "FloatStorage" => F32,
            "LongStorage" => I64,
            "UInt64Storage" => U64,
            "DoubleStorage" => F64,
            "ComplexFloatStorage" => C64,
            _ => return None,
        })
    }

    pub fn storage_name(self) -> &'static str {
        use DType::*;
        match self {
            Bool => "BoolStorage",
            U8 => "ByteStorage",
            I8 => "CharStorage",
            F8E5M2 => "Float8_e5m2Storage",
            F8E4M3 => "Float8_e4m3fnStorage",
            I16 => "ShortStorage",
            U16 => "UInt16Storage",
            F16 => "HalfStorage",
            BF16 => "BFloat16Storage",
            I32 => "IntStorage",
            U32 => "UInt32Storage",
            F32 => "FloatStorage",
            I64 => "LongStorage",
            U64 => "UInt64Storage",
            F64 => "DoubleStorage",
            C64 => "ComplexFloatStorage",
        }
    }

    pub fn is_numeric_comparable(self) -> bool {
        !matches!(self, DType::C64)
    }

    /// Decode element `i` of a little-endian buffer to f64 (for diff statistics).
    #[inline]
    pub fn get_f64(self, b: &[u8], i: usize) -> f64 {
        use DType::*;
        let s = self.size();
        let e = &b[i * s..i * s + s];
        match self {
            Bool | U8 => e[0] as f64,
            I8 => e[0] as i8 as f64,
            F8E5M2 => f8_e5m2_to_f64(e[0]),
            F8E4M3 => f8_e4m3fn_to_f64(e[0]),
            I16 => i16::from_le_bytes([e[0], e[1]]) as f64,
            U16 => u16::from_le_bytes([e[0], e[1]]) as f64,
            F16 => half::f16::from_le_bytes([e[0], e[1]]).to_f64(),
            BF16 => half::bf16::from_le_bytes([e[0], e[1]]).to_f64(),
            I32 => i32::from_le_bytes(e.try_into().unwrap()) as f64,
            U32 => u32::from_le_bytes(e.try_into().unwrap()) as f64,
            F32 => f32::from_le_bytes(e.try_into().unwrap()) as f64,
            I64 => i64::from_le_bytes(e.try_into().unwrap()) as f64,
            U64 => u64::from_le_bytes(e.try_into().unwrap()) as f64,
            F64 => f64::from_le_bytes(e.try_into().unwrap()),
            C64 => f32::from_le_bytes(e[0..4].try_into().unwrap()) as f64,
        }
    }
}

impl fmt::Display for DType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.st_name())
    }
}

fn f8_e5m2_to_f64(v: u8) -> f64 {
    // e5m2 is the top byte of an IEEE f16
    half::f16::from_bits((v as u16) << 8).to_f64()
}

fn f8_e4m3fn_to_f64(v: u8) -> f64 {
    let sign = if v & 0x80 != 0 { -1.0 } else { 1.0 };
    let exp = ((v >> 3) & 0x0f) as i32;
    let man = (v & 0x07) as f64;
    if exp == 0x0f && (v & 0x07) == 0x07 {
        return f64::NAN;
    }
    if exp == 0 {
        sign * (man / 8.0) * 2f64.powi(-6)
    } else {
        sign * (1.0 + man / 8.0) * 2f64.powi(exp - 7)
    }
}

pub fn numel(shape: &[u64]) -> u64 {
    shape.iter().product()
}

/// Parse sizes like "5GB", "500MB", "300KiB", "1024".
pub fn parse_size(s: &str) -> Result<u64> {
    let t = s.trim();
    let idx = t
        .find(|c: char| !(c.is_ascii_digit() || c == '.'))
        .unwrap_or(t.len());
    let (num, unit) = t.split_at(idx);
    let n: f64 = num.parse().map_err(|_| anyhow::anyhow!("bad size {s:?}"))?;
    let mult: f64 = match unit.trim().to_ascii_uppercase().as_str() {
        "" | "B" => 1.0,
        "KB" | "K" => 1e3,
        "MB" | "M" => 1e6,
        "GB" | "G" => 1e9,
        "TB" | "T" => 1e12,
        "KIB" => 1024.0,
        "MIB" => 1024.0 * 1024.0,
        "GIB" => 1024.0 * 1024.0 * 1024.0,
        "TIB" => 1024f64.powi(4),
        u => bail!("unknown size unit {u:?} in {s:?}"),
    };
    Ok((n * mult) as u64)
}

pub fn human_bytes(n: u64) -> String {
    let units = ["B", "KiB", "MiB", "GiB", "TiB"];
    let mut v = n as f64;
    let mut u = 0;
    while v >= 1024.0 && u + 1 < units.len() {
        v /= 1024.0;
        u += 1;
    }
    if u == 0 {
        format!("{n} B")
    } else {
        format!("{v:.2} {}", units[u])
    }
}

pub fn human_count(n: u64) -> String {
    let v = n as f64;
    if v >= 1e9 {
        format!("{:.2}B", v / 1e9)
    } else if v >= 1e6 {
        format!("{:.2}M", v / 1e6)
    } else if v >= 1e3 {
        format!("{:.2}K", v / 1e3)
    } else {
        format!("{n}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn sizes() {
        assert_eq!(parse_size("5GB").unwrap(), 5_000_000_000);
        assert_eq!(parse_size("300KB").unwrap(), 300_000);
        assert_eq!(parse_size("2MiB").unwrap(), 2 * 1024 * 1024);
        assert_eq!(parse_size("1024").unwrap(), 1024);
    }
    #[test]
    fn f8() {
        assert_eq!(f8_e4m3fn_to_f64(0x38), 1.0);
        assert_eq!(f8_e4m3fn_to_f64(0x7e), 448.0);
        assert_eq!(f8_e5m2_to_f64(0x3c), 1.0);
    }
}
