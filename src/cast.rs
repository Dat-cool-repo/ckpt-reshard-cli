//! Floating-point dtype conversion (`--dtype`), bit-compatible with torch's `Tensor.to(dtype)` on CPU.
//!
//! * Every narrowing conversion rounds to nearest, ties to even (RNE), like torch.
//! * f64 sources are first rounded to f32 and then to the target. torch does the same:
//!   c10::Half/BFloat16/Float8 are constructed from `float`.
//! * fp32 -> bf16 and fp32 -> fp16 port c10's scalar code (`round_to_nearest_even`,
//!   `fp16_ieee_from_fp32_value`). Overflow goes to ±inf.
//! * fp32 -> float8_e4m3fn ports torch 2.14's `fp8e4m3fn_from_fp32_value`: RNE, **saturating**.
//!   e4m3fn has no inf; finite overflow, ±inf and roundings that would carry into the NaN pattern
//!   become ±448, and NaN stays NaN. (torch releases before the saturating change returned NaN.)
//! * fp32 -> float8_e5m2 ports `fp8e5m2_from_fp32_value`. |x| >= 61440 overflows to ±inf.
//! * NaN payloads are canonicalised. torch's vectorised kernels emit different NaN bit patterns
//!   than its scalar path (for example 0xffff vs 0x7fc0 for bf16), so only "is NaN" is torch-compatible.
//! * Widening conversions (to f32/f64) are exact.
//! * Non-floating tensors (ints, bool) and complex are never converted.

use crate::dtype::DType;
use crate::writer::glob_match;
use anyhow::{Result, bail};

pub fn is_float(d: DType) -> bool {
    matches!(
        d,
        DType::F64 | DType::F32 | DType::F16 | DType::BF16 | DType::F8E4M3 | DType::F8E5M2
    )
}

/// Parse `--dtype` values: fp32/float32, bf16/bfloat16, fp16/float16/half, fp8/fp8_e4m3/float8_e4m3fn,
/// fp8_e5m2/float8_e5m2, fp64/float64.
pub fn parse_dtype(s: &str) -> Result<DType> {
    Ok(match s.to_ascii_lowercase().as_str() {
        "fp32" | "f32" | "float32" | "float" => DType::F32,
        "bf16" | "bfloat16" => DType::BF16,
        "fp16" | "f16" | "float16" | "half" => DType::F16,
        "fp64" | "f64" | "float64" | "double" => DType::F64,
        "fp8" | "fp8_e4m3" | "e4m3" | "f8_e4m3" | "float8_e4m3fn" | "fp8_e4m3fn" => DType::F8E4M3,
        "fp8_e5m2" | "e5m2" | "f8_e5m2" | "float8_e5m2" => DType::F8E5M2,
        _ => bail!("unknown --dtype {s:?} (use fp32, bf16, fp16, fp8_e4m3, fp8_e5m2 or fp64)"),
    })
}

/// Which tensors to convert and to what.
#[derive(Clone, Debug)]
pub struct CastSpec {
    pub to: DType,
    /// tensors matching any of these globs keep their dtype (e.g. norms, router weights)
    pub exclude: Vec<String>,
}

impl CastSpec {
    /// Output dtype for a tensor named `name` stored as `src`.
    pub fn target(spec: &Option<CastSpec>, name: &str, src: DType) -> DType {
        match spec {
            Some(c) if is_float(src) && !c.exclude.iter().any(|p| glob_match(p, name)) => c.to,
            _ => src,
        }
    }
}

// ------------------------------------------------------------------ scalar kernels

#[inline]
pub fn f32_to_bf16(x: f32) -> u16 {
    if x.is_nan() {
        return 0x7fc0 | ((x.to_bits() >> 16) as u16 & 0x8000);
    }
    let b = x.to_bits();
    let bias = 0x7fff + ((b >> 16) & 1);
    (b.wrapping_add(bias) >> 16) as u16
}

#[inline]
pub fn f32_to_f16(f: f32) -> u16 {
    // c10/util/Half.h fp16_ieee_from_fp32_value (exact port; relies on IEEE f32 RNE arithmetic)
    let scale_to_inf = f32::from_bits(0x7780_0000); // 0x1.0p+112
    let scale_to_zero = f32::from_bits(0x0880_0000); // 0x1.0p-110
    let mut base = (f.abs() * scale_to_inf) * scale_to_zero;
    let w = f.to_bits();
    let shl1_w = w.wrapping_add(w);
    let sign = w & 0x8000_0000;
    let mut bias = shl1_w & 0xff00_0000;
    if bias < 0x7100_0000 {
        bias = 0x7100_0000;
    }
    base += f32::from_bits((bias >> 1) + 0x0780_0000);
    let bits = base.to_bits();
    let exp_bits = (bits >> 13) & 0x0000_7c00;
    let mantissa_bits = bits & 0x0000_0fff;
    let nonsign = exp_bits + mantissa_bits;
    ((sign >> 16)
        | if shl1_w > 0xff00_0000 {
            0x7e00
        } else {
            nonsign
        }) as u16
}

#[inline]
pub fn f32_to_e4m3fn(f: f32) -> u8 {
    // torch/headeronly/util/Float8_e4m3fn.h fp8e4m3fn_from_fp32_value (torch 2.14, saturating)
    let fp8_max: u32 = 1087 << 20; // 480.0, first value not representable
    let denorm_mask: u32 = 141 << 23;
    let mut f_bits = f.to_bits();
    let sign = f_bits & 0x8000_0000;
    f_bits ^= sign;
    let result: u8 = if f_bits >= fp8_max {
        if f_bits > 0x7f80_0000 { 0x7f } else { 0x7e } // NaN stays NaN; overflow/inf saturate
    } else if f_bits < (121 << 23) {
        // below 2^-6: denormal range, let the FPU round by adding a magic number
        let t = (f32::from_bits(f_bits) + f32::from_bits(denorm_mask)).to_bits();
        t.wrapping_sub(denorm_mask) as u8
    } else {
        let mant_odd = (f_bits >> 20) & 1;
        f_bits = f_bits.wrapping_add(((7u32.wrapping_sub(127)) << 23).wrapping_add(0x7ffff));
        f_bits = f_bits.wrapping_add(mant_odd);
        let r = (f_bits >> 20) as u8;
        if r == 0x7f { 0x7e } else { r } // rounding carried into the NaN pattern: saturate
    };
    result | (sign >> 24) as u8
}

#[inline]
pub fn f32_to_e5m2(f: f32) -> u8 {
    // c10/util/Float8_e5m2.h fp8e5m2_from_fp32_value
    let fp32_inf: u32 = 255 << 23;
    let fp8_max: u32 = 143 << 23; // 65536
    let denorm_mask: u32 = 134 << 23;
    let mut f_bits = f.to_bits();
    let sign = f_bits & 0x8000_0000;
    f_bits ^= sign;
    let result: u8 = if f_bits >= fp8_max {
        if f_bits > fp32_inf { 0x7f } else { 0x7c }
    } else if f_bits < (113 << 23) {
        let t = (f32::from_bits(f_bits) + f32::from_bits(denorm_mask)).to_bits();
        t.wrapping_sub(denorm_mask) as u8
    } else {
        let mant_odd = (f_bits >> 21) & 1;
        f_bits = f_bits.wrapping_add(((15u32.wrapping_sub(127)) << 23).wrapping_add(0xfffff));
        f_bits = f_bits.wrapping_add(mant_odd);
        (f_bits >> 21) as u8
    };
    result | (sign >> 24) as u8
}

#[inline]
pub fn e4m3fn_to_f32(v: u8) -> f32 {
    let sign = if v & 0x80 != 0 { -1.0f32 } else { 1.0 };
    let exp = ((v >> 3) & 0x0f) as i32;
    let man = (v & 0x07) as f32;
    if v & 0x7f == 0x7f {
        return f32::NAN.copysign(sign);
    }
    if exp == 0 {
        sign * man * (1.0 / 512.0) // man/8 * 2^-6
    } else {
        sign * (1.0 + man / 8.0) * f32::powi(2.0, exp - 7)
    }
}

#[inline]
pub fn e5m2_to_f32(v: u8) -> f32 {
    half::f16::from_bits((v as u16) << 8).to_f32()
}

#[inline]
fn bf16_to_f32(b: u16) -> f32 {
    f32::from_bits((b as u32) << 16)
}

#[inline]
fn f16_to_f32(b: u16) -> f32 {
    half::f16::from_bits(b).to_f32() // exact
}

/// Decode element `i` (little-endian) of a float buffer to f64 (exact for every float dtype).
#[inline]
fn load_f64(src: DType, b: &[u8]) -> f64 {
    match src {
        DType::F64 => f64::from_le_bytes(b[..8].try_into().unwrap()),
        _ => load_f32(src, b) as f64,
    }
}

/// Decode to f32: exact for every type except F64 (rounded RNE, like torch's double->float).
#[inline]
fn load_f32(src: DType, b: &[u8]) -> f32 {
    match src {
        DType::F32 => f32::from_le_bytes(b[..4].try_into().unwrap()),
        DType::F64 => f64::from_le_bytes(b[..8].try_into().unwrap()) as f32,
        DType::BF16 => bf16_to_f32(u16::from_le_bytes([b[0], b[1]])),
        DType::F16 => f16_to_f32(u16::from_le_bytes([b[0], b[1]])),
        DType::F8E4M3 => e4m3fn_to_f32(b[0]),
        DType::F8E5M2 => e5m2_to_f32(b[0]),
        _ => unreachable!(),
    }
}

/// Convert a buffer of whole `src` elements into `dst` elements, appending to `out`.
pub fn convert_into(src: DType, dst: DType, input: &[u8], out: &mut Vec<u8>) -> Result<()> {
    let ss = src.size();
    if !input.len().is_multiple_of(ss) {
        bail!(
            "cast: buffer of {} bytes is not a whole number of {src} elements",
            input.len()
        );
    }
    if src == dst {
        out.extend_from_slice(input);
        return Ok(());
    }
    if !is_float(src) || !is_float(dst) {
        bail!("cast: {src} -> {dst} is not a floating-point conversion");
    }
    let n = input.len() / ss;
    out.reserve(n * dst.size());
    let it = input.chunks_exact(ss);
    match dst {
        DType::F64 => it.for_each(|e| out.extend_from_slice(&load_f64(src, e).to_le_bytes())),
        DType::F32 => it.for_each(|e| out.extend_from_slice(&load_f32(src, e).to_le_bytes())),
        DType::BF16 => {
            it.for_each(|e| out.extend_from_slice(&f32_to_bf16(load_f32(src, e)).to_le_bytes()))
        }
        DType::F16 => {
            it.for_each(|e| out.extend_from_slice(&f32_to_f16(load_f32(src, e)).to_le_bytes()))
        }
        DType::F8E4M3 => it.for_each(|e| out.push(f32_to_e4m3fn(load_f32(src, e)))),
        DType::F8E5M2 => it.for_each(|e| out.push(f32_to_e5m2(load_f32(src, e)))),
        _ => unreachable!(),
    }
    Ok(())
}

pub fn convert(src: DType, dst: DType, input: &[u8]) -> Result<Vec<u8>> {
    let mut v = Vec::new();
    convert_into(src, dst, input, &mut v)?;
    Ok(v)
}

/// A byte sink (receives a tensor's bytes in pieces).
pub type Sink<'a> = dyn FnMut(&[u8]) -> Result<()> + 'a;

/// Wrap a byte sink so that `src` elements written to it come out as `dst` elements.
/// Pieces need not be element-aligned (a partial element is carried to the next call).
pub struct CastSink<'a> {
    src: DType,
    dst: DType,
    carry: Vec<u8>,
    buf: Vec<u8>,
    inner: &'a mut Sink<'a>,
}

impl<'a> CastSink<'a> {
    pub fn new(src: DType, dst: DType, inner: &'a mut Sink<'a>) -> Self {
        CastSink {
            src,
            dst,
            carry: Vec::new(),
            buf: Vec::new(),
            inner,
        }
    }
    pub fn write(&mut self, mut b: &[u8]) -> Result<()> {
        if self.src == self.dst {
            return (self.inner)(b);
        }
        let es = self.src.size();
        if !self.carry.is_empty() {
            let need = es - self.carry.len();
            let take = need.min(b.len());
            self.carry.extend_from_slice(&b[..take]);
            b = &b[take..];
            if self.carry.len() < es {
                return Ok(());
            }
            self.buf.clear();
            convert_into(self.src, self.dst, &self.carry, &mut self.buf)?;
            self.carry.clear();
            (self.inner)(&self.buf)?;
        }
        let whole = b.len() / es * es;
        // convert in bounded pieces so a huge tensor never needs a second full-size buffer
        for piece in b[..whole].chunks(16 << 20) {
            self.buf.clear();
            convert_into(self.src, self.dst, piece, &mut self.buf)?;
            (self.inner)(&self.buf)?;
        }
        self.carry.extend_from_slice(&b[whole..]);
        Ok(())
    }
    pub fn finish(self) -> Result<()> {
        if !self.carry.is_empty() {
            bail!("cast: trailing partial element");
        }
        Ok(())
    }
}

/// Stream `produce`'s bytes (of dtype `src`) into `w` converted to `dst`.
pub fn cast_stream(
    src: DType,
    dst: DType,
    w: &mut Sink,
    produce: &mut dyn FnMut(&mut Sink) -> Result<()>,
) -> Result<()> {
    if src == dst {
        return produce(w);
    }
    let mut sink = CastSink::new(src, dst, w);
    produce(&mut |b| sink.write(b))?;
    sink.finish()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bf16_rne() {
        assert_eq!(f32_to_bf16(1.0), 0x3f80);
        // 1 + 2^-8 is exactly half way between bf16 1.0 and 1.0078125 -> ties to even (1.0)
        assert_eq!(f32_to_bf16(f32::from_bits(0x3f80_8000)), 0x3f80);
        // 1.0078125 + half ulp -> ties to even (up to 1.015625 = 0x3f82)
        assert_eq!(f32_to_bf16(f32::from_bits(0x3f81_8000)), 0x3f82);
        assert_eq!(f32_to_bf16(f32::from_bits(0x3f80_8001)), 0x3f81);
        assert_eq!(f32_to_bf16(f32::MAX), 0x7f80); // overflow to inf
        assert!(bf16_to_f32(f32_to_bf16(f32::NAN)).is_nan());
    }

    #[test]
    fn f16_rne_matches_half_crate() {
        // half's from_f32 is also RNE; check a sweep of bit patterns (non-NaN)
        let mut x: u32 = 0;
        while x < u32::MAX - 9973 {
            let f = f32::from_bits(x);
            if !f.is_nan() {
                assert_eq!(f32_to_f16(f), half::f16::from_f32(f).to_bits(), "{x:#x}");
            }
            x += 9973;
        }
        assert_eq!(f32_to_f16(65520.0), 0x7c00); // rounds to inf
        assert_eq!(f32_to_f16(65519.0), 0x7bff);
    }

    #[test]
    fn fp8_values() {
        assert_eq!(f32_to_e4m3fn(1.0), 0x38);
        assert_eq!(f32_to_e4m3fn(448.0), 0x7e);
        assert_eq!(f32_to_e4m3fn(464.0), 0x7e); // midpoint 448..480: ties to even (448)
        assert_eq!(f32_to_e4m3fn(470.0), 0x7e); // would round into the NaN slot: saturates
        assert_eq!(f32_to_e4m3fn(480.0), 0x7e); // overflow saturates (no inf in e4m3fn)
        assert_eq!(f32_to_e4m3fn(-f32::INFINITY), 0xfe);
        assert_eq!(f32_to_e4m3fn(f32::NAN), 0x7f);
        assert_eq!(f32_to_e4m3fn(2f32.powi(-9)), 0x01); // smallest denormal
        assert_eq!(f32_to_e4m3fn(2f32.powi(-10)), 0x00); // tie -> even (0)
        assert_eq!(f32_to_e5m2(1.0), 0x3c);
        assert_eq!(f32_to_e5m2(57344.0), 0x7b);
        assert_eq!(f32_to_e5m2(f32::INFINITY), 0x7c);
        assert_eq!(f32_to_e5m2(70000.0), 0x7c);
        assert!(e4m3fn_to_f32(0x7f).is_nan());
        for v in 0..=255u8 {
            if v & 0x7f != 0x7f {
                assert_eq!(f32_to_e4m3fn(e4m3fn_to_f32(v)), v, "{v:#x}");
            }
            if v & 0x7f <= 0x7c {
                assert_eq!(f32_to_e5m2(e5m2_to_f32(v)), v, "{v:#x}");
            }
        }
    }

    #[test]
    fn sink_carries_partial_elements() {
        let vals: Vec<f32> = vec![1.0, -2.5, 3.25];
        let bytes: Vec<u8> = vals.iter().flat_map(|v| v.to_le_bytes()).collect();
        let mut out = Vec::new();
        {
            let mut w = |b: &[u8]| -> Result<()> {
                out.extend_from_slice(b);
                Ok(())
            };
            let mut s = CastSink::new(DType::F32, DType::BF16, &mut w);
            for piece in bytes.chunks(3) {
                s.write(piece).unwrap();
            }
            s.finish().unwrap();
        }
        assert_eq!(out, convert(DType::F32, DType::BF16, &bytes).unwrap());
        assert_eq!(out.len(), 6);
    }
}
