//! Per-tensor comparison of two checkpoints (any formats).

use crate::ckpt::Checkpoint;
use crate::dtype::DType;
use crate::writer::glob_match;
use anyhow::Result;
use rayon::prelude::*;
use serde::Serialize;
use std::collections::HashMap;

#[derive(Clone, Debug, Default)]
pub struct DiffOpts {
    pub tolerance: f64,
    pub strip_prefix_a: Option<String>,
    pub strip_prefix_b: Option<String>,
    pub include: Vec<String>,
    pub ignore_missing: bool,
}

#[derive(Debug, Serialize, Clone, Copy, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
pub enum Status {
    Identical,
    WithinTolerance,
    Different,
    ShapeMismatch,
    DtypeMismatch,
}

#[derive(Debug, Serialize, Clone)]
pub struct TensorDiff {
    pub name: String,
    pub status: Status,
    pub dtype_a: String,
    pub dtype_b: String,
    pub shape_a: Vec<u64>,
    pub shape_b: Vec<u64>,
    pub numel: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_abs: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mean_abs: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rel_l2: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cosine: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub changed_frac: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub nan_mismatch: Option<u64>,
}

#[derive(Debug, Serialize)]
pub struct DiffReport {
    pub a: String,
    pub b: String,
    pub tolerance: f64,
    pub only_in_a: Vec<String>,
    pub only_in_b: Vec<String>,
    pub compared: usize,
    pub identical: usize,
    pub within_tolerance: usize,
    pub different: usize,
    pub shape_mismatch: usize,
    pub dtype_mismatch: usize,
    pub equal: bool,
    pub tensors: Vec<TensorDiff>,
}

fn map_name(n: &str, strip: &Option<String>) -> String {
    match strip {
        Some(p) => n.strip_prefix(p.as_str()).unwrap_or(n).to_string(),
        None => n.to_string(),
    }
}

#[derive(Default, Clone, Copy)]
struct Acc {
    max_abs: f64,
    sum_abs: f64,
    dd: f64,
    aa: f64,
    bb: f64,
    ab: f64,
    changed: u64,
    nan_mismatch: u64,
}

const BLK: usize = 1 << 14;

fn decode(dt: DType, b: &[u8], start: usize, out: &mut [f64]) {
    match dt {
        DType::F32 => {
            for (i, o) in out.iter_mut().enumerate() {
                let j = (start + i) * 4;
                *o = f32::from_le_bytes(b[j..j + 4].try_into().unwrap()) as f64;
            }
        }
        DType::BF16 => {
            for (i, o) in out.iter_mut().enumerate() {
                let j = (start + i) * 2;
                *o = f32::from_bits((u16::from_le_bytes([b[j], b[j + 1]]) as u32) << 16) as f64;
            }
        }
        DType::F16 => {
            for (i, o) in out.iter_mut().enumerate() {
                let j = (start + i) * 2;
                *o = half::f16::from_le_bytes([b[j], b[j + 1]]).to_f64();
            }
        }
        _ => {
            for (i, o) in out.iter_mut().enumerate() {
                *o = dt.get_f64(b, start + i);
            }
        }
    }
}

fn stats(da: DType, a: &[u8], db: DType, b: &[u8], n: usize, tol: f64) -> Acc {
    let mut acc = Acc::default();
    let mut xa = vec![0f64; BLK];
    let mut xb = vec![0f64; BLK];
    let mut s = 0;
    while s < n {
        let m = BLK.min(n - s);
        decode(da, a, s, &mut xa[..m]);
        decode(db, b, s, &mut xb[..m]);
        for i in 0..m {
            let (x, y) = (xa[i], xb[i]);
            if x.is_nan() || y.is_nan() {
                if !(x.is_nan() && y.is_nan()) {
                    acc.nan_mismatch += 1;
                    acc.changed += 1;
                }
                continue;
            }
            let d = (x - y).abs();
            if d.is_nan() {
                // inf - inf with equal signs is equal; different signs differ
                if x != y {
                    acc.changed += 1;
                    acc.max_abs = f64::INFINITY;
                }
                continue;
            }
            if d > acc.max_abs {
                acc.max_abs = d;
            }
            if d > tol || (tol == 0.0 && x != y) {
                acc.changed += 1;
            }
            acc.sum_abs += d;
            acc.dd += d * d;
            acc.aa += x * x;
            acc.bb += y * y;
            acc.ab += x * y;
        }
        s += m;
    }
    acc
}

pub fn diff(a: &Checkpoint, b: &Checkpoint, o: &DiffOpts) -> Result<DiffReport> {
    let keep = |n: &str| o.include.is_empty() || o.include.iter().any(|p| glob_match(p, n));
    let ma: HashMap<String, usize> = a
        .tensors
        .iter()
        .enumerate()
        .map(|(i, t)| (map_name(&t.name, &o.strip_prefix_a), i))
        .filter(|(n, _)| keep(n))
        .collect();
    let mb: HashMap<String, usize> = b
        .tensors
        .iter()
        .enumerate()
        .map(|(i, t)| (map_name(&t.name, &o.strip_prefix_b), i))
        .filter(|(n, _)| keep(n))
        .collect();
    let mut only_a: Vec<String> = ma
        .keys()
        .filter(|k| !mb.contains_key(*k))
        .cloned()
        .collect();
    let mut only_b: Vec<String> = mb
        .keys()
        .filter(|k| !ma.contains_key(*k))
        .cloned()
        .collect();
    only_a.sort();
    only_b.sort();
    let mut common: Vec<(String, usize, usize)> = ma
        .iter()
        .filter_map(|(k, &i)| mb.get(k).map(|&j| (k.clone(), i, j)))
        .collect();
    // keep A's order
    common.sort_by_key(|c| c.1);

    let results: Vec<Result<TensorDiff>> = common
        .par_iter()
        .map(|(name, i, j)| {
            let ta = &a.tensors[*i];
            let tb = &b.tensors[*j];
            let mut d = TensorDiff {
                name: name.clone(),
                status: Status::Identical,
                dtype_a: ta.dtype.to_string(),
                dtype_b: tb.dtype.to_string(),
                shape_a: ta.shape.clone(),
                shape_b: tb.shape.clone(),
                numel: ta.numel(),
                max_abs: None,
                mean_abs: None,
                rel_l2: None,
                cosine: None,
                changed_frac: None,
                nan_mismatch: None,
            };
            if ta.shape != tb.shape {
                d.status = Status::ShapeMismatch;
                return Ok(d);
            }
            let ba = a.read(ta)?;
            let bb = b.read(tb)?;
            if ta.dtype == tb.dtype && ba[..] == bb[..] {
                drop((ba, bb));
                a.release(ta);
                b.release(tb);
                d.max_abs = Some(0.0);
                d.cosine = Some(1.0);
                d.changed_frac = Some(0.0);
                return Ok(d);
            }
            if !ta.dtype.is_numeric_comparable() || !tb.dtype.is_numeric_comparable() {
                d.status = if ta.dtype != tb.dtype {
                    Status::DtypeMismatch
                } else {
                    Status::Different
                };
                return Ok(d);
            }
            let n = ta.numel() as usize;
            let acc = stats(ta.dtype, &ba, tb.dtype, &bb, n, o.tolerance);
            drop((ba, bb));
            a.release(ta);
            b.release(tb);
            d.max_abs = Some(acc.max_abs);
            d.mean_abs = Some(if n > 0 { acc.sum_abs / n as f64 } else { 0.0 });
            d.rel_l2 = Some(if acc.aa > 0.0 {
                (acc.dd / acc.aa).sqrt()
            } else {
                acc.dd.sqrt()
            });
            d.cosine = Some(if acc.aa > 0.0 && acc.bb > 0.0 {
                acc.ab / (acc.aa.sqrt() * acc.bb.sqrt())
            } else if acc.aa == acc.bb {
                1.0
            } else {
                0.0
            });
            d.changed_frac = Some(if n > 0 {
                acc.changed as f64 / n as f64
            } else {
                0.0
            });
            if acc.nan_mismatch > 0 {
                d.nan_mismatch = Some(acc.nan_mismatch);
            }
            d.status = if ta.dtype != tb.dtype {
                Status::DtypeMismatch
            } else if acc.max_abs <= o.tolerance && acc.nan_mismatch == 0 {
                Status::WithinTolerance
            } else {
                Status::Different
            };
            Ok(d)
        })
        .collect();
    let tensors: Vec<TensorDiff> = results.into_iter().collect::<Result<_>>()?;
    let count = |s: Status| tensors.iter().filter(|t| t.status == s).count();
    let mut r = DiffReport {
        a: a.path.display().to_string(),
        b: b.path.display().to_string(),
        tolerance: o.tolerance,
        compared: tensors.len(),
        identical: count(Status::Identical),
        within_tolerance: count(Status::WithinTolerance),
        different: count(Status::Different),
        shape_mismatch: count(Status::ShapeMismatch),
        dtype_mismatch: count(Status::DtypeMismatch),
        only_in_a: only_a,
        only_in_b: only_b,
        equal: false,
        tensors,
    };
    r.equal = r.different == 0
        && r.shape_mismatch == 0
        && r.dtype_mismatch == 0
        && (o.ignore_missing || (r.only_in_a.is_empty() && r.only_in_b.is_empty()));
    Ok(r)
}

pub fn print_report(r: &DiffReport, all: bool) {
    println!("A: {}\nB: {}", r.a, r.b);
    for n in &r.only_in_a {
        println!("  only in A: {n}");
    }
    for n in &r.only_in_b {
        println!("  only in B: {n}");
    }
    let shown: Vec<&TensorDiff> = r
        .tensors
        .iter()
        .filter(|t| all || t.status != Status::Identical)
        .collect();
    if !shown.is_empty() {
        println!(
            "  {:<52} {:<16} {:>11} {:>11} {:>11} {:>10} {:>9}",
            "tensor", "status", "max_abs", "mean_abs", "rel_l2", "cosine", "changed"
        );
        for t in shown {
            let f = |v: Option<f64>| v.map(|x| format!("{x:.3e}")).unwrap_or_else(|| "-".into());
            let extra = match t.status {
                Status::ShapeMismatch => format!("  {:?} vs {:?}", t.shape_a, t.shape_b),
                Status::DtypeMismatch => format!("  {} vs {}", t.dtype_a, t.dtype_b),
                _ => String::new(),
            };
            println!(
                "  {:<52} {:<16} {:>11} {:>11} {:>11} {:>10} {:>8.4}%{}",
                t.name,
                format!("{:?}", t.status),
                f(t.max_abs),
                f(t.mean_abs),
                f(t.rel_l2),
                t.cosine
                    .map(|x| format!("{x:.6}"))
                    .unwrap_or_else(|| "-".into()),
                t.changed_frac.unwrap_or(0.0) * 100.0,
                extra
            );
        }
    }
    println!(
        "compared {} tensors: {} identical, {} within tolerance ({}), {} different, {} shape mismatch, {} dtype mismatch; {} only in A, {} only in B",
        r.compared,
        r.identical,
        r.within_tolerance,
        r.tolerance,
        r.different,
        r.shape_mismatch,
        r.dtype_mismatch,
        r.only_in_a.len(),
        r.only_in_b.len()
    );
    println!(
        "{}",
        if r.equal {
            "RESULT: EQUAL"
        } else {
            "RESULT: DIFFERENT"
        }
    );
}
