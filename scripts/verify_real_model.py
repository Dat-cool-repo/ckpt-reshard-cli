#!/usr/bin/env python3
"""verify_real_model.py ORIG_DIR OUT_DIR "TP..." [--logits]

Checks ckpt outputs (made with the commands in README "Real models") against the original HF checkpoint with torch/safetensors/transformers:
round trips are bit-identical, dtype casts equal torch.Tensor.to, every TP rank tensor equals the
torch slicing implied by examples/llama_tp_rules.yaml (GQA heads, kv replication, vocab padding),
and (with --logits) the round-tripped models give identical logits.
"""
import gc
import json
import shutil
import sys
from pathlib import Path

import torch
from safetensors import safe_open

orig, out = Path(sys.argv[1]), Path(sys.argv[2])
tps = [int(x) for x in sys.argv[3].split()]
do_logits = "--logits" in sys.argv
cfg = json.loads((orig / "config.json").read_text())
H, KV = cfg["num_attention_heads"], cfg["num_key_value_heads"]
fails = []
report = {}


class Tensors:
    """name -> lazily loaded tensor over a file or a directory of safetensors files."""

    def __init__(self, p: Path):
        files = [p] if p.is_file() else sorted(p.glob("*.safetensors"))
        self.where = {}
        self.handles = [safe_open(str(f), framework="pt") for f in files]
        for h in self.handles:
            for k in h.keys():
                self.where[k] = h

    def keys(self):
        return sorted(self.where)

    def get(self, k):
        return self.where[k].get_tensor(k)


def bits(t):
    """bit pattern view, so NaN payloads and -0.0 compare exactly"""
    return t.contiguous().view({1: torch.uint8, 2: torch.int16, 4: torch.int32, 8: torch.int64}[t.element_size()])


def same(a, b):
    return a.dtype == b.dtype and a.shape == b.shape and torch.equal(bits(a), bits(b))


O = Tensors(orig)
names = O.keys()


def check_identical(label, p, cast=None):
    T = Tensors(p)
    if T.keys() != names:
        fails.append(f"{label}: tensor names differ")
        return
    bad = 0
    for k in names:
        want = O.get(k)
        if cast is not None:
            want = want.to(cast)
        if not same(T.get(k), want):
            bad += 1
            if bad < 4:
                fails.append(f"{label}: {k} differs")
    report[label] = f"{len(names)} tensors " + ("bit-identical" if not bad else f"{bad} DIFFER")


def expected_shard(k, t, tp, r):
    """torch slicing for rank r of tp, following examples/llama_tp_rules.yaml"""
    def chunk(x, dim):
        return x.chunk(tp, dim=dim)[r]
    if k.endswith((".self_attn.q_proj.weight", ".self_attn.q_proj.bias")):
        return chunk(t, 0)  # H % tp == 0, heads are contiguous rows
    if k.endswith((".self_attn.k_proj.weight", ".self_attn.k_proj.bias",
                   ".self_attn.v_proj.weight", ".self_attn.v_proj.bias")):
        if KV % tp == 0:
            return chunk(t, 0)
        rep = tp // KV  # kv head replicated on `rep` consecutive ranks
        hd = t.shape[0] // KV
        h = r // rep
        return t.narrow(0, h * hd, hd)
    if k.endswith(".self_attn.o_proj.weight") or k.endswith(".mlp.down_proj.weight"):
        return chunk(t, 1)
    if k.endswith((".mlp.gate_proj.weight", ".mlp.up_proj.weight")):
        return chunk(t, 0)
    if k.endswith(".embed_tokens.weight") or k == "lm_head.weight":
        q = tp * 64
        rows = -(-t.shape[0] // q) * q
        pad = torch.zeros((rows - t.shape[0],) + tuple(t.shape[1:]), dtype=t.dtype)
        return chunk(torch.cat([t, pad]), 0)
    return t  # replicated


def check_tp(label, d, tp, cast=None):
    ranks = [Tensors(d / f"tp_rank_{r:02d}") for r in range(tp)]
    if len(list(d.glob("tp_rank_*"))) != tp:
        fails.append(f"{label}: expected {tp} rank dirs")
    bad = 0
    for k in names:
        t = O.get(k)
        for r in range(tp):
            want = expected_shard(k, t, tp, r)
            if cast is not None:
                want = want.to(cast)
            if not same(ranks[r].get(k), want):
                bad += 1
                if bad < 4:
                    fails.append(f"{label}: {k} rank {r} differs from torch slicing")
    report[label] = f"{len(names)} tensors x {tp} ranks " + ("== torch slicing" if not bad else f"{bad} DIFFER")


check_identical("dcp->safetensors", out / "rt.safetensors")
check_identical("dcp->hf", out / "rt_hf")
check_identical("reshard 500MB", out / "sh500")
check_identical("reshard 3", out / "sh3")
check_identical("--dtype fp32 (vs .float())", out / "fp32.safetensors", torch.float32)
check_identical("--dtype fp16 (vs .to(fp16))", out / "fp16.safetensors", torch.float16)
check_identical("fp32 -> --dtype bf16", out / "bf16.safetensors")
for tp in tps:
    check_tp(f"tp{tp} split", out / f"tp{tp}", tp)
    check_identical(f"tp{tp} merge", out / f"tp{tp}_merged.safetensors")
check_tp(f"tp{tps[0]} split --dtype fp16", out / f"tp{tps[0]}_fp16", tps[0], torch.float16)
if len(tps) > 1:
    a, b = out / f"tp{tps[-1]}to{tps[0]}", out / f"tp{tps[0]}"
    same_files = all(
        (a / f).read_bytes() == (b / f).read_bytes()
        for f in [f"tp_rank_{r:02d}/model.safetensors" for r in range(tps[0])] + ["tp_plan.json"]
    )
    report[f"tp{tps[-1]} -> tp{tps[0]} re-split"] = "byte-identical to a fresh split" if same_files else "DIFFERS"
    if not same_files:
        fails.append("re-split differs from fresh split")

if do_logits:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    def hf_dir(src: Path, name: str) -> Path:
        d = out / f"_logits_{name}"
        shutil.rmtree(d, ignore_errors=True)
        d.mkdir()
        for f in orig.iterdir():
            if f.suffix in (".json", ".txt") and "index" not in f.name:
                shutil.copy(f, d / f.name)
        if src.is_file():
            (d / "model.safetensors").symlink_to(src)
        else:
            for f in src.iterdir():
                if f.name.endswith(".safetensors") or f.name.endswith(".index.json"):
                    (d / f.name).symlink_to(f)
        return d

    torch.manual_seed(0)
    tok = AutoTokenizer.from_pretrained(orig)
    ids = tok("The quick brown fox jumps over the lazy dog. Checkpoints should round-trip exactly.",
              return_tensors="pt").input_ids

    def logits(d):
        m = AutoModelForCausalLM.from_pretrained(d, dtype=torch.float32)
        m.eval()
        with torch.no_grad():
            lg = m(ids).logits
        del m
        gc.collect()
        return lg

    ref = logits(orig)
    for name, src in [("dcp_roundtrip", out / "rt_hf"), ("reshard_500MB", out / "sh500"),
                      (f"tp{tps[-1]}_merged", out / f"tp{tps[-1]}_merged.safetensors")]:
        lg = logits(hf_dir(src, name))
        eq = torch.equal(lg, ref)
        report[f"logits {name}"] = f"identical={eq} max|diff|={(lg - ref).abs().max().item():.3g} shape={tuple(lg.shape)}"
        if not eq:
            fails.append(f"logits differ for {name}")
        shutil.rmtree(out / f"_logits_{name}")

for k, v in report.items():
    print(f"{k:40s} {v}")
print("FAILURES:" if fails else "ALL CHECKS PASSED", *fails, sep="\n  ")
sys.exit(1 if fails else 0)
