"""End-to-end tests: real torch.distributed (gloo) DCP/FSDP checkpoints vs the `ckpt` binary.

Env:
  CKPT_BIN       path to the release binary (default: $CARGO_TARGET_DIR/release/ckpt)
  CKPT_FIXTURES  fixture dir created by scripts/make_fixtures.py (generated if missing;
                 default: $CKPT_DATA_DIR/fixtures, and CKPT_DATA_DIR defaults to <repo>/data)
"""

import hashlib
import json
import os
import pathlib
import struct
import subprocess
import sys

import pytest
import torch
from safetensors.torch import load_file, save_file

ROOT = pathlib.Path(__file__).resolve().parent.parent
BIN = os.environ.get("CKPT_BIN") or os.path.join(
    os.environ.get("CARGO_TARGET_DIR", str(ROOT / "target")), "release", "ckpt"
)
FIX = pathlib.Path(
    os.environ.get("CKPT_FIXTURES")
    or pathlib.Path(os.environ.get("CKPT_DATA_DIR") or ROOT / "data") / "fixtures"
)
RULES = ROOT / "examples" / "gpt2_tp_rules.yaml"


def ckpt(*args, check=True):
    r = subprocess.run([BIN, *map(str, args)], capture_output=True, text=True)
    if check and r.returncode != 0:
        raise AssertionError(f"ckpt {' '.join(map(str, args))} -> {r.returncode}\n{r.stdout}\n{r.stderr}")
    return r


@pytest.fixture(scope="session", autouse=True)
def fixtures():
    if not (FIX / "reference" / "model.safetensors").exists():
        subprocess.run([sys.executable, str(ROOT / "scripts" / "make_fixtures.py"), "--out", str(FIX)], check=True)
    return FIX


def assert_same(a: dict, b: dict):
    assert sorted(a) == sorted(b)
    for k in a:
        assert a[k].dtype == b[k].dtype, k
        assert a[k].shape == b[k].shape, k
        assert torch.equal(a[k], b[k]), k


def raw_tensor_digests(path):
    """sha256 of every tensor's raw bytes across one safetensors file or a directory of shards."""
    p = pathlib.Path(path)
    files = [p] if p.is_file() else sorted(p.glob("*.safetensors"))
    out = {}
    for f in files:
        b = f.read_bytes()
        n = struct.unpack("<Q", b[:8])[0]
        h = json.loads(b[8 : 8 + n])
        for k, v in h.items():
            if k == "__metadata__":
                continue
            s, e = v["data_offsets"]
            out[k] = (v["dtype"], tuple(v["shape"]), hashlib.sha256(b[8 + n + s : 8 + n + e]).hexdigest())
    return out


# ------------------------------------------------------------------ inspect


def test_inspect_detects_formats():
    ref = load_file(FIX / "reference" / "model.safetensors")
    n_params = sum(t.numel() for t in ref.values())
    cases = {
        "dcp_fsdp": "pytorch-dcp",
        "dcp_fsdp_st": "pytorch-dcp",
        "dcp_2d": "pytorch-dcp",
        "hf_sharded": "hf-sharded-safetensors",
        "reference/model.safetensors": "safetensors",
        "hf_sharded/model.safetensors.index.json": "hf-sharded-safetensors",
    }
    for rel, fmt in cases.items():
        j = json.loads(ckpt("inspect", FIX / rel, "--json").stdout)
        assert j["format"] == fmt, rel
    j = json.loads(ckpt("inspect", FIX / "hf_sharded", "--json").stdout)
    assert j["total_params"] == n_params
    assert j["num_tensors"] == len(ref)
    assert len(j["files"]) == len(list((FIX / "hf_sharded").glob("*.safetensors")))
    t = {x["name"]: x for x in j["tensors"]}
    assert t["transformer.wte.weight"]["shape"] == list(ref["transformer.wte.weight"].shape)


def test_inspect_dcp_details():
    j = json.loads(ckpt("inspect", FIX / "dcp_fsdp", "--json").stdout)
    t = {x["name"]: x for x in j["tensors"]}
    wte = t["model.transformer.wte.weight"]
    assert wte["grid"] == [4, 1]  # FSDP: 4 row shards
    assert sum(c["sizes"][0] for c in wte["chunks"]) == wte["shape"][0]
    assert j["non_tensor_items"]["optim.param_groups.0.lr"] == pytest.approx(0.01)
    assert j["non_tensor_items"]["optim.param_groups.0.betas"] == [0.9, 0.999]
    j2 = json.loads(ckpt("inspect", FIX / "dcp_2d", "--json").stdout)
    t2 = {x["name"]: x for x in j2["tensors"]}
    assert t2["transformer.wte.weight"]["grid"] == [2, 2]
    assert t2["transformer.wte.weight"]["dtype"] == "BF16"


# ------------------------------------------------------------------ DCP -> safetensors


@pytest.mark.parametrize(
    "src,ref,extra",
    [
        ("dcp_fsdp", "reference/model.safetensors", ["--include", "model.*", "--strip-prefix", "model."]),
        ("dcp_fsdp_st", "reference/model.safetensors", ["--strip-prefix", "model."]),
        ("dcp_2d", "reference_2d/model.safetensors", []),
        ("dcp_fsdp", "reference_optim/optim.safetensors", ["--include", "optim.*"]),
    ],
)
def test_dcp_to_single_safetensors_exact(tmp_path, src, ref, extra):
    out = tmp_path / "out.safetensors"
    ckpt("convert", FIX / src, "-o", out, *extra)
    assert_same(load_file(out), load_file(FIX / ref))
    r = ckpt("diff", out, FIX / ref)
    assert "RESULT: EQUAL" in r.stdout


def test_dcp_to_hf_sharded_loads_in_transformers(tmp_path):
    from transformers import GPT2LMHeadModel

    out = tmp_path / "hf"
    ckpt("convert", FIX / "dcp_fsdp", "-o", out, "--include", "model.*", "--strip-prefix", "model.", "--max-shard-size", "300KB")
    idx = json.loads((out / "model.safetensors.index.json").read_text())
    shards = sorted(out.glob("model-*.safetensors"))
    assert len(shards) > 1
    assert set(idx["weight_map"].values()) == {s.name for s in shards}
    assert idx["metadata"]["total_size"] == sum(t.numel() * t.element_size() for t in load_file(FIX / "reference/model.safetensors").values())
    for f in ("config.json", "generation_config.json"):
        (out / f).write_bytes((FIX / "hf_sharded" / f).read_bytes())
    ids = torch.arange(16).reshape(2, 8)
    a = GPT2LMHeadModel.from_pretrained(out).eval()
    b = GPT2LMHeadModel.from_pretrained(FIX / "hf_sharded").eval()
    with torch.no_grad():
        assert torch.equal(a(ids).logits, b(ids).logits)


# ------------------------------------------------------------------ HF sharded <-> single, reshard


def test_hf_sharded_single_roundtrip_preserves_bytes(tmp_path):
    single = tmp_path / "single.safetensors"
    ckpt("convert", FIX / "hf_sharded", "-o", single)
    assert raw_tensor_digests(single) == raw_tensor_digests(FIX / "hf_sharded")
    back = tmp_path / "back"
    ckpt("convert", single, "-o", back, "--max-shard-size", "600KB")
    assert raw_tensor_digests(back) == raw_tensor_digests(FIX / "hf_sharded")
    assert (back / "model.safetensors.index.json").exists()
    assert_same(load_file(single), load_file(FIX / "reference/model.safetensors"))


@pytest.mark.parametrize("n", [1, 2, 3, 5])
def test_reshard_num_shards(tmp_path, n):
    out = tmp_path / f"r{n}"
    ckpt("reshard", FIX / "hf_sharded", "-o", out, "--num-shards", n)
    shards = sorted(out.glob("*.safetensors"))
    assert len(shards) == n
    assert raw_tensor_digests(out) == raw_tensor_digests(FIX / "hf_sharded")
    assert (out / "config.json").exists()  # aux files copied
    assert ckpt("diff", out, FIX / "hf_sharded").returncode == 0


# ------------------------------------------------------------------ safetensors -> DCP (torch loads it)


def test_safetensors_to_dcp_loads_in_torch(tmp_path):
    out = tmp_path / "dcp"
    ckpt("convert", FIX / "reference/model.safetensors", "--to", "dcp", "-o", out, "--dcp-ranks", 3)
    assert (out / ".metadata").exists() and (out / "__2_0.distcp").exists()
    # our own reader
    assert "RESULT: EQUAL" in ckpt("diff", out, FIX / "reference/model.safetensors").stdout
    # torch: dcp_to_torch_save + FSDP2 dcp.load on 2 gloo ranks (resharding 3 -> 2)
    r = subprocess.run(
        [sys.executable, ROOT / "scripts" / "verify_dcp_load.py", out, FIX / "reference/model.safetensors", FIX / "hf_sharded", "--nproc", "2"],
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0, r.stderr[-3000:]


def test_dcp_to_dcp_rechunk(tmp_path):
    out = tmp_path / "dcp1"
    ckpt("convert", FIX / "dcp_2d", "--to", "dcp", "-o", out)
    assert "RESULT: EQUAL" in ckpt("diff", out, FIX / "dcp_2d").stdout


# ------------------------------------------------------------------ tensor-parallel split / merge


def expected_tp_shard(name, t, tp, r):
    """Reference TP slicing with torch, mirroring examples/gpt2_tp_rules.yaml."""
    def split(x, dim, parts=1):
        blocks = x.chunk(parts, dim=dim)
        return torch.cat([b.chunk(tp, dim=dim)[r] for b in blocks], dim=dim)

    if name.endswith("attn.c_attn.weight"):
        return split(t, 1, 3)
    if name.endswith("attn.c_attn.bias"):
        return split(t, 0, 3)
    if name.endswith("attn.c_proj.weight") or name.endswith("mlp.c_proj.weight"):
        return split(t, 0)
    if name.endswith("mlp.c_fc.weight"):
        return split(t, 1)
    if name.endswith("mlp.c_fc.bias"):
        return split(t, 0)
    return t


def test_tp_split_merge_resplit(tmp_path):
    ref = load_file(FIX / "reference/model.safetensors")
    tp4 = tmp_path / "tp4"
    ckpt("reshard", FIX / "hf_sharded", "--tp", 4, "--rules", RULES, "-o", tp4)
    for r in range(4):
        got = load_file(tp4 / f"tp_rank_{r:02d}" / "model.safetensors")
        exp = {k: expected_tp_shard(k, v, 4, r).contiguous() for k, v in ref.items()}
        assert_same(got, exp)
    merged = tmp_path / "merged.safetensors"
    ckpt("reshard", tp4, "-o", merged)
    assert_same(load_file(merged), ref)
    # tp4 -> tp2 directly (merge + split per tensor) equals a fresh tp2 split
    tp2a, tp2b = tmp_path / "tp2a", tmp_path / "tp2b"
    ckpt("reshard", tp4, "--tp", 2, "-o", tp2a)
    ckpt("reshard", FIX / "dcp_fsdp", "--include", "model.*", "--tp", 2, "--rules", RULES, "-o", tmp_path / "tmp_prefixed")
    ckpt("reshard", FIX / "hf_sharded", "--tp", 2, "--rules", RULES, "-o", tp2b)
    for r in range(2):
        assert raw_tensor_digests(tp2a / f"tp_rank_{r:02d}" / "model.safetensors") == raw_tensor_digests(
            tp2b / f"tp_rank_{r:02d}" / "model.safetensors"
        )


def test_tp_indivisible_dim_is_an_error(tmp_path):
    rules = tmp_path / "r.yaml"
    rules.write_text('rules:\n  - {pattern: "*wte.weight", dim: 0}\n')  # vocab 1001 is odd
    r = ckpt("reshard", FIX / "hf_sharded", "--tp", 2, "--rules", rules, "-o", tmp_path / "x", check=False)
    assert r.returncode == 2 and "not divisible" in r.stderr


# ------------------------------------------------------------------ diff semantics


def test_diff_reports_changes(tmp_path):
    ref = load_file(FIX / "reference/model.safetensors")
    mod = dict(ref)
    w = mod["transformer.h.0.mlp.c_fc.weight"].clone()
    w[0, 0] += 1e-3
    mod["transformer.h.0.mlp.c_fc.weight"] = w
    del mod["transformer.ln_f.bias"]
    mod["extra.weight"] = torch.zeros(3)
    mod["transformer.wpe.weight"] = mod["transformer.wpe.weight"].to(torch.bfloat16)
    p = tmp_path / "mod.safetensors"
    save_file(mod, p)
    r = ckpt("diff", FIX / "reference/model.safetensors", p, "--json", check=False)
    assert r.returncode == 1
    j = json.loads(r.stdout)
    assert j["only_in_a"] == ["transformer.ln_f.bias"]
    assert j["only_in_b"] == ["extra.weight"]
    t = {x["name"]: x for x in j["tensors"]}
    c = t["transformer.h.0.mlp.c_fc.weight"]
    assert c["status"] == "different"
    assert c["max_abs"] == pytest.approx(1e-3, rel=1e-3)
    assert c["changed_frac"] == pytest.approx(1 / w.numel())
    assert c["cosine"] > 0.999999
    assert t["transformer.wpe.weight"]["status"] == "dtype-mismatch"
    # tolerance + ignore-missing on the remaining numeric change
    r = ckpt("diff", FIX / "reference/model.safetensors", p, "--tolerance", "1e-2", "--ignore-missing",
             "--include", "transformer.h.*", check=False)
    assert r.returncode == 0, r.stdout


def test_diff_dcp_vs_dcp_same_weights_different_layout():
    # FSDP (4 row shards, fp32) vs HF sharded: same values, different layouts
    r = ckpt("diff", FIX / "dcp_fsdp", FIX / "hf_sharded", "--strip-prefix-a", "model.", "--include", "transformer.*", "--include", "lm_head.*")
    assert "RESULT: EQUAL" in r.stdout


# ------------------------------------------------------------------ safety


def test_malicious_metadata_is_refused(tmp_path):
    marker = tmp_path / "pwned"

    class Evil:
        def __reduce__(self):
            return (os.system, (f"touch {marker}",))

    import pickle

    d = tmp_path / "evil_dcp"
    d.mkdir()
    (d / ".metadata").write_bytes(pickle.dumps(Evil()))
    r = ckpt("inspect", d, check=False)
    assert r.returncode == 2
    assert "refusing pickle global" in r.stderr
    assert not marker.exists()
    # same payload hidden inside a tensor chunk of an otherwise valid checkpoint
    good = FIX / "dcp_2d"
    bad = tmp_path / "evil_chunk"
    bad.mkdir()
    for f in good.iterdir():
        (bad / f.name).write_bytes(f.read_bytes())
    blob = bytearray((bad / "__0_0.distcp").read_bytes())
    payload = b"cposix\nsystem\n"
    i = blob.find(b"ctorch._utils\n_rebuild_tensor_v2\n")
    assert i >= 0
    blob[i : i + len(payload)] = payload  # rewrite the first global of the first chunk pickle
    (bad / "__0_0.distcp").write_bytes(bytes(blob))
    r = ckpt("convert", bad, "-o", tmp_path / "x.safetensors", check=False)
    assert r.returncode == 2 and "refusing pickle global" in r.stderr, r.stderr
