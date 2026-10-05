"""End-to-end tests for --dtype, GQA/vocab-padded TP, Megatron-LM, DeepSpeed ZeRO, zip64 and --verify.

Fixtures (generated if missing, see scripts/):
  $CKPT_FIXTURES/megatron/   real megatron-core TP2/PP2 checkpoints (legacy + torch_dist) + references
  $CKPT_FIXTURES/deepspeed/  real DeepSpeed ZeRO-1/2/3 checkpoints + zero_to_fp32 references
Python never unpickles a checkpoint here except to produce references with the frameworks' own tools
(at fixture time) and to load our *outputs* (safetensors, and DCP written by ckpt).
"""

import json
import os
import pathlib
import shutil
import struct
import subprocess
import sys
import zipfile

import pytest
import torch
from safetensors.torch import load_file, save_file

from test_roundtrip import BIN, FIX, ROOT, assert_same, ckpt, raw_tensor_digests

MFIX = FIX / "megatron"
DSFIX = FIX / "deepspeed"
LLAMA_RULES = ROOT / "examples" / "llama_tp_rules.yaml"


@pytest.fixture(scope="session")
def megatron():
    if not (MFIX / "reference" / "logits.safetensors").exists():
        subprocess.run([sys.executable, str(ROOT / "scripts" / "make_megatron_fixtures.py"), "--out", str(MFIX)], check=True)
    return MFIX


@pytest.fixture(scope="session")
def deepspeed():
    if not (DSFIX / "zero3" / "reference.safetensors").exists():
        subprocess.run([sys.executable, str(ROOT / "scripts" / "make_deepspeed_fixtures.py"), "--out", str(DSFIX)], check=True)
    return DSFIX


def bits(t):
    """Integer view of a tensor's raw bits (for bit-exact comparisons incl. -0.0)."""
    return t.contiguous().view({1: torch.uint8, 2: torch.int16, 4: torch.int32, 8: torch.int64}[t.element_size()])


def assert_bit_exact(got, exp, what):
    assert got.dtype == exp.dtype and got.shape == exp.shape, what
    gn, en = torch.isnan(got.float()), torch.isnan(exp.float())
    assert torch.equal(gn, en), f"{what}: NaN positions differ"
    g, e = bits(got)[~gn], bits(exp)[~en]
    bad = (g != e).nonzero()
    assert bad.numel() == 0, f"{what}: {bad.numel()} mismatches, first at {bad[0].item()}: got {g[bad[0]].item():#x} exp {e[bad[0]].item():#x}"


# ------------------------------------------------------------------ --dtype


def dtype_source(path):
    torch.manual_seed(0)
    f32_specials = torch.tensor([0.0, -0.0, float("inf"), float("-inf"), float("nan"), 1e-45, -1e-45, 1.17e-38,
                                 3.4028235e38, -3.4028235e38, 65504.0, 65519.99, 65520.0, 448.0, 464.0, 464.01,
                                 470.0, 479.9, 480.0, 57344.0, 61439.0, 61440.0, 65536.0, 2.0 ** -9, 2.0 ** -10,
                                 2.0 ** -16, 2.0 ** -17, 2.0 ** -24, 2.0 ** -25, 2.0 ** -25 * 1.0001, 1 + 2 ** -8,
                                 1 + 3 * 2 ** -8, 1 + 2 ** -11, 1 + 3 * 2 ** -11, 1 + 2 ** -4, 1 + 3 * 2 ** -4])
    # every bf16 / fp16 value, their midpoints and +-1 ulp around the midpoints (as fp32)
    all16 = torch.arange(-32768, 32768, dtype=torch.int32).to(torch.int16)
    bf = all16.view(torch.bfloat16).float()
    hf = all16.view(torch.float16).float()
    f8a = torch.arange(256, dtype=torch.int32).to(torch.uint8)
    e4 = f8a.view(torch.float8_e4m3fn).float()
    e5 = f8a.view(torch.float8_e5m2).float()
    grid = torch.cat([bf, hf, e4, e5])
    grid = grid[torch.isfinite(grid)].unique()
    mid = (grid[1:].double() + grid[:-1].double()) / 2
    mids = torch.cat([mid.float(), (mid.float().view(torch.int32) + 1).view(torch.float32),
                      (mid.float().view(torch.int32) - 1).view(torch.float32)])
    rnd = torch.randint(-(2 ** 31), 2 ** 31 - 1, (1 << 20,), dtype=torch.int64).to(torch.int32).view(torch.float32)
    f32 = torch.cat([f32_specials, grid, mids, rnd, torch.randn(4096) * 300])
    f64 = torch.cat([torch.randn(8192, dtype=torch.float64) * 1000, mid, mid + 1e-12, mid - 1e-12])
    save_file({
        "a.f32": f32.reshape(-1, 1),
        "b.f64": f64,
        "c.bf16": all16.view(torch.bfloat16).clone(),
        "d.f16": all16.view(torch.float16).clone(),
        "e.e4m3": f8a.view(torch.float8_e4m3fn).clone(),
        "f.e5m2": f8a.view(torch.float8_e5m2).clone(),
        "g.i64": torch.arange(-5, 5),
        "h.norm.weight": torch.randn(7),
    }, path)


DTYPES = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16, "fp8_e4m3": torch.float8_e4m3fn,
          "fp8_e5m2": torch.float8_e5m2, "fp64": torch.float64}


@pytest.mark.parametrize("name", list(DTYPES))
def test_dtype_cast_bit_exact_vs_torch(tmp_path, name):
    src = tmp_path / "src.safetensors"
    dtype_source(src)
    out = tmp_path / "out.safetensors"
    ckpt("convert", src, "-o", out, "--dtype", name)
    ref, got = load_file(src), load_file(out)
    for k, t in ref.items():
        if not t.dtype.is_floating_point:
            assert torch.equal(got[k], t), k  # ints untouched
            continue
        assert_bit_exact(got[k], t.to(DTYPES[name]), f"{k} -> {name}")


def test_dtype_keep_and_other_outputs(tmp_path):
    src = FIX / "reference" / "model.safetensors"
    ref = load_file(src)
    out = tmp_path / "bf16.safetensors"
    ckpt("convert", src, "-o", out, "--dtype", "bf16", "--keep-dtype", "*ln_*")
    got = load_file(out)
    for k, t in ref.items():
        exp = t if ".ln_" in k else t.to(torch.bfloat16)
        assert_bit_exact(got[k], exp, k)
    # DCP output (torch loads it), HF sharded output, TP split + merge all honour --dtype
    dcp_out = tmp_path / "dcp"
    ckpt("convert", src, "--to", "dcp", "--dcp-ranks", 2, "-o", dcp_out, "--dtype", "fp16")
    from torch.distributed.checkpoint.format_utils import dcp_to_torch_save
    dcp_to_torch_save(str(dcp_out), str(tmp_path / "full.pt"))
    full = torch.load(tmp_path / "full.pt", weights_only=True)  # our own output, plain tensors only
    for k, t in ref.items():
        assert_bit_exact(full[k], t.to(torch.float16), k)
    tp = tmp_path / "tp2"
    ckpt("reshard", FIX / "hf_sharded", "--tp", 2, "--rules", ROOT / "examples" / "gpt2_tp_rules.yaml", "-o", tp, "--dtype", "bf16")
    merged = tmp_path / "m.safetensors"
    ckpt("reshard", tp, "-o", merged)
    got = load_file(merged)
    for k, t in ref.items():
        assert_bit_exact(got[k], t.to(torch.bfloat16), k)
    # dcp_fsdp (FSDP2, torch-written) -> fp8 single file
    out8 = tmp_path / "fp8.safetensors"
    ckpt("convert", FIX / "dcp_fsdp", "-o", out8, "--include", "model.*", "--strip-prefix", "model.", "--dtype", "fp8_e4m3")
    got = load_file(out8)
    for k, t in ref.items():
        assert_bit_exact(got[k], t.to(torch.float8_e4m3fn), k)


# ------------------------------------------------------------------ GQA / padded-vocab tensor parallel


def tiny_qwen2(path, vocab=250, kv=2):
    from transformers import Qwen2Config, Qwen2ForCausalLM

    cfg = Qwen2Config(vocab_size=vocab, hidden_size=64, intermediate_size=96, num_hidden_layers=2,
                      num_attention_heads=8, num_key_value_heads=kv, head_dim=8, max_position_embeddings=64,
                      tie_word_embeddings=False)
    torch.manual_seed(3)
    m = Qwen2ForCausalLM(cfg)
    with torch.no_grad():  # random biases (default init is zero) so bias slicing is checked too
        for n, p in m.named_parameters():
            if n.endswith("bias"):
                p.normal_()
    m.save_pretrained(path)
    return m, cfg


def expected_shard(name, t, tp, r, nh, nkv, pad):
    """Manual torch slicing for examples/llama_tp_rules.yaml."""
    hd = None

    def heads_split(x, n, dim, kv):
        if n % tp == 0:
            return x.chunk(tp, dim=dim)[r]
        assert kv and tp % n == 0
        return x.chunk(n, dim=dim)[r // (tp // n)]

    if ".q_proj." in name or ".o_proj.weight" in name:
        return heads_split(t, nh, 0 if ".q_proj." in name else 1, False)
    if ".k_proj." in name or ".v_proj." in name:
        return heads_split(t, nkv, 0, True)
    if ".qkv_proj." in name:
        hd = t.shape[0] // (nh + 2 * nkv)
        q, k, v = t.split([nh * hd, nkv * hd, nkv * hd])
        return torch.cat([heads_split(q, nh, 0, False), heads_split(k, nkv, 0, True), heads_split(v, nkv, 0, True)])
    if ".gate_proj." in name or ".up_proj." in name:
        return t.chunk(tp, 0)[r]
    if ".down_proj." in name:
        return t.chunk(tp, 1)[r]
    if name.endswith("embed_tokens.weight") or name == "lm_head.weight":
        q = tp * pad
        padded = torch.cat([t, torch.zeros((-t.shape[0]) % q, t.shape[1], dtype=t.dtype)])
        return padded.chunk(tp, 0)[r]
    return t


def test_tp_gqa_qwen2_split_merge(tmp_path):
    src = tmp_path / "qwen2"
    model, cfg = tiny_qwen2(src)
    ref = load_file(src / "model.safetensors")
    # add a Phi-3 style fused qkv_proj copy of layer 0 to exercise unequal fused sections
    fused = dict(ref)
    p = "model.layers.0.self_attn."
    fused[p + "qkv_proj.weight"] = torch.cat([ref[p + "q_proj.weight"], ref[p + "k_proj.weight"], ref[p + "v_proj.weight"]])
    save_file(fused, src / "model.safetensors", metadata={"format": "pt"})
    ref = fused
    for tp in (1, 2, 4, 8):  # kv heads = 2: tp 4 and 8 replicate kv heads, tp 8 = 1 q head per rank
        out = tmp_path / f"tp{tp}"
        ckpt("reshard", src, "--tp", tp, "--rules", LLAMA_RULES, "-o", out)
        for r in range(tp):
            got = load_file(out / f"tp_rank_{r:02d}" / "model.safetensors")
            for k, t in ref.items():
                exp = expected_shard(k, t, tp, r, 8, 2, 64).contiguous()
                assert torch.equal(got[k], exp), f"tp{tp} rank{r} {k}"
        merged = tmp_path / f"m{tp}.safetensors"
        ckpt("reshard", out, "-o", merged)
        assert_same(load_file(merged), ref)
    # re-split tp8 -> tp2 without rules == direct tp2 split
    ckpt("reshard", tmp_path / "tp8", "--tp", 2, "-o", tmp_path / "tp2b")
    for r in range(2):
        assert raw_tensor_digests(tmp_path / "tp2" / f"tp_rank_{r:02d}" / "model.safetensors") == raw_tensor_digests(
            tmp_path / "tp2b" / f"tp_rank_{r:02d}" / "model.safetensors")
    plan = json.loads((tmp_path / "tp4" / "tp_plan.json").read_text())
    assert plan["tensors"]["model.embed_tokens.weight"]["orig_len"] == 250
    assert load_file(tmp_path / "tp4" / "tp_rank_03" / "model.safetensors")["model.embed_tokens.weight"].shape[0] == 64


def test_tp_gqa_attention_semantics(tmp_path):
    """Per-rank attention on the TP shards, summed over ranks (row-parallel o_proj), equals full attention."""
    src = tmp_path / "qwen2"
    model, cfg = tiny_qwen2(src)
    from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb

    x = torch.randn(1, 6, 64)
    pos = torch.arange(6)[None]
    cos, sin = model.model.rotary_emb(x, pos)

    def attn(sd, pre, nh, nkv):
        q = (x @ sd[pre + "q_proj.weight"].T + sd[pre + "q_proj.bias"]).view(1, 6, nh, 8).transpose(1, 2)
        k = (x @ sd[pre + "k_proj.weight"].T + sd[pre + "k_proj.bias"]).view(1, 6, nkv, 8).transpose(1, 2)
        v = (x @ sd[pre + "v_proj.weight"].T + sd[pre + "v_proj.bias"]).view(1, 6, nkv, 8).transpose(1, 2)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        k = k.repeat_interleave(nh // nkv, dim=1)  # HF repeat_kv: kv head j serves q heads j*g..(j+1)*g-1
        v = v.repeat_interleave(nh // nkv, dim=1)
        o = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
        return o.transpose(1, 2).reshape(1, 6, nh * 8) @ sd[pre + "o_proj.weight"].T

    ref = load_file(src / "model.safetensors")
    pre = "model.layers.1.self_attn."
    full = attn(ref, pre, 8, 2)
    for tp in (2, 4, 8):
        out = tmp_path / f"tp{tp}"
        ckpt("reshard", src, "--tp", tp, "--rules", LLAMA_RULES, "-o", out)
        total = sum(attn(load_file(out / f"tp_rank_{r:02d}" / "model.safetensors"), pre, 8 // tp, max(1, 2 // tp))
                    for r in range(tp))
        torch.testing.assert_close(total, full, rtol=1e-5, atol=1e-5)


def test_tp_gqa_errors(tmp_path):
    src = tmp_path / "qwen2"
    tiny_qwen2(src, kv=2)
    r = ckpt("reshard", src, "--tp", 3, "--rules", LLAMA_RULES, "-o", tmp_path / "x", check=False)
    assert r.returncode == 2 and "heads" in r.stderr, r.stderr
    (src / "config.json").rename(src / "cfg.json")
    r = ckpt("reshard", src, "--tp", 2, "--rules", LLAMA_RULES, "-o", tmp_path / "y", check=False)
    assert r.returncode == 2 and "is not defined" in r.stderr and "--model-config" in r.stderr, r.stderr
    ckpt("reshard", src, "--tp", 2, "--rules", LLAMA_RULES, "--model-config", src / "cfg.json", "-o", tmp_path / "z")


# ------------------------------------------------------------------ Megatron-LM


def test_megatron_legacy_inspect_and_merge(megatron):
    j = json.loads(ckpt("inspect", megatron / "legacy", "--json").stdout)
    assert j["format"] == "megatron-legacy"
    assert j["info"]["tensor_parallel"] == 2 and j["info"]["pipeline_parallel"] == 2
    assert j["info"]["iteration"] == 10
    t = {x["name"]: x for x in j["tensors"]}
    assert t["decoder.layers.3.self_attention.linear_qkv.weight"]["shape"] == [128, 64]  # PP stage 1 layer 1 -> global 3
    assert t["embedding.word_embeddings.weight"]["shape"] == [256, 64]  # padded vocab
    # TP2/PP2 merge is bit-exact vs megatron-core's own TP=1 gather of the same weights
    r = ckpt("diff", megatron / "legacy", megatron / "reference" / "model.safetensors")
    assert "RESULT: EQUAL" in r.stdout
    # a single rank file opens as a plain torch.save dict
    j = json.loads(ckpt("inspect", megatron / "legacy" / "iter_0000010" / "mp_rank_01_001" / "model_optim_rng.pt", "--json").stdout)
    assert j["format"] == "torch-save"
    assert any(x["name"] == "model.output_layer.weight" for x in j["tensors"])
    assert j["scalars"]["iteration"] == 10


@pytest.mark.parametrize("src", ["legacy", "torch_dist"])
def test_megatron_to_hf_logits(tmp_path, megatron, src):
    from transformers import AutoModelForCausalLM

    out = tmp_path / "hf"
    ckpt("convert", megatron / src, "-o", out)
    cfg = json.loads((out / "config.json").read_text())
    assert cfg["architectures"] == ["Qwen2ForCausalLM"] and cfg["vocab_size"] == 250 and cfg["num_key_value_heads"] == 4
    m = AutoModelForCausalLM.from_pretrained(out, dtype=torch.float32).eval()
    ref = load_file(megatron / "reference" / "logits.safetensors")
    with torch.no_grad():
        logits = m(ref["input_ids"]).logits
    # vs the real TP2/PP2 Megatron forward pass
    torch.testing.assert_close(logits, ref["logits"], rtol=1e-5, atol=1e-5)
    # explicit de-interleave check against the TP=1 reference
    mref = load_file(megatron / "reference" / "model.safetensors")
    hf = load_file(out / "model.safetensors")
    qkv = mref["decoder.layers.2.self_attention.linear_qkv.weight"].view(4, 2 + 2, 8, 64)
    assert torch.equal(hf["model.layers.2.self_attn.q_proj.weight"], qkv[:, :2].reshape(-1, 64))
    assert torch.equal(hf["model.layers.2.self_attn.k_proj.weight"], qkv[:, 2].reshape(-1, 64))
    assert torch.equal(hf["model.layers.2.self_attn.v_proj.weight"], qkv[:, 3].reshape(-1, 64))
    fc1 = mref["decoder.layers.2.mlp.linear_fc1.weight"]
    assert torch.equal(hf["model.layers.2.mlp.gate_proj.weight"], fc1[:96])
    assert torch.equal(hf["model.layers.2.mlp.up_proj.weight"], fc1[96:])
    assert torch.equal(hf["lm_head.weight"], mref["output_layer.weight"][:250])


def test_megatron_torch_dist_and_options(tmp_path, megatron):
    j = json.loads(ckpt("inspect", megatron / "torch_dist", "--json").stdout)
    assert j["format"] == "megatron-torch-dist" and j["info"]["args"]["num_layers"] == 4
    raw = json.loads(ckpt("inspect", megatron / "torch_dist", "--raw", "--json").stdout)
    t = {x["name"]: x for x in raw["tensors"]}
    assert t["decoder.layers.self_attention.linear_qkv.weight"]["shape"] == [4, 128, 64]  # stacked layers
    a, b = tmp_path / "a", tmp_path / "b"
    ckpt("convert", megatron / "legacy", "-o", a)
    ckpt("convert", megatron / "torch_dist", "-o", b)
    assert raw_tensor_digests(a) == raw_tensor_digests(b)
    # --keep-names: merged Megatron names; --vocab-size unpads
    k = tmp_path / "k.safetensors"
    ckpt("convert", megatron / "legacy", "-o", k, "--keep-names", "--vocab-size", 250)
    sd = load_file(k)
    mref = load_file(megatron / "reference" / "model.safetensors")
    assert torch.equal(sd["embedding.word_embeddings.weight"], mref["embedding.word_embeddings.weight"][:250])
    assert torch.equal(sd["decoder.layers.1.mlp.linear_fc1.weight"], mref["decoder.layers.1.mlp.linear_fc1.weight"])
    # bf16 HF export
    h = tmp_path / "h"
    ckpt("convert", megatron / "legacy", "-o", h, "--dtype", "bf16")
    assert json.loads((h / "config.json").read_text())["torch_dtype"] == "bfloat16"
    assert load_file(h / "model.safetensors")["model.norm.weight"].dtype == torch.bfloat16


def test_megatron_malicious_rank_file_refused(tmp_path, megatron):
    marker = tmp_path / "pwned"

    class Evil:
        def __reduce__(self):
            return (os.system, (f"touch {marker}",))

    d = tmp_path / "evil" / "iter_0000001" / "mp_rank_00"
    d.mkdir(parents=True)
    torch.save({"args": None, "iteration": 1, "model": {"w": torch.zeros(2)}, "rng_state": [Evil()]}, d / "model_optim_rng.pt")
    (tmp_path / "evil" / "latest_checkpointed_iteration.txt").write_text("1")
    r = ckpt("inspect", tmp_path / "evil", check=False)
    assert r.returncode == 2 and "refusing pickle global `posix.system`" in r.stderr, r.stderr
    assert not marker.exists()


# ------------------------------------------------------------------ DeepSpeed ZeRO


@pytest.mark.parametrize("stage", [1, 2, 3])
def test_deepspeed_zero_to_fp32(tmp_path, deepspeed, stage):
    d = deepspeed / f"zero{stage}"
    j = json.loads(ckpt("inspect", d, "--json").stdout)
    assert j["format"] == "deepspeed-zero" and j["info"]["zero_stage"] == stage and j["info"]["world_size"] == 3
    out = tmp_path / "fp32.safetensors"
    ckpt("convert", d, "-o", out)
    # bit-exact vs DeepSpeed's own zero_to_fp32 (incl. frozen params, buffers, tied weights)
    assert_same(load_file(out), load_file(d / "reference.safetensors"))
    assert "RESULT: EQUAL" in ckpt("diff", d / "global_step1", d / "reference.safetensors").stdout


def test_deepspeed_raw_and_errors(tmp_path, deepspeed):
    f = deepspeed / "zero2" / "global_step1" / "mp_rank_00_model_states.pt"
    j = json.loads(ckpt("inspect", f, "--raw", "--json").stdout)
    assert j["format"] == "torch-save"
    assert any(x["name"].startswith("module.") for x in j["tensors"])  # bf16 module weights
    bad = tmp_path / "ds"
    shutil.copytree(deepspeed / "zero2" / "global_step1", bad)
    (bad / "bf16_zero_pp_rank_2_mp_rank_00_optim_states.pt").unlink()
    r = ckpt("inspect", bad, check=False)
    assert r.returncode == 2 and "expected 3" in r.stderr, r.stderr
    # malicious object in an optimizer-state file is refused
    marker = tmp_path / "pwned"

    class Evil:
        def __reduce__(self):
            return (os.system, (f"touch {marker}",))

    evil = tmp_path / "evil"
    shutil.copytree(deepspeed / "zero2" / "global_step1", evil)
    torch.save({"optimizer_state_dict": {"zero_stage": 2, "x": Evil()}}, evil / "bf16_zero_pp_rank_0_mp_rank_00_optim_states.pt")
    r = ckpt("inspect", evil, check=False)
    assert r.returncode == 2 and "refusing pickle global" in r.stderr and not marker.exists()


# ------------------------------------------------------------------ zip64 + CRC verification


def test_forced_zip64_dcp_loads_in_torch(tmp_path):
    out = tmp_path / "dcp"
    env = dict(os.environ, CKPT_FORCE_ZIP64="1")
    r = subprocess.run([BIN, "convert", FIX / "reference" / "model.safetensors", "--to", "dcp", "--dcp-ranks", "2", "-o", out],
                       env=env, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    from torch.distributed.checkpoint.format_utils import dcp_to_torch_save
    dcp_to_torch_save(str(out), str(tmp_path / "full.pt"))
    assert_same(torch.load(tmp_path / "full.pt", weights_only=True), load_file(FIX / "reference" / "model.safetensors"))
    # every chunk archive is a valid zip64 archive for python's zipfile (CRC checked by testzip)
    j = json.loads(ckpt("inspect", out, "--json", "--chunks").stdout)
    blob = (out / "__0_0.distcp").read_bytes()
    c = [c for t in j["tensors"] for c in t["chunks"] if c["file"] == "__0_0.distcp"][0]
    import io
    z = zipfile.ZipFile(io.BytesIO(blob[c["offset"]: c["offset"] + c["length"]]))
    assert z.testzip() is None
    assert all(i.extra[:2] == b"\x01\x00" for i in z.infolist())  # zip64 extra field present
    assert "RESULT: EQUAL" in ckpt("--verify", "diff", out, FIX / "reference" / "model.safetensors").stdout


def test_verify_detects_corruption(tmp_path, megatron):
    bad = tmp_path / "dcp"
    shutil.copytree(FIX / "dcp_fsdp", bad)
    j = json.loads(ckpt("inspect", bad, "--json", "--chunks").stdout)
    t = {x["name"]: x for x in j["tensors"]}["model.transformer.wte.weight"]
    c = t["chunks"][0]
    blob = bytearray((bad / c["file"]).read_bytes())
    # flip one byte in the middle of the tensor data of the first chunk (data is the largest entry)
    pos = c["offset"] + c["length"] // 2
    blob[pos] ^= 0x40
    (bad / c["file"]).write_bytes(bytes(blob))
    ok = ckpt("convert", bad, "-o", tmp_path / "x.safetensors", "--include", "model.*")  # no --verify: silently wrong
    assert ok.returncode == 0
    r = ckpt("--verify", "convert", bad, "-o", tmp_path / "y.safetensors", "--include", "model.*", check=False)
    assert r.returncode == 2 and "CRC32 mismatch" in r.stderr, r.stderr
    r = ckpt("inspect", "--verify", bad, check=False)
    assert r.returncode == 2 and "CRC32 mismatch" in r.stderr
    good = ckpt("inspect", "--verify", "--summary", FIX / "dcp_fsdp")
    assert "CRC32 OK" in good.stdout
    # torch.save based formats too
    assert "CRC32 OK" in ckpt("inspect", "--verify", "--summary", megatron / "legacy").stdout
    leg = tmp_path / "leg"
    shutil.copytree(megatron / "legacy", leg)
    f = leg / "iter_0000010" / "mp_rank_00_000" / "model_optim_rng.pt"
    b = bytearray(f.read_bytes())
    zf = zipfile.ZipFile(f)
    info = [i for i in zf.infolist() if i.filename.endswith("/data/0")][0]
    hdr = info.header_offset
    nlen, xlen = struct.unpack("<HH", b[hdr + 26: hdr + 30])
    b[hdr + 30 + nlen + xlen + 3] ^= 1
    f.write_bytes(bytes(b))
    r = ckpt("--verify", "convert", leg, "-o", tmp_path / "z.safetensors", "--keep-names", check=False)
    assert r.returncode == 2 and "CRC32 mismatch" in r.stderr, r.stderr


@pytest.mark.skipif(not os.environ.get("CKPT_BIG_TESTS"), reason="set CKPT_BIG_TESTS=1 (writes ~8.6 GB to $CKPT_BIG_DIR)")
def test_zip64_real_4gib_chunk():
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "test_zip64_big.py")], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
