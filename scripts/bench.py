#!/usr/bin/env python3
"""Benchmark `ckpt` against the torch-native path on a ~480 MB fp32 GPT-2-style DCP checkpoint.

Creates (once) <dir>/dcp: FSDP2 checkpoint written by 4 gloo ranks (10 layers, d=768, vocab 32000,
untied head = ~121M params), then times with /usr/bin/time (wall seconds, peak RSS):

  torch baseline : dcp_to_torch_save -> torch.load -> safetensors save_file
  ckpt           : inspect, convert DCP -> safetensors, convert DCP -> HF sharded (100MB),
                   reshard HF -> 3 shards, diff (DCP vs safetensors), safetensors -> DCP (4 ranks)

Usage: bench.py [--dir DIR] [--bin PATH]   (DIR default: $CKPT_DATA_DIR/bench, i.e. <repo>/data/bench)
"""

import argparse
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

TORCH_BASELINE = r"""
import sys, torch, tempfile, os
from torch.distributed.checkpoint.format_utils import dcp_to_torch_save
from safetensors.torch import save_file
src, out = sys.argv[1], sys.argv[2]
tmp = out + ".pt"
dcp_to_torch_save(src, tmp)
sd = torch.load(tmp, weights_only=True)
sd = {k: v.contiguous() for k, v in sd["model"].items()}  # planner_data re-nests {"model": {...}}
save_file(sd, out)
os.remove(tmp)
"""


def make(d, py):
    code = r"""
import os, sys, socket, torch, torch.distributed as dist, torch.multiprocessing as mp
import torch.distributed.checkpoint as dcp
def port():
    s = socket.socket(); s.bind(("127.0.0.1", 0)); p = s.getsockname()[1]; s.close(); return p
def w(rank, world, p, out):
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(p)); torch.set_num_threads(1)
    dist.init_process_group("gloo", rank=rank, world_size=world)
    from transformers import GPT2Config, GPT2LMHeadModel
    from torch.distributed.fsdp import fully_shard
    from torch.distributed.device_mesh import init_device_mesh
    torch.manual_seed(0)
    m = GPT2LMHeadModel(GPT2Config(n_layer=10, n_embd=768, n_head=12, vocab_size=32000, n_positions=1024, tie_word_embeddings=False))
    mesh = init_device_mesh("cpu", (world,))
    for b in m.transformer.h: fully_shard(b, mesh=mesh)
    fully_shard(m, mesh=mesh)
    dcp.save({"model": m.state_dict()}, checkpoint_id=out)
    dist.destroy_process_group()
if __name__ == "__main__":
    mp.spawn(w, args=(4, port(), sys.argv[1]), nprocs=4, join=True)
"""
    script = os.path.join(d, "_make_bench_dcp.py")  # spawn needs an importable __main__
    with open(script, "w") as f:
        f.write(code)
    subprocess.run([py, script, os.path.join(d, "dcp")], check=True)


def timed(label, cmd, results):
    r = subprocess.run(["/usr/bin/time", "-f", "%e %M", *cmd], capture_output=True, text=True)
    last = r.stderr.strip().splitlines()[-1]
    secs, kb = last.split()
    ok = "ok" if r.returncode in (0,) else f"exit {r.returncode}"
    results.append((label, float(secs), int(kb) / 1024, ok))
    print(f"{label:<55} {float(secs):7.2f} s   peak RSS {int(kb)/1024:8.1f} MiB   {ok}", flush=True)
    if r.returncode not in (0,):
        print(r.stdout[-2000:], r.stderr[-2000:])


def main():
    ap = argparse.ArgumentParser()
    data = os.environ.get("CKPT_DATA_DIR") or os.path.join(HERE, "..", "data")
    ap.add_argument("--dir", default=os.path.join(data, "bench"), help="default: $CKPT_DATA_DIR/bench (<repo>/data/bench)")
    ap.add_argument("--bin", default=os.environ.get("CKPT_BIN"))
    a = ap.parse_args()
    d, b, py = a.dir, a.bin, sys.executable
    os.makedirs(d, exist_ok=True)
    if not os.path.exists(os.path.join(d, "dcp", ".metadata")):
        print("creating benchmark DCP checkpoint (4 gloo ranks)...", flush=True)
        make(d, py)
    for x in ["torch.safetensors", "ckpt.safetensors", "hf", "hf3", "dcp_out"]:
        p = os.path.join(d, x)
        shutil.rmtree(p, ignore_errors=True) if os.path.isdir(p) else (os.path.exists(p) and os.remove(p))
    res = []
    timed("torch: dcp_to_torch_save + torch.load + save_file", [py, "-c", TORCH_BASELINE, f"{d}/dcp", f"{d}/torch.safetensors"], res)
    timed("ckpt inspect (DCP)", [b, "inspect", f"{d}/dcp", "--summary"], res)
    timed("ckpt convert DCP -> single safetensors", [b, "convert", f"{d}/dcp", "--strip-prefix", "model.", "-o", f"{d}/ckpt.safetensors"], res)
    timed("ckpt convert DCP -> HF sharded (100MB)", [b, "convert", f"{d}/dcp", "--strip-prefix", "model.", "--max-shard-size", "100MB", "-o", f"{d}/hf"], res)
    timed("ckpt reshard HF -> 3 shards", [b, "reshard", f"{d}/hf", "--num-shards", "3", "-o", f"{d}/hf3"], res)
    timed("ckpt diff DCP vs torch-produced safetensors", [b, "diff", f"{d}/dcp", f"{d}/torch.safetensors", "--strip-prefix-a", "model."], res)
    timed("ckpt convert safetensors -> DCP (4 ranks)", [b, "convert", f"{d}/ckpt.safetensors", "--to", "dcp", "--dcp-ranks", "4", "-o", f"{d}/dcp_out"], res)
    size = os.path.getsize(f"{d}/ckpt.safetensors")
    print(f"\ncheckpoint size: {size/2**20:.1f} MiB")


if __name__ == "__main__":
    main()
