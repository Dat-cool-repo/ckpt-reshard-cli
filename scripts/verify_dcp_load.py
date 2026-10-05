#!/usr/bin/env python3
"""Load a DCP checkpoint written by `ckpt convert --to dcp` with real torch.distributed.

1. Non-distributed: torch's own `dcp_to_torch_save` -> torch.load -> compare with reference.
2. Distributed: N gloo processes, FSDP2-sharded GPT-2, `dcp.load` (DCP reshards on load),
   then gather the full state dict on rank 0 and compare with the reference.

Usage: verify_dcp_load.py DCP_DIR REFERENCE.safetensors CONFIG_DIR [--nproc 2]
Exit code 0 on bit-exact match.
"""

import argparse
import os
import socket
import sys
import tempfile

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
import torch.multiprocessing as mp
from safetensors.torch import load_file


def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def compare(got, ref, what):
    bad = [k for k in ref if k not in got or got[k].dtype != ref[k].dtype or not torch.equal(got[k], ref[k])]
    extra = [k for k in got if k not in ref]
    if bad or extra:
        print(f"{what}: MISMATCH bad={bad[:5]} extra={extra[:5]}", file=sys.stderr)
        return False
    print(f"{what}: OK ({len(ref)} tensors bit-exact)", file=sys.stderr)
    return True


def worker(rank, world, port, args, results):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    torch.set_num_threads(1)
    dist.init_process_group("gloo", rank=rank, world_size=world)
    from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict, set_model_state_dict
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.fsdp import fully_shard
    from transformers import GPT2Config, GPT2LMHeadModel

    cfg = GPT2Config.from_pretrained(args.config_dir)
    torch.manual_seed(7)  # different init than the checkpoint: load must overwrite everything
    model = GPT2LMHeadModel(cfg)
    mesh = init_device_mesh("cpu", (world,))
    for b in model.transformer.h:
        fully_shard(b, mesh=mesh)
    fully_shard(model, mesh=mesh)
    sd = get_model_state_dict(model)
    dcp.load(sd, checkpoint_id=args.dcp_dir)
    set_model_state_dict(model, sd)
    full = get_model_state_dict(model, options=StateDictOptions(full_state_dict=True, cpu_offload=True))
    if rank == 0:
        ref = load_file(args.reference)
        results.put(compare({k: v.contiguous() for k, v in full.items()}, ref, f"FSDP2 dcp.load on {world} ranks"))
    dist.barrier()
    dist.destroy_process_group()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dcp_dir")
    ap.add_argument("reference")
    ap.add_argument("config_dir")
    ap.add_argument("--nproc", type=int, default=2)
    args = ap.parse_args()

    from torch.distributed.checkpoint.format_utils import dcp_to_torch_save

    ok = True
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "full.pt")
        dcp_to_torch_save(args.dcp_dir, p)
        got = torch.load(p, weights_only=True)
        ok &= compare(got, load_file(args.reference), "dcp_to_torch_save")

    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    mp.start_processes(worker, args=(args.nproc, free_port(), args, q), nprocs=args.nproc, join=True, start_method="spawn")
    ok &= q.get()
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
