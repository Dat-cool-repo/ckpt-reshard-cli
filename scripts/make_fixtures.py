#!/usr/bin/env python3
"""Create real PyTorch Distributed Checkpoint (DCP) fixtures on CPU.

Runs N processes with the gloo backend (torch.multiprocessing.spawn) on a tiny,
randomly initialised GPT-2, and writes:

  <out>/reference/model.safetensors     full (unsharded) weights after 1 Adam step, from rank 0
  <out>/hf_sharded/                     same weights via transformers save_pretrained(max_shard_size=...)
  <out>/dcp_fsdp/                       FSDP2 (fully_shard) DCP: {"model": ..., "optim": ...}, torch_save chunks
  <out>/dcp_fsdp_st/                    same as dcp_fsdp but DCP SerializationFormat.SAFETENSORS chunks
  <out>/dcp_2d/                         2x2 DeviceMesh, 2-D weights Shard(0)+Shard(1), bf16 (TP-like blocks)
  <out>/reference_2d/model.safetensors  full bf16 reference for dcp_2d
  <out>/reference_optim/optim.safetensors  full optimizer tensors (flattened DCP names) for dcp_fsdp

Usage: python make_fixtures.py --out DIR [--nproc 4] [--layers 2] [--embd 128] [--vocab 1001]

The committed tests/fixtures/tiny set was made with
  --layers 1 --embd 16 --vocab 37 --positions 8 --hf-shard 4KB
"""

import argparse
import os
import socket
import sys

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
import torch.multiprocessing as mp


def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def make_model(args):
    from transformers import GPT2Config, GPT2LMHeadModel

    cfg = GPT2Config(
        n_layer=args.layers,
        n_embd=args.embd,
        n_head=4,
        vocab_size=args.vocab,  # deliberately not divisible by world size -> uneven shards
        n_positions=args.positions,
        tie_word_embeddings=False,
    )
    torch.manual_seed(1234)
    return GPT2LMHeadModel(cfg)


def flatten(prefix, obj, out):
    """Flatten nested dict/list the same way DCP's flatten_state_dict does (dot-joined)."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            flatten(f"{prefix}.{k}" if prefix else str(k), v, out)
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            flatten(f"{prefix}.{i}", v, out)
    else:
        out[prefix] = obj


def worker(rank, world, port, args):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    torch.set_num_threads(1)
    dist.init_process_group("gloo", rank=rank, world_size=world)

    from safetensors.torch import save_file
    from torch.distributed.checkpoint.filesystem import FileSystemWriter, SerializationFormat
    from torch.distributed.checkpoint.state_dict import (
        StateDictOptions,
        get_model_state_dict,
        get_optimizer_state_dict,
    )
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.fsdp import fully_shard
    from torch.distributed.tensor import Replicate, Shard, distribute_tensor

    out = args.out
    # ---------------- 1) FSDP2 checkpoint (fp32, model + optimizer) ----------------
    model = make_model(args)
    mesh = init_device_mesh("cpu", (world,))
    for block in model.transformer.h:
        fully_shard(block, mesh=mesh)
    fully_shard(model, mesh=mesh)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    torch.manual_seed(99)
    ids = torch.randint(0, args.vocab, (2, min(16, args.positions)))
    loss = model(input_ids=ids, labels=ids).loss
    loss.backward()
    opt.step()
    opt.zero_grad()

    msd = get_model_state_dict(model)
    osd = get_optimizer_state_dict(model, opt)
    state = {"model": msd, "optim": osd}
    dcp.save(state, checkpoint_id=f"{out}/dcp_fsdp")
    dcp.save(
        {"model": msd},
        storage_writer=FileSystemWriter(
            f"{out}/dcp_fsdp_st", serialization_format=SerializationFormat.SAFETENSORS
        ),
    )

    full = StateDictOptions(full_state_dict=True, cpu_offload=True)
    full_msd = get_model_state_dict(model, options=full)
    full_osd = get_optimizer_state_dict(model, opt, options=full)
    if rank == 0:
        ref = {k: v.contiguous() for k, v in full_msd.items()}
        os.makedirs(f"{out}/reference", exist_ok=True)
        save_file(ref, f"{out}/reference/model.safetensors")
        # HF-sharded copy of the same weights (fresh unsharded model + load)
        hf = make_model(args)
        hf.load_state_dict(ref)
        hf.save_pretrained(f"{out}/hf_sharded", max_shard_size=args.hf_shard)
        flat = {}
        flatten("optim", full_osd, flat)
        flat = {k: v.contiguous() for k, v in flat.items() if isinstance(v, torch.Tensor)}
        os.makedirs(f"{out}/reference_optim", exist_ok=True)
        save_file(flat, f"{out}/reference_optim/optim.safetensors")
    dist.barrier()

    # ---------------- 2) 2-D sharded checkpoint (bf16, TP-like blocks) ----------------
    if world % 2 == 0 and world >= 4:
        mesh2 = init_device_mesh("cpu", (2, world // 2), mesh_dim_names=("dp", "tp"))
        base = make_model(args).to(torch.bfloat16).state_dict()
        sd2 = {}
        for k, v in base.items():
            if v.dim() >= 2:
                sd2[k] = distribute_tensor(v, mesh2, [Shard(0), Shard(1)])
            else:
                sd2[k] = distribute_tensor(v, mesh2, [Shard(0), Replicate()])
        dcp.save(sd2, checkpoint_id=f"{out}/dcp_2d")
        if rank == 0:
            os.makedirs(f"{out}/reference_2d", exist_ok=True)
            save_file({k: v.contiguous() for k, v in base.items()}, f"{out}/reference_2d/model.safetensors")
        dist.barrier()

    dist.destroy_process_group()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--nproc", type=int, default=4)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--embd", type=int, default=128)
    ap.add_argument("--vocab", type=int, default=1001)
    ap.add_argument("--positions", type=int, default=128)
    ap.add_argument("--hf-shard", default="300KB")
    args = ap.parse_args()
    args.out = os.path.abspath(args.out)
    os.makedirs(args.out, exist_ok=True)
    mp.spawn(worker, args=(args.nproc, free_port(), args), nprocs=args.nproc, join=True)
    from scrub_dcp_metadata import scrub_tree

    scrub_tree(args.out)  # drop the absolute save path torch records in each .metadata
    print(f"fixtures written to {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
