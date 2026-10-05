#!/usr/bin/env python3
"""Create *real* DeepSpeed ZeRO checkpoints on CPU (DS_ACCELERATOR=cpu, gloo backend).

A tiny model is trained for one step with bf16 + ZeRO stage 1, 2 and 3 on 3 data-parallel ranks
(3 ranks -> uneven partitions and alignment padding). It has odd-sized params, two param groups,
a registered buffer, a frozen layer and a tied (shared) weight. Written by engine.save_checkpoint:

  <out>/zero{1,2,3}/latest
  <out>/zero{1,2,3}/global_step1/{mp_rank_00_model_states.pt | zero_pp_rank_R_mp_rank_00_model_states.pt}
                                 bf16_zero_pp_rank_R_mp_rank_00_optim_states.pt
  <out>/zero{1,2,3}/reference.safetensors   DeepSpeed's own zero_to_fp32 result
                                            (get_fp32_state_dict_from_zero_checkpoint), tied weights cloned

Needs: pip install deepspeed (DS_BUILD_OPS=0); a C++ compiler for DeepSpeed's small CPU comm op.
"""

import argparse
import os
import socket
import sys

os.environ.setdefault("DS_ACCELERATOR", "cpu")
os.environ.setdefault("TORCH_EXTENSIONS_DIR", os.path.join(sys.prefix, "torch_extensions"))

import torch  # noqa: E402
import torch.multiprocessing as mp  # noqa: E402

WORLD = 3


class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = torch.nn.Embedding(37, 12)
        self.l1 = torch.nn.Linear(12, 7)
        self.up = torch.nn.Linear(7, 12)
        self.norm = torch.nn.LayerNorm(12)
        self.head = torch.nn.Linear(12, 37, bias=False)
        self.head.weight = self.emb.weight  # tied -> DeepSpeed "shared_params"
        self.register_buffer("scale", torch.tensor([1.5, 2.5, 3.25]))
        self.frozen = torch.nn.Linear(3, 5)
        self.frozen.requires_grad_(False)

    def forward(self, x):
        h = self.norm(self.up(torch.relu(self.l1(self.emb(x)))))
        return self.head(h) * self.scale[0] + self.frozen.weight.sum()


def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def worker(rank, world, port, stage, out):
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port), RANK=str(rank), WORLD_SIZE=str(world),
                      LOCAL_RANK=str(rank), LOCAL_WORLD_SIZE=str(world))
    torch.set_num_threads(1)
    import deepspeed

    deepspeed.init_distributed(dist_backend="gloo")
    torch.manual_seed(0)
    m = Tiny()
    decay = [p for n, p in m.named_parameters() if p.requires_grad and p.dim() >= 2]
    no_decay = [p for n, p in m.named_parameters() if p.requires_grad and p.dim() < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": 0.1}, {"params": no_decay, "weight_decay": 0.0}], lr=1e-2)
    cfg = {"train_micro_batch_size_per_gpu": 2, "zero_optimization": {"stage": stage},
           "bf16": {"enabled": True}, "zero_allow_untested_optimizer": True}
    eng, _, _, _ = deepspeed.initialize(model=m, optimizer=opt, config=cfg)
    g = torch.Generator().manual_seed(rank)
    for _ in range(2):
        x = torch.randint(0, 37, (2, 5), generator=g)
        loss = eng(x).float().pow(2).mean()
        eng.backward(loss)
        eng.step()
    eng.save_checkpoint(f"{out}/zero{stage}", tag="global_step1")


def reference(out, stage):
    from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint
    from safetensors.torch import save_file

    sd = get_fp32_state_dict_from_zero_checkpoint(f"{out}/zero{stage}")
    save_file({k: v.detach().clone().contiguous() for k, v in sd.items()}, f"{out}/zero{stage}/reference.safetensors")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = os.path.abspath(a.out)
    for stage in (1, 2, 3):
        mp.spawn(worker, args=(WORLD, free_port(), stage, out), nprocs=WORLD, join=True)
        reference(out, stage)
    print(f"deepspeed fixtures written to {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
