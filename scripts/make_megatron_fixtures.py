#!/usr/bin/env python3
"""Create *real* Megatron-LM checkpoints on CPU (gloo) with megatron-core.

A tiny Qwen2-like GPT (GQA, SwiGLU, RMSNorm, RoPE, qkv bias, untied embeddings, padded vocab) is
built with megatron-core's GPTModel under tensor parallel = 2 and pipeline parallel = 2 (4 gloo ranks).
Weights get one Adam step so they are not just the init. Then:

  <out>/legacy/                       Megatron-LM "torch" format (ckpt_format=torch):
      latest_checkpointed_iteration.txt
      iter_0000010/mp_rank_{tp:02d}_{pp:03d}/model_optim_rng.pt
    The state dict is produced by megatron.training.checkpointing.generate_state_dict (real args
    Namespace from megatron.training.arguments.parse_args, model.state_dict_for_save_checkpoint(),
    optimizer state, rng state) and written with torch.save to get_checkpoint_name(...), i.e. exactly
    what save_checkpoint() does for CheckpointType.LEGACY.
  <out>/torch_dist/iter_0000010/      Megatron "torch_dist" (DCP-based) checkpoint written with
                                      megatron.core.dist_checkpointing.save(model.sharded_state_dict())
  <out>/reference/model.safetensors   the full (TP=1, global layer numbering) Megatron state dict, gathered
                                      on rank 0 by megatron-core itself via the torch_dist round trip
                                      (dist_checkpointing.load into a TP=1/PP=1 model)
  <out>/reference/logits.safetensors  logits of the TP2/PP2 model's real forward pass (input_ids too)
  <out>/reference/config.json         orig vocab size etc. used by the tests

CPU-only shims (documented, do not change what is saved):
  * megatron.core.transformer.moe.ops.paged_stash imports triton (GPU kernels for MoE) at import time;
    an inert stand-in module is registered so megatron.core imports without triton.
  * torch.cuda.get_rng_state is unavailable on CPU: rng_state["cuda_rng_state"] holds the CPU RNG state;
    torch.cuda.synchronize (called by the torch_dist writer before copying tensors) is a no-op and
    torch.cuda.current_device() returns "cpu" (the writer all-reduces a failure flag on it).
  * the CUDA RNG tracker fork() used around attention dropout is replaced by a no-op context for the
    eval-mode reference forward (dropout is 0).
"""

import argparse
import contextlib
import json
import os
import socket
import sys
import types

TP, PP = 2, 2
ITER = 10
# model size; "tiny" (--tiny) is used for the fixtures committed under tests/fixtures/tiny/megatron
SIZES = {
    "small": dict(layers=4, hidden=64, ffn=96, heads=8, groups=4, kv=8, vocab=250, div=64),
    "tiny": dict(layers=2, hidden=16, ffn=24, heads=4, groups=2, kv=4, vocab=50, div=16),
}
SZ = SIZES[os.environ.get("CKPT_MEGATRON_SIZE", "small")]
ORIG_VOCAB = SZ["vocab"]


def _shim_megatron_imports():
    m = types.ModuleType("megatron.core.transformer.moe.ops.paged_stash")
    m.GLOBAL_BLOCK_SIZE = 1024
    m.paged_stash_copy_kernel = m.paged_stash_pop_kernel = None
    sys.modules[m.__name__] = m


_shim_megatron_imports()

import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402
import torch.multiprocessing as mp  # noqa: E402


def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def megatron_args(tp, pp, ckpt_format):
    from megatron.training.arguments import parse_args

    argv = [
        "prog",
        "--num-layers", str(SZ["layers"]), "--hidden-size", str(SZ["hidden"]), "--ffn-hidden-size", str(SZ["ffn"]),
        "--num-attention-heads", str(SZ["heads"]), "--group-query-attention", "--num-query-groups", str(SZ["groups"]),
        "--kv-channels", str(SZ["kv"]), "--seq-length", "16", "--max-position-embeddings", "32",
        "--position-embedding-type", "rope", "--rotary-base", "10000", "--normalization", "RMSNorm",
        "--swiglu", "--disable-bias-linear", "--add-qkv-bias",
        "--untie-embeddings-and-output-weights", "--make-vocab-size-divisible-by", str(SZ["div"]),
        "--tensor-model-parallel-size", str(tp), "--pipeline-model-parallel-size", str(pp),
        "--micro-batch-size", "2", "--global-batch-size", "2", "--lr", "0.01",
        "--tokenizer-type", "NullTokenizer", "--vocab-size", str(ORIG_VOCAB),
        "--ckpt-format", ckpt_format, "--save", "/nonexistent", "--use-cpu-initialization",
        "--no-gradient-accumulation-fusion", "--no-bias-swiglu-fusion", "--no-masked-softmax-fusion",
        "--no-rope-fusion", "--no-persist-layer-norm",
    ]
    old = sys.argv
    sys.argv = argv
    try:
        args = parse_args()
    finally:
        sys.argv = old
    # what the tokenizer builder computes in Megatron-LM (pad to make_vocab_size_divisible_by * tp)
    mult = args.make_vocab_size_divisible_by * tp
    args.padded_vocab_size = ((ORIG_VOCAB + mult - 1) // mult) * mult
    args.iteration = ITER
    args.world_size = tp * pp
    args.data_parallel_size = 1
    args.params_dtype = torch.float32
    return args


def build_model(args, pre, post):
    import torch.nn.functional as F
    from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
    from megatron.core.models.gpt.gpt_model import GPTModel
    from megatron.core.transformer.transformer_config import TransformerConfig

    cfg = TransformerConfig(
        num_layers=args.num_layers, hidden_size=args.hidden_size, ffn_hidden_size=args.ffn_hidden_size,
        num_attention_heads=args.num_attention_heads, num_query_groups=args.num_query_groups,
        kv_channels=args.kv_channels, use_cpu_initialization=True, gated_linear_unit=True,
        activation_func=F.silu, normalization="RMSNorm", layernorm_epsilon=args.layernorm_epsilon,
        add_bias_linear=False, add_qkv_bias=True, pipeline_dtype=torch.float32,
        tensor_model_parallel_size=args.tensor_model_parallel_size,
        pipeline_model_parallel_size=args.pipeline_model_parallel_size,
        bias_activation_fusion=False, masked_softmax_fusion=False, persist_layer_norm=False,
        bias_dropout_fusion=False, apply_rope_fusion=False, gradient_accumulation_fusion=False,
        hidden_dropout=0.0, attention_dropout=0.0,
    )
    return GPTModel(
        config=cfg, transformer_layer_spec=get_gpt_layer_local_spec(normalization="RMSNorm"),
        vocab_size=args.padded_vocab_size, max_sequence_length=args.max_position_embeddings,
        position_embedding_type="rope", rotary_base=args.rotary_base,
        share_embeddings_and_output_weights=False, pre_process=pre, post_process=post,
        parallel_output=False,
    )


class FP32Optimizer:
    """Same state_dict() contract as megatron.core.optimizer.FP32Optimizer (wraps a torch optimizer)."""

    is_stub_optimizer = False

    def __init__(self, opt):
        self.optimizer = opt

    def state_dict(self):
        return self.optimizer.state_dict()


def causal_mask(s):
    return torch.triu(torch.ones(s, s, dtype=torch.bool), 1)[None, None]


def forward_pp(model, ids, rank, pp_rank, tp_world):
    """Real 2-stage pipeline forward: stage 0 -> send hidden -> stage 1 -> gathered logits."""
    from megatron.core import parallel_state as ps

    pos = torch.arange(ids.shape[1])[None].expand_as(ids)
    with torch.no_grad():
        if pp_rank == 0:
            h = model(ids, pos, causal_mask(ids.shape[1]))
            dist.send(h.contiguous(), dst=rank + tp_world)
            return None
        h = torch.empty(ids.shape[1], ids.shape[0], model.config.hidden_size)
        dist.recv(h, src=rank - tp_world)
        model.set_input_tensor(h)
        return model(ids, pos, causal_mask(ids.shape[1]))


def worker(rank, world, port, out):
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    torch.set_num_threads(1)
    dist.init_process_group("gloo", rank=rank, world_size=world)
    from megatron.core import dist_checkpointing
    from megatron.core import parallel_state as ps
    from megatron.core import tensor_parallel
    from megatron.training import checkpointing as mckpt
    from megatron.training.global_vars import set_args

    ps.initialize_model_parallel(tensor_model_parallel_size=TP, pipeline_model_parallel_size=PP)
    tp_rank, pp_rank = ps.get_tensor_model_parallel_rank(), ps.get_pipeline_model_parallel_rank()
    tracker = tensor_parallel.get_cuda_rng_tracker()
    tracker.fork = lambda *a, **k: contextlib.nullcontext()
    torch.cuda.get_rng_state = lambda *a, **k: torch.get_rng_state()  # CPU stand-in, see docstring
    torch.cuda.synchronize = lambda *a, **k: None  # no CUDA streams on CPU
    torch.cuda.current_device = lambda: "cpu"  # torch_dist finalize all-reduces a status flag on it

    args = megatron_args(TP, PP, "torch")
    set_args(args)
    torch.manual_seed(1234)
    model = build_model(args, ps.is_pipeline_first_stage(), ps.is_pipeline_last_stage())

    # one optimizer step so weights are not the init (grads are random but deterministic per rank)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    g = torch.Generator().manual_seed(100 + rank)
    for p in model.parameters():
        p.grad = torch.randn(p.shape, generator=g)
    opt.step()
    opt.zero_grad()
    # keep TP-replicated params (norms, qkv bias is TP-split) identical across TP ranks like real training
    for name, p in model.named_parameters():
        if not getattr(p, "tensor_model_parallel", False):
            dist.broadcast(p.data, src=ps.get_tensor_model_parallel_src_rank(), group=ps.get_tensor_model_parallel_group())

    # ---------------- legacy torch format ----------------
    rng_state = mckpt.get_rng_state("torch", ps.get_tensor_model_parallel_group(), ps.get_pipeline_model_parallel_group())
    sd = mckpt.generate_state_dict(args, [model], FP32Optimizer(opt), None, rng_state, iteration=ITER)
    name = mckpt.get_checkpoint_name(f"{out}/legacy", ITER)
    os.makedirs(os.path.dirname(name), exist_ok=True)
    torch.save(sd, name)
    dist.barrier()
    if rank == 0:
        with open(mckpt.get_checkpoint_tracker_filename(f"{out}/legacy"), "w") as f:
            f.write(str(ITER))

    # ---------------- torch_dist format ----------------
    args.ckpt_format = "torch_dist"
    tdir = f"{out}/torch_dist/iter_{ITER:07d}"
    os.makedirs(tdir, exist_ok=True)
    msd = {"model": model.sharded_state_dict(), "args": args, "iteration": ITER, "checkpoint_version": 3.0}
    dist_checkpointing.save(msd, tdir)
    dist.barrier()
    if rank == 0:
        with open(f"{out}/torch_dist/latest_checkpointed_iteration.txt", "w") as f:
            f.write(str(ITER))

    # ---------------- reference logits: real TP2/PP2 forward ----------------
    model.eval()
    ids = torch.randint(0, ORIG_VOCAB, (2, 12), generator=torch.Generator().manual_seed(7))
    logits = forward_pp(model, ids, rank, pp_rank, TP)
    if logits is not None and tp_rank == 0:
        from safetensors.torch import save_file

        os.makedirs(f"{out}/reference", exist_ok=True)
        save_file({"logits": logits[..., :ORIG_VOCAB].contiguous(), "input_ids": ids},
                  f"{out}/reference/logits.safetensors")
    dist.barrier()
    dist.destroy_process_group()


def gather_reference(out):
    """Load the torch_dist checkpoint into a TP=1/PP=1 model with megatron-core (it does the resharding)."""
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(free_port()))
    dist.init_process_group("gloo", rank=0, world_size=1)
    from megatron.core import dist_checkpointing
    from megatron.core import parallel_state as ps
    from safetensors.torch import save_file

    ps.initialize_model_parallel(1, 1)
    args = megatron_args(1, 1, "torch_dist")
    mult = SZ["div"] * TP  # keep the TP=2 padding of the saved checkpoint
    args.padded_vocab_size = ((ORIG_VOCAB + mult - 1) // mult) * mult
    model = build_model(args, True, True)
    tdir = f"{out}/torch_dist/iter_{ITER:07d}"
    loaded = dist_checkpointing.load({"model": model.sharded_state_dict()}, tdir)
    model.load_state_dict(loaded["model"])
    sd = {k: v.contiguous() for k, v in model.state_dict_for_save_checkpoint().items() if torch.is_tensor(v)}
    save_file(sd, f"{out}/reference/model.safetensors")
    with open(f"{out}/reference/config.json", "w") as f:
        json.dump({"orig_vocab_size": ORIG_VOCAB, "padded_vocab_size": args.padded_vocab_size, "tp": TP, "pp": PP,
                   "num_layers": args.num_layers, "num_attention_heads": args.num_attention_heads,
                   "num_query_groups": args.num_query_groups, "kv_channels": args.kv_channels}, f, indent=1)
    dist.destroy_process_group()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--tiny", action="store_true", help="2 layers, hidden 16 (for the committed Rust test fixtures)")
    a = ap.parse_args()
    if a.tiny:
        os.environ["CKPT_MEGATRON_SIZE"] = "tiny"  # inherited by the spawned workers
        global SZ, ORIG_VOCAB
        SZ = SIZES["tiny"]
        ORIG_VOCAB = SZ["vocab"]
    out = os.path.abspath(a.out)
    os.makedirs(out, exist_ok=True)
    # non-daemonic: the torch_dist writer starts a multiprocessing Manager
    mp.start_processes(worker, args=(TP * PP, free_port(), out), nprocs=TP * PP, join=True, daemon=False, start_method="spawn")
    ctx = mp.get_context("spawn")
    p = ctx.Process(target=gather_reference, args=(out,))
    p.start()
    p.join()
    if p.exitcode != 0:
        sys.exit("reference gathering failed")
    from scrub_dcp_metadata import scrub_tree

    scrub_tree(f"{out}/torch_dist")  # drop the absolute save path torch records in .metadata
    print(f"megatron fixtures written to {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
