# Motivation and background

This is the original problem statement behind `ckpt`, plus a survey of existing tools. For usage, see the
[README](../README.md).

## The problem

Large training runs save checkpoints in framework-specific sharded formats laid out by the
parallelism config (TP/PP/DP sizes). Moving a checkpoint to a different parallelism layout, a
different framework, or to Hugging Face format for evaluation usually means a one-off Python script
that:

- imports the original training framework (Megatron-LM, DeepSpeed, torch.distributed) at the exact
  version that wrote the checkpoint;
- loads the **whole model** (often with optimizer state) into RAM;
- unpickles every shard with `torch.load(weights_only=False)`, which executes arbitrary code embedded
  in the file.

Practitioners running 10k-GPU jobs describe this as a routine pain:

- [Llama3训练每3小时崩一次？豆包大模型、港大团队为脆皮万卡训练提效](https://juejin.cn/post/7400580315181252619)
  ("Llama 3 training crashes every 3 hours? ByteDance Doubao and HKU make fragile 10k-GPU training
  more efficient", Juejin): formats differ between Megatron, FSDP and DeepSpeed, and resharding needs
  hand-written scripts.
- [Tokyo 30の舞台裏 (Turing)](https://zenn.dev/turing_motors/articles/588954c08dccc0) ("Behind the
  scenes of Tokyo 30", Zenn): large-cluster operations and checkpoint/restart pain.
- [pytorch/torchtitan#850](https://github.com/pytorch/torchtitan/issues/850): DCP shards can't be read
  without custom code.

## Existing tools (surveyed Oct 2026)

| Tool | Limitation |
|---|---|
| torch `format_utils.dcp_to_torch_save` / `torch_save_to_dcp` | Needs torch; loads the **entire** state dict into RAM (756 MiB RSS and 10x slower than `ckpt` on a 461 MiB checkpoint, see the README benchmarks). |
| torch `_consolidate_hf_safetensors` | Private API; only handles DCP written with `HuggingFaceStorageWriter`. |
| HF accelerate `merge-weights` | Python and torch; loads the full DCP into memory. |
| DeepSpeed `zero_to_fp32.py` | Python and torch; `torch.load(weights_only=False)` of every rank file (executes arbitrary pickles); holds the full model in RAM. |
| Megatron-Bridge, Megatron-LM `tools/checkpoint/convert.py`, verl `model_merger` | Python and torch; need Megatron importable and the full model in memory; each covers one direction. |
| PyTorch DCP resharding | Happens inside a running PyTorch job. |
| Rust/Go tools (`safetensors_explorer`, `ztensor-cli`, `safetensors-browser`, `pt-loader`) | safetensors/GGUF or single `torch.save` files only; none reads DCP, Megatron or DeepSpeed shards. |

ByteCheckpoint was reported as open-sourced but has not been evaluated here.

Nothing found offers a torch-free, streaming, non-executing reader that covers DCP, Megatron and
DeepSpeed with `inspect` / `diff` / `reshard` / `convert` in one binary. That is the gap `ckpt` fills.

## Original scope

```
ckpt inspect ./step_10000/                 # format detection, tensor list, shapes, dtypes, shard map
ckpt diff ./step_10000 ./step_11000        # per-tensor L2/max-abs diff, changed-fraction
ckpt convert ./megatron_tp4_pp2 --to hf -o ./hf/
ckpt reshard ./hf --tp 8 --rules rules.yaml -o ./tp8/
```

- Format detection and readers for DCP, HF safetensors (sharded) and Megatron `mp_rank_XX` (via a safe
  pickle subset).
- Streaming and mmap: never hold the full model in RAM.
- Resharding rules for column-parallel and row-parallel linear layers, and for vocab-parallel
  embeddings.
- Read from and write to S3/GCS through `object_store`.

Stretch goals:

- DeepSpeed ZeRO stage 1/2/3 and FSDP sharded/full state dicts.
- Optimizer state resharding (Adam moments).
- A `verify` mode: checksums, plus a forward-pass smoke test through a tiny Python hook.
- Parallel upload/download for very large checkpoints.

Everything above except S3/GCS, optimizer-state resharding and the forward-pass hook is implemented.
The README's *Limitations* section lists what is still missing.

## Risks identified up front

- Megatron's naming and fused-QKV layouts vary by version and model family. `ckpt` therefore refuses
  layouts it does not know rather than guessing.
- Correctness bugs are silent. Every conversion path is therefore checked bit-exact against the
  framework's own tooling, and the Megatron → HF path is also checked against a real forward pass.
- Testing at scale is limited without access to large checkpoints. Synthetic big tensors cover this,
  for example the real 4 GiB+ zip64 round trip in `scripts/test_zip64_big.py`.
