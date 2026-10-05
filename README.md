# ckpt

[![CI](https://github.com/Dat-cool-repo/ckpt-reshard-cli/actions/workflows/ci.yml/badge.svg)](https://github.com/Dat-cool-repo/ckpt-reshard-cli/actions/workflows/ci.yml)
[![License: MIT OR Apache-2.0](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)](#license)

**Inspect, convert, reshard and diff distributed training checkpoints from one fast Rust binary, with
no Python, no torch and no pickle execution.**

`ckpt` reads Hugging Face safetensors (single or sharded), PyTorch Distributed Checkpoint (DCP, as
written by FSDP2 / DTensor), Megatron-LM (legacy `torch` and `torch_dist`), DeepSpeed ZeRO 1/2/3 and
plain `torch.save` files. It can:

- merge TP/PP-sharded Megatron checkpoints into HF Llama/Qwen2 checkpoints;
- rebuild fp32 weights from ZeRO partitions;
- split and merge tensor-parallel layouts, including GQA and padded vocabularies;
- cast between fp32/bf16/fp16/fp8 with results bit-identical to `torch.Tensor.to`.

Tensors are memory-mapped and streamed, so peak memory is about the size of the largest tensor, not
the size of the model.

```console
$ ckpt inspect ./megatron_ckpt --summary
path:    ./megatron_ckpt
format:  megatron-legacy
tensors: 17   params: 6.03K (6032)   bytes: 23.56 KiB (24128)
dtypes:  F32=6.03K
iteration: 10
checkpoint_version: 3.0
tensor_parallel: 2
pipeline_parallel: 2
args:
  num_layers                               2
  ...
$ ckpt convert ./megatron_ckpt -o ./hf            # TP/PP merge -> HF Qwen2 safetensors + config.json
$ ckpt diff ./step_1000 ./step_2000 --json        # per-tensor max/mean abs diff, rel L2, cosine
```

## Contents

- [Why](#why)
- [Supported formats](#supported-formats)
- [Install](#install)
- [Usage](#usage): [inspect](#inspect) · [convert](#convert) · [reshard](#reshard) ·
  [diff](#diff) · [`--dtype`](#--dtype-casts) · [`--verify`](#--verify-crc32-checks)
- [Safety model](#safety-model)
- [Correctness](#correctness)
- [Benchmarks](#benchmarks)
- [Limitations](#limitations)
- [S3 / GCS](#s3--gcs)
- [Development](#development)
- [License](#license)

## Why

Today, converting a distributed checkpoint (Megatron TP4/PP2 to HF, ZeRO-3 to fp32, FSDP DCP to
safetensors, TP8 to TP2) usually means a one-off Python script that:

- **needs torch and the original training code**, often at the exact Megatron or DeepSpeed version
  that wrote the files;
- **loads the whole model into RAM**, often with optimizer state;
- **runs arbitrary code**, because `torch.load(weights_only=False)` on every rank file executes
  whatever the pickle says. That is the default in `zero_to_fp32.py` and in most conversion scripts.

`ckpt` is a single static binary that understands these layouts directly:

- it parses the pickles with a restricted, non-executing interpreter;
- it maps the shards and streams tensor bytes to the output;
- it has been checked bit-exact against each framework's own tools.

See [docs/MOTIVATION.md](docs/MOTIVATION.md) for the background and a survey of existing tools.

## Supported formats

| Format | Detected by | inspect | convert (source) | convert (target) | reshard (source) | diff |
|---|---|:-:|:-:|:-:|:-:|:-:|
| safetensors (single file) | `*.safetensors` | yes | yes | yes (`--to safetensors`) | yes | yes |
| HF sharded safetensors | `model.safetensors.index.json` | yes | yes | yes (`--to hf`) | yes | yes |
| PyTorch DCP (FSDP2 / DTensor, `torch_save` or safetensors chunks, n-D shard grids) | `.metadata` + `__R_N.distcp` | yes | yes | yes (`--to dcp`) | yes | yes |
| `torch.save` file | `.pt` / `.bin` / `.pth` | yes | yes | no | yes | yes |
| Megatron-LM legacy (`ckpt_format=torch`), TP and PP | `latest_checkpointed_iteration.txt`, `iter_*/mp_rank_TT[_PPP]/model_optim_rng.pt` | yes | yes¹ | no | yes¹ | yes |
| Megatron-LM `torch_dist` | `iter_*/.metadata` + `metadata.json` | yes | yes¹ | no | yes¹ | yes |
| DeepSpeed ZeRO stage 1/2/3 | `latest`, `global_stepN/*_model_states.pt` + `*_optim_states.pt` | yes | yes² | no | yes² | yes |
| `ckpt` tensor-parallel layout | `tp_rank_XX/` + `tp_plan.json` | yes | yes | yes (`reshard --tp`) | yes (merge / re-split) | yes |

¹ TP/PP-merged and, unless `--keep-names` is given, mapped to HF Llama/Qwen2 names with a generated
`config.json`. Optimizer state is listed but not merged.
² Reconstructed fp32 weights, as DeepSpeed's `zero_to_fp32.py` would produce them.

## Install

From GitHub (installs the `ckpt` binary into `~/.cargo/bin`; the trailing package name is needed
because the repository also contains the `fuzz/` crate):

```bash
cargo install --git https://github.com/Dat-cool-repo/ckpt-reshard-cli ckpt-reshard-cli
```

From source:

```bash
git clone https://github.com/Dat-cool-repo/ckpt-reshard-cli
cd ckpt-reshard-cli
cargo build --release        # -> target/release/ckpt
```

You need a recent stable Rust toolchain (the crate uses edition 2024). Nothing else is required at
runtime: no Python, no CUDA, no torch.

## Usage

```
ckpt [--verify] [--threads N] <inspect|convert|reshard|diff> ...
```

Every command auto-detects the input format. Exit codes are `0` for success (or *equal* for `diff`),
`1` for *different* (`diff` only) and `2` for errors.

### inspect

```bash
ckpt inspect ./step_1000                        # format, tensors, dtypes, shapes, params, bytes per file
ckpt inspect ./step_1000 --summary              # totals only
ckpt inspect ./step_1000 --json                 # machine-readable
ckpt inspect ./dcp_ckpt --chunks                # DCP shard grid: per-chunk offsets, sizes and files
ckpt inspect ./step_1000 --filter '*layers.0.*' # glob filter (repeatable)
ckpt inspect ./megatron_ckpt                    # + TP/PP size, iteration, key Megatron args
ckpt inspect ./megatron_ckpt --hf-arch auto     # as it would look after the HF mapping
ckpt --raw inspect ./megatron_ckpt/iter_0000010 # torch_dist as stored (stacked layers)
ckpt inspect ./ds_ckpt                          # + ZeRO stage, world size, param groups, DeepSpeed version
ckpt inspect ./ds_ckpt/global_step1/mp_rank_00_model_states.pt   # one raw torch.save file
```

Non-tensor entries such as optimizer hyperparameters, iteration counters and Megatron `args` are
decoded and shown as values.

### convert

```bash
# DCP (FSDP2) -> one safetensors file, keeping only the model weights
ckpt convert ./dcp_ckpt -o model.safetensors --include 'model.*' --strip-prefix model.

# anything -> HF sharded safetensors (+ model.safetensors.index.json)
ckpt convert ./dcp_ckpt -o ./hf --to hf --max-shard-size 5GB

# safetensors -> DCP with 8 rank files (row-split like FSDP; loadable by torch.distributed.checkpoint)
ckpt convert ./model.safetensors --to dcp --dcp-ranks 8 -o ./dcp_ckpt

# Megatron (legacy or torch_dist, any TP/PP) -> HF Llama/Qwen2 safetensors + config.json
ckpt convert ./megatron_ckpt -o ./hf
ckpt --hf-arch llama convert ./megatron_ckpt -o ./hf            # force the architecture
ckpt convert ./megatron_ckpt -o m.safetensors --keep-names --vocab-size 50257   # merged Megatron names

# DeepSpeed ZeRO 1/2/3 -> fp32 weights (what zero_to_fp32.py produces), or bf16 with fp32 norms
ckpt convert ./ds_ckpt -o fp32.safetensors
ckpt convert ./ds_ckpt -o ./hf --dtype bf16 --keep-dtype '*norm*'
```

Name filters: `--include GLOB`, `--exclude GLOB`, `--strip-prefix P` and `--add-prefix P`. The output
format follows the `--out` extension unless `--to safetensors|hf|dcp` is given. When converting an HF
directory, config and tokenizer files are copied too (`--no-aux` disables this).

The Megatron → HF mapping (`--hf-arch auto|llama|qwen2`):

- splits the GQA-interleaved `linear_qkv` into `q_proj`/`k_proj`/`v_proj`, with biases;
- splits SwiGLU `linear_fc1` into `gate_proj`/`up_proj`;
- unpads the vocabulary to `args.vocab_size` or `--vocab-size`;
- writes `config.json` from the Megatron args.

`auto` picks Qwen2 for models with only a qkv bias, and Llama otherwise.

### reshard

Change the shard count or size of an HF checkpoint:

```bash
ckpt reshard ./hf -o ./hf3 --num-shards 3
ckpt reshard ./hf -o ./hf_2g --max-shard-size 2GB
```

Split a model into tensor-parallel ranks. This writes `tp_rank_XX/model.safetensors` plus a
`tp_plan.json` that records the resolved layout:

```bash
ckpt reshard ./qwen2 --tp 8 --rules examples/llama_tp_rules.yaml -o ./tp8   # GQA + vocab padding
ckpt reshard ./tp8 --tp 2 -o ./tp2                  # re-split: no rules needed, the plan has them
ckpt reshard ./tp8 -o merged.safetensors            # merge back (bit-exact)
ckpt --hf-arch auto reshard ./megatron_ckpt --tp 4 --rules examples/llama_tp_rules.yaml -o ./tp4
```

#### TP rules

Rules are YAML. The first rule whose `pattern` matches a tensor wins, and unmatched tensors use
`default` (`replicate`). See [`examples/llama_tp_rules.yaml`](examples/llama_tp_rules.yaml)
(Llama/Qwen2/Mistral, GQA, fused qkv, vocab padding) and
[`examples/gpt2_tp_rules.yaml`](examples/gpt2_tp_rules.yaml) (GPT-2 Conv1D, fused qkv).

```yaml
default: replicate
config:                       # optional; otherwise read from <src>/config.json or --model-config
  num_attention_heads: 32
  num_key_value_heads: 8
rules:
  - {pattern: "*.self_attn.q_proj.weight", dim: 0, heads: num_attention_heads}   # column parallel
  - {pattern: "*.self_attn.k_proj.weight", dim: 0, kv_heads: num_key_value_heads}
  - {pattern: "*.self_attn.o_proj.weight", dim: 1, heads: num_attention_heads}   # row parallel
  - pattern: "*.self_attn.qkv_proj.weight"                                        # fused, unequal q/k/v
    dim: 0
    sections: [{heads: num_attention_heads}, {kv_heads: num_key_value_heads}, {kv_heads: num_key_value_heads}]
  - {pattern: "*.mlp.gate_up_proj.weight", dim: 0, parts: 2}                      # fused, equal blocks
  - {pattern: "*.embed_tokens.weight", dim: 0, pad_multiple: 64}                  # vocab parallel
```

| Key | Meaning |
|---|---|
| `dim` | Split dimension: 0 is column parallel for `nn.Linear` `[out, in]`, 1 is row parallel. |
| `parts: N` | The tensor is N equal fused blocks along `dim`; each block is split separately. |
| `heads` / `kv_heads` | Split on head boundaries; the value names a head count from `config` / `config.json`. |
| `sections` | Unequal fused blocks such as q, k and v under GQA. Each rank gets its q heads plus its k and v heads. |
| `pad_multiple: M` | Zero-pad rows to a multiple of `tp * M` (Megatron's `--make-vocab-size-divisible-by`). The padding is removed on merge. |

With GQA, when `tp > num_key_value_heads` (and `tp % kv_heads == 0`), each kv head is replicated so that
rank `r` holds kv head `r // (tp / kv_heads)`. That is exactly the kv head its q heads use under HF's
`repeat_kv`. Merging checks that the replicas are bit-identical.

### diff

```bash
ckpt diff ./step_1000 ./step_2000                         # any two formats
ckpt diff ./dcp_ckpt ./hf --strip-prefix-a model. --ignore-missing --tolerance 1e-6  # skip optim.*
ckpt diff a.safetensors b.safetensors --json --all        # per-tensor stats for every tensor
ckpt diff ./ds_ckpt ./hf --include '*.mlp.*' --ignore-missing
```

For each tensor, `diff` reports max and mean absolute difference, relative L2, cosine similarity and
the fraction of changed elements. Shape, dtype and missing-name mismatches are reported too. Exit code 0
means equal (within `--tolerance`), 1 means different.

### `--dtype` casts

`convert` and `reshard`, including TP split/merge and DCP output, accept
`--dtype fp32|bf16|fp16|fp8_e4m3|fp8_e5m2|fp64`:

```bash
ckpt convert ./dcp_ckpt -o model-bf16.safetensors --strip-prefix model. --dtype bf16
ckpt convert ./model.safetensors -o model-fp8.safetensors --dtype fp8_e4m3 --keep-dtype '*norm*' --keep-dtype 'lm_head.*'
ckpt reshard ./hf --tp 4 --rules examples/llama_tp_rules.yaml --dtype bf16 -o ./tp4
```

Only floating-point tensors are cast; integer and bool tensors are kept, and so are tensors matching
`--keep-dtype`. The conversions are ports of c10's scalar routines with round-to-nearest-even, and
match `torch.Tensor.to` **bit for bit** on every finite and infinite value:

- f64 inputs are first rounded to f32, as torch does.
- `float8_e4m3fn` saturates to ±448 on overflow and ±inf, as torch 2.14 does. Older torch releases
  returned NaN here.
- `float8_e5m2` overflows to ±inf.
- fp8 is a plain cast with no scaling factors, the same as `t.to(torch.float8_e4m3fn)`.
- NaN payloads are canonicalised. torch's own vectorised and scalar kernels disagree on them.

### `--verify` CRC32 checks

`--verify` is a global flag. It CRC32-checks every zip entry that is read: DCP `torch_save` chunks,
`data.pkl` records, and Megatron, DeepSpeed and `torch.save` storages. `inspect --verify` checks every
archive in the checkpoint.

```bash
ckpt inspect ./dcp_ckpt --verify --summary          # ... CRC32 OK
ckpt --verify convert ./megatron_ckpt -o ./hf
ckpt --verify diff ./dcp_ckpt ./hf
```

A mismatch fails with exit code 2:

```
CRC32 mismatch in zip entry "archive/data/0" (… bytes at offset …): central directory says 0x…, data hashes to 0x… -- the file is corrupted
```

Without `--verify`, reads stay zero-copy and unhashed. DCP output larger than 4 GiB per chunk is
written as ZIP64.

## Safety model

DCP `.metadata`, DCP `torch_save` chunks, `.pt` files, Megatron rank files and DeepSpeed state files are
all Python pickles. `ckpt` never runs Python and never unpickles with Python. `src/pickle.rs` is a small
pickle stack machine written in Rust:

- **Nothing is executed.** `GLOBAL`/`STACK_GLOBAL` only record an inert `(module, name)` pair, and
  `REDUCE`, `NEWOBJ` and `BUILD` only record an inert object node. Nothing is imported or called. The
  readers then pattern-match those nodes, for example `torch._utils._rebuild_tensor_v2(storage,
  offset, size, stride)`, to locate tensor bytes.
- **Per-format allowlists**, as defence in depth. Any global outside the allowlist aborts the load with
  `refusing pickle global ...`. That covers `os.system`, `posix.system`, `builtins.eval`/`exec`,
  `subprocess.*` and so on. Each list is the minimum its real writers need:
  - **Checkpoint:** DCP metadata/planner dataclasses, `torch.Size`, dtypes, storage classes,
    `_rebuild_tensor_v2`, `_rebuild_parameter`, `_rebuild_from_type_v2(…, torch.Tensor, …)`,
    `OrderedDict`, `pathlib` paths and `_codecs.encode`.
  - **Megatron** adds `argparse.Namespace`, the enums found in real `args` (`AttnBackend`,
    `ModelType`, `signal.Signals`, …) and numpy's `_reconstruct`/`ndarray`/`dtype` for the RNG state.
  - **DeepSpeed** adds `LossScaler`/`DynamicLossScaler`, `ZeroStageEnum` and `fragment_address`.
- **Refused opcodes.** Extension-registry opcodes and out-of-band buffers are rejected.
- **No stack exhaustion.** Containers live in a flat arena, so cycles and deep nesting are harmless,
  and graph walkers have depth limits.
- **Bounds checks.**
  - Zip, safetensors, DCP and `torch.save` offsets are checked against file sizes.
  - Every tensor view (offset, sizes and strides × dtype) is checked against its storage.
  - File names taken from indexes, metadata and tracker files may not escape the checkpoint directory.

The test suites build malicious checkpoints with `os.system` payloads in four places: a DCP
`.metadata`, a pickle inside a DCP chunk, a Megatron rank file and a DeepSpeed optimizer file. All four
are refused with exit code 2, and the payload never runs. See [tests/README.md](tests/README.md).

## Correctness

Each path is checked against the reference implementation, on CPU, with torch 2.14.1, transformers
5.18, megatron-core 0.19.2 and deepspeed 0.19.7:

| What | Reference | Result |
|---|---|---|
| DCP (FSDP2, 2-D `Shard(0)+Shard(1)` bf16, safetensors-chunk DCP) → safetensors | torch `get_model_state_dict(full_state_dict=True)` | bit-exact, model and optimizer tensors |
| safetensors → DCP (`--to dcp`, also forced ZIP64) | `dcp_to_torch_save` and FSDP2 `dcp.load` on 2 gloo ranks | bit-exact; also loads with `torch.load(mmap=True)` and passes Python `zipfile.testzip` |
| HF sharded output | `transformers` `from_pretrained` | loads, same tensors |
| `--dtype` (all 6 targets) | `torch.Tensor.to` | bit-exact on every bf16/fp16/fp8 value, all midpoints ±1 ulp, specials, 1M random fp32 bit patterns and random f64 |
| TP split for GQA (tp = 1, 2, 4, 8, with kv replication and vocab padding) | manual torch slicing; per-rank attention summed over ranks vs full attention | identical shards; split → merge bit-exact; tp8 → tp2 equals a fresh tp2 split |
| Megatron legacy TP2/PP2 merge | megatron-core's own TP=1 gather (`dist_checkpointing.load` into a TP1/PP1 model) | bit-exact |
| Megatron → HF Qwen2 | logits of the real TP2/PP2 Megatron forward pass vs `Qwen2ForCausalLM.from_pretrained` | max abs diff 2e-7 |
| Megatron `torch_dist` → HF | the legacy conversion | identical files |
| DeepSpeed ZeRO 1/2/3 (bf16, 3 ranks, uneven partitions, 2 param groups, frozen and tied params) | DeepSpeed `get_fp32_state_dict_from_zero_checkpoint` | bit-exact |
| ZIP64 DCP chunk > 4 GiB (one 4.19 GiB tensor) | Python `zipfile`, `torch.load(mmap=True)` | structure valid, CRC OK, all 4.5e9 bytes match |

The Megatron and DeepSpeed fixtures are real checkpoints written by megatron-core and DeepSpeed
themselves, on CPU with gloo; see [Development](#development). One deliberate difference from
`zero_to_fp32`: buffers keep their stored dtype, whereas `zero_to_fp32` applies `.float()` to them.
`--dtype fp32` gives the same result.

## Benchmarks

The checkpoint was an FSDP2 DCP of about 121M fp32 params (461 MiB, GPT-2 style with 10 layers,
d=768, vocab 32000) written by 4 gloo ranks. Measured with `scripts/bench.py` on an Intel i9-13900H,
WSL2, ext4, warm page cache:

| Task | Time | Peak RSS |
|---|---:|---:|
| torch: `dcp_to_torch_save` + `torch.load` + `safetensors.save_file` | 2.79 s | 756 MiB |
| `ckpt convert` DCP → single safetensors | **0.27 s** | **109 MiB** |
| `ckpt convert` DCP → HF sharded (100 MB shards) | 0.26 s | 109 MiB |
| `ckpt reshard` HF → 3 shards | 0.28 s | 103 MiB |
| `ckpt diff` DCP vs safetensors (461 MiB each side) | 0.10 s | 377 MiB |
| `ckpt convert` safetensors → DCP (4 ranks, CRC32 per record) | 0.59 s | 126 MiB |
| `ckpt inspect` (DCP) | < 0.01 s | 6 MiB |

These dtype and verify runs were on the same checkpoint and machine, but under heavier background
load: a plain DCP → safetensors convert took 0.71–0.78 s in the same runs.

| Task | Time | Peak RSS |
|---|---:|---:|
| `ckpt convert` DCP → safetensors `--dtype bf16` | 0.45–0.76 s | 119 MiB |
| `ckpt convert` safetensors → `--dtype fp16` | 0.73–0.85 s | 113 MiB |
| `ckpt convert` safetensors → `--dtype fp8_e4m3` | 0.92–1.08 s | 109 MiB |
| `ckpt --verify convert` DCP → safetensors (CRC of every chunk) | 0.58–0.77 s | 109 MiB |

`ckpt`'s peak RSS tracks the largest tensor, here the 98 MB embedding, while torch's grows with the
model. Writing a 4.19 GiB ZIP64 DCP chunk took 197 s on a Windows drive mounted in WSL (drvfs, about
20 MB/s for every tool); that run is I/O-bound.

## Limitations

These are not supported yet:

- **Object storage:** reading from or writing to S3/GCS (see below).
- **Megatron:**
  - optimizer-state merging (legacy optimizer and `distrib_optim.pt`);
  - virtual-pipeline (`model0`, `model1`, …) and expert-parallel (`mp_rank_TT_PPP_EEE`) layouts;
  - HF mappings for MLA, MoE and qk-layernorm (Qwen3) models;
  - HF → Megatron.

  Learned position embeddings, interleaved RoPE, qk-layernorm, virtual PP and expert parallel are
  refused with a clear error rather than converted wrongly.
- **DeepSpeed:**
  - reconstruction of optimizer moments (`exp_avg`, `exp_avg_sq`);
  - universal checkpoints;
  - ZeRO combined with model parallelism (`mp_rank_01` and up).
- **Tensor parallel:**
  - output is `tp_rank_XX/` safetensors, not Megatron rank files;
  - no pipeline-parallel output layouts.
- **Output formats:** no `torch.save` or Megatron/DeepSpeed output. Outputs are safetensors, HF sharded
  safetensors, DCP and the `ckpt` TP layout.
- **Memory:** `diff` assembles non-contiguous tensors in memory, one tensor per thread at a time. It
  could compare chunk by chunk instead.
- **Platforms:** the release binary is built and unit-tested on Linux, macOS and Windows by CI. The
  end-to-end suites against real torch, Megatron and DeepSpeed run on Linux only.

## S3 / GCS

Object storage is not implemented yet, but the readers were designed for it. Every reader already
works on byte ranges:

- safetensors headers;
- the DCP `.metadata` plus per-chunk `(offset, length)`;
- zip central directories read from the tail;
- `torch.save` storages by entry offset.

The plan is a small `Source` trait (`len`, `read_range`) backed by the
[`object_store`](https://crates.io/crates/object_store) crate, replacing the `Mmap` behind each
mapped file. Reads become ranged GETs with a small block cache: headers and central directories are
cached, and tensor bytes are streamed. Writes become multipart uploads, one part stream per output
file. The writers are already sequential, apart from the safetensors header, which is computed up
front. Prefetching the next tensor's ranges in parallel would hide latency. Details are in
[docs/OBJECT_STORAGE.md](docs/OBJECT_STORAGE.md).

## Development

```
src/main.rs        CLI (clap)                     src/ckpt.rs       format detection, unified mmap view
src/pickle.rs      restricted pickle VM           src/dcp.rs        DCP .metadata + chunk decoding
src/torchsave.rs   torch.save archive walker      src/dcp_write.rs  DCP writer (pickle emitter, zip/zip64)
src/megatron.rs    Megatron legacy/torch_dist, TP/PP merge, HF mapping
src/deepspeed.rs   ZeRO 1/2/3 fp32 reconstruction src/cast.rs       --dtype conversion (torch-exact)
src/zipread.rs     zero-copy zip reader + CRC     src/safetensors.rs header parse + streaming writer
src/writer.rs      single/HF-sharded output       src/tp.rs         TP rules (GQA, padding), split/merge
src/diff.rs        per-tensor stats               src/inspect.rs    inspect output
```

### Rust tests (no Python needed)

```bash
cargo fmt --all -- --check
cargo clippy --all-targets -- -D warnings
cargo test --release           # unit tests + CLI tests on tests/fixtures/tiny
```

The CLI tests run against tiny committed checkpoints in `tests/fixtures/tiny`, about 1.3 MB in
total. These are self-generated, randomly initialised toy models (hidden size 12–16), not real
model weights. They cover DCP, HF sharded, Megatron TP2/PP2 legacy and `torch_dist`, and DeepSpeed
ZeRO-2/3. See [tests/README.md](tests/README.md) for how each was made.

### End-to-end tests (Linux, CPU torch)

The pytest suites generate real checkpoints with torch.distributed (FSDP2, 4 gloo ranks),
megatron-core (TP2/PP2) and DeepSpeed (ZeRO-1/2/3, 3 ranks). They then compare `ckpt` against the
frameworks' own loaders and against transformers. Everything runs on CPU. The script below needs
[`uv`](https://github.com/astral-sh/uv) and a C++ compiler, because DeepSpeed JIT-builds a small CPU
comm op.

```bash
source scripts/env.sh     # CARGO_TARGET_DIR, CKPT_DATA_DIR (default ./data), CKPT_FIXTURES, CKPT_BIN
scripts/test_all.sh       # cargo build + test, creates a venv (CPU torch, transformers,
                          # megatron-core, deepspeed), generates fixtures, runs pytest
```

| Script | Purpose |
|---|---|
| `scripts/make_fixtures.py --out DIR` | DCP fixtures: FSDP2 model+optim, safetensors-chunk DCP, 2-D sharded bf16, HF sharded, references |
| `scripts/make_megatron_fixtures.py --out DIR [--tiny]` | Real Megatron-LM legacy and `torch_dist` checkpoints (TP2/PP2), the TP1 reference and forward-pass logits |
| `scripts/make_deepspeed_fixtures.py --out DIR` | Real DeepSpeed ZeRO-1/2/3 checkpoints and `zero_to_fp32` references |
| `scripts/scrub_dcp_metadata.py DIR` | Remove the absolute save path torch records in DCP `.metadata` (the generators call it) |
| `scripts/verify_dcp_load.py DCP REF CFG` | Load a `ckpt`-written DCP with `dcp_to_torch_save` and with FSDP2 `dcp.load` |
| `scripts/test_zip64_big.py` | Real 4.19 GiB ZIP64 round trip; needs about 8.5 GB of free disk (`CKPT_BIG_DIR`) |
| `scripts/bench.py [--dir DIR]` | The benchmark above |

Generated data goes to `$CKPT_DATA_DIR`, which defaults to `./data` (git-ignored). CI runs fmt, clippy
and `cargo test` on Linux, macOS and Windows. A Linux job also runs the DCP, dtype and GQA-TP pytest
suites with CPU torch. The Megatron and DeepSpeed suites need megatron-core and DeepSpeed, so they are
run locally through `scripts/test_all.sh`.

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or <https://www.apache.org/licenses/LICENSE-2.0>)
- MIT license ([LICENSE-MIT](LICENSE-MIT) or <https://opensource.org/licenses/MIT>)

at your option.

Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in the
work by you, as defined in the Apache-2.0 license, shall be dual licensed as above, without any
additional terms or conditions.
