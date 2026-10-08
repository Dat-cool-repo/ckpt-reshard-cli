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

Tensors are memory-mapped and streamed, so `convert`, resharding and TP splitting need about as much
memory as the largest tensor, not the whole model: converting the 0.94 GiB Qwen2.5-0.5B, whose
largest tensor is 260 MiB, peaks at 260–300 MiB of RSS on Linux. Merging TP ranks and `diff` hold a few
tensors at once (see [Testing](#testing)). On macOS and Windows the mapped pages stay resident, so peak
RSS is about the size of the files read; see [Platforms](#platforms).

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
- [Testing](#testing)
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
| `ckpt` tensor-parallel layout | `tp_rank_XX/` + `tp_plan.json` | no³ | no³ | yes (`reshard --tp`) | yes (merge / re-split) | no³ |

¹ TP/PP-merged and, unless `--keep-names` is given, mapped to HF Llama/Qwen2 names with a generated
`config.json`. Optimizer state is listed but not merged.
² Reconstructed fp32 weights, as DeepSpeed's `zero_to_fp32.py` would produce them.
³ Only `reshard` reads the TP layout. Merge it first (`ckpt reshard ./tp4 -o merged.safetensors`), or
inspect a single rank (`ckpt inspect ./tp4/tp_rank_00`).

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

You need Rust 1.88 or newer (the `rust-version`; builds and tests were checked with 1.88.0). Nothing
else is required at runtime: no Python, no CUDA, no torch.

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
ckpt --raw inspect ./ds_ckpt/global_step1/mp_rank_00_model_states.pt   # one raw torch.save file
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
- **Bounds checks, before anything is allocated.**
  - Every shape, offset and length read from a file is checked with overflow-checked arithmetic.
    Shapes may have at most 64 dims and 2^56 bytes.
  - Zip, safetensors, DCP and `torch.save` offsets are checked against file sizes.
  - Every tensor view (offset, sizes and strides × dtype) is checked against its storage.
  - After opening, every tensor's chunks, views and pieces must lie inside the tensor and its
    files. A tensor may not declare more bytes than the files holding it contain, so a hostile
    header cannot make `ckpt` allocate more memory than the checkpoint occupies on disk.
  - File names taken from indexes, metadata and tracker files may not escape the checkpoint directory.
- **Resource limits.**
  - The pickle VM caps its stack at 16M values, MARK nesting at 65,536, and memo and node arena at
    50M entries each.
  - Values copied out of existing containers (`REDUCE` arguments, `set(...)`, `OrderedDict(...)`)
    may total at most 4× the pickle size plus 1M. That stops memo-reference abuse, where a small
    pickle rebuilds one big memoised list millions of times.
  - The state-dict walker visits at most 4× as many nodes as the pickle has, plus 4096, so a DAG of
    shared references cannot unfold into an exponentially large tree. Tensor names are capped at
    1024 bytes.
  - Zip entries must not overlap or repeat a name, and compressed entries are refused, so there is
    nothing to inflate (no zip bombs). Safetensors headers are capped at 100 MB, as in the reference
    implementation, and overlapping tensors are refused.
  - Megatron `torch_dist` checkpoints may stack at most 65,536 layers.
- **No panics.** Malformed input produces an error and exit code 2. Should a bug still panic, the
  panic is reported as an internal error with exit code 2.

The test suites build malicious checkpoints with `os.system` payloads in four places: a DCP
`.metadata`, a pickle inside a DCP chunk, a Megatron rank file and a DeepSpeed optimizer file. All four
are refused with exit code 2, and the payload never runs. See [tests/README.md](tests/README.md). The
resource limits are covered by hand-crafted hostile files and by fuzzing; see [Testing](#testing).

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
| Megatron → HF Qwen2 | logits of the real TP2/PP2 Megatron forward pass vs `Qwen2ForCausalLM.from_pretrained` | max abs diff 2.4e-7 (legacy and `torch_dist`) |
| Megatron `torch_dist` → HF | the legacy conversion | identical files |
| DeepSpeed ZeRO 1/2/3 (bf16, 3 ranks, uneven partitions, 2 param groups, frozen and tied params) | DeepSpeed `get_fp32_state_dict_from_zero_checkpoint` | bit-exact |
| ZIP64 DCP chunk > 4 GiB (one 4.19 GiB tensor) | Python `zipfile`, `torch.load(mmap=True)` | structure valid, CRC OK, all 4.5e9 bytes match |

The Megatron and DeepSpeed fixtures are real checkpoints written by megatron-core and DeepSpeed
themselves, on CPU with gloo; see [Development](#development). One deliberate difference from
`zero_to_fp32`: buffers keep their stored dtype, whereas `zero_to_fp32` applies `.float()` to them.
`--dtype fp32` gives the same result.

## Testing

| Suite | What it covers | Where it runs |
|---|---|---|
| Unit + CLI tests (`cargo test`) | Every format on the committed tiny fixtures, dtype casts, TP split/merge, CRC checks, and the four malicious-pickle cases | CI: Linux, macOS, Windows |
| Hostile-input tests ([`tests/hostile.rs`](tests/hostile.rs)) | Hand-crafted hostile files, built at test time (below) | CI: Linux, macOS, Windows |
| End-to-end pytest ([`tests/`](tests/README.md)) | Real FSDP2/DCP, Megatron-LM and DeepSpeed checkpoints against the frameworks' own loaders | CI: DCP, dtype and TP suites on Linux. Local: Megatron and DeepSpeed |
| Fuzzing ([`fuzz/`](fuzz)) | 8 libFuzzer targets, one per reader (below) | Local, before releases |
| Real models | Qwen2.5-0.5B-Instruct from the Hugging Face Hub (below) | Local |

### Hostile inputs

Each of these must exit with code 2 and a clear message, without a panic, within 20 s:

- safetensors with shapes whose product overflows `u64`, or a `0` dim hiding an overflowing
  stride;
- a 10^12-element shape backed by 8 bytes, negative or overflowing `data_offsets`, overlapping
  tensors, 65 dimensions, and a header length of 2^64-1;
- a DCP `.metadata` declaring 2^62 or 2^30 elements over a 64-byte file, negative or overflowing
  storage offsets and lengths, and chunk offsets of `i64::MAX`;
- a DCP chunk record whose view claims 2^40 elements of a 16-element storage;
- `torch.save` records with negative sizes, overflowing strides, or a storage larger than its zip
  entry;
- zip archives with overlapping or duplicate entries, compressed (deflate) entries, a zip64
  locator at offset 2^64-1, or a central directory past the end of the file;
- pickles with 100,000 nested MARKs, or a list nested 200,000 deep;
- memo-reference abuse: `set(memo[0])` repeated 1,000 times over a 100,000-item list, about 10^8
  copies without the limit;
- shared-reference DAGs that unfold into 2^60 paths (as values or as dict keys), or into 2^20
  copies of a tensor record;
- path traversal in `relative_path` and in HF index shard names.

Every input the fuzzer finds is minimized into `tests/fuzz_regressions/<target>/` and replayed by
the same test file.

### Fuzzing

`fuzz/` is a [cargo-fuzz](https://github.com/rust-fuzz/cargo-fuzz) crate. Every target that opens
a checkpoint then exercises it the way the CLI would: inspect with chunk grids and decoded
non-tensor items, `--verify`, read, stream and `read_box` every tensor, cast to bf16, and diff it
against itself. A global allocator aborts on any single allocation over 256 MiB, so a trusted
declared size shows up as a crash.

| Target | Input |
|---|---|
| `pickle` | raw bytes into the restricted pickle VM (all three allowlists), then the JSON walker |
| `dcp_metadata` | raw bytes as a DCP `.metadata` |
| `torch_save` | raw bytes as a `torch.save` zip: zip reader, DCP chunk decoder, and a whole `.pt` state dict |
| `safetensors` | raw bytes as a safetensors file, and as a safetensors-format DCP chunk |
| `hf_index` | `model.safetensors.index.json` next to the committed HF shards |
| `dcp_dir` | one `.metadata` or `.distcp` file of a committed DCP replaced (FSDP2, safetensors chunks, 2-D grid, Megatron `torch_dist` with the HF mapping) |
| `megatron` | one `model_optim_rng.pt` of the TP2/PP2 fixture replaced (merge, `--vocab-size`, HF mapping) |
| `deepspeed` | one model/optim states file of the ZeRO-2/ZeRO-3 fixtures replaced |

Before the 0.1 release every target ran for 14–38 minutes of fuzzing (about 2.7 CPU-hours in all,
≤ 2 h of wall time). The first 2 minutes of each were with AddressSanitizer. The long runs used no
sanitizer, so the directory targets were 30× faster, but kept debug assertions (overflow checks), the
allocation guard, an RSS limit of 1–2 GB and `-timeout=10`. Seeds come from the committed
fixtures (`fuzz/make_seeds.py`).

| Target | Minutes | Executions | Crashes |
|---|---:|---:|---|
| `pickle` | 14 | 3.69M | 0 |
| `dcp_metadata` | 14 | 6.44M | 0 |
| `torch_save` | 14¹ | 3.38M | 1, fixed (below) |
| `safetensors` | 14 | 4.98M | 0 |
| `hf_index` | 14 | 4.35M | 0 |
| `dcp_dir` | 38 | 263K | 0 |
| `megatron` | 38 | 895K | 0 |
| `deepspeed` | 14 | 1.06M | 0 |

¹ Plus an earlier `torch_save` campaign that found the crash; its logs were lost in a VM restart.
The 14 minutes counted here ran on the fixed code.

The crash was a 7.5 KB pickle whose dict keys were themselves dicts, nested many levels deep. Rendering a
non-string key as JSON escapes the quotes of the key inside it, so the text doubled at every level and
one allocation reached 512 MiB. Rendered keys are now capped at 128 characters, and copied strings
have a byte budget. The input is in `tests/fuzz_regressions/torch_save/`.

The hand-written hostile tests came out of a code review done alongside the fuzzing. Before the fixes, overflowing shapes
and offsets panicked, and a DCP or `torch.save` header could make `ckpt` allocate whatever size it
declared. A ZeRO-3 checkpoint whose ranks disagreed on partition sizes could loop forever, and
memo references and shared-reference DAGs could blow up memory or time.

To run a target (Linux, nightly Rust):

```bash
cargo install cargo-fuzz
python3 fuzz/make_seeds.py /tmp/seeds          # seed corpora from tests/fixtures/tiny
cd fuzz
cargo +nightly fuzz run -s none -a dcp_dir /tmp/corpus/dcp_dir /tmp/seeds/dcp_dir -- \
    -jobs=2 -workers=2 -rss_limit_mb=2048 -timeout=10 -max_total_time=1200
```

### Real models

[Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct) from the Hugging Face Hub:
0.94 GiB of bf16 safetensors, 290 tensors, 14 query heads and 2 kv heads, tied embeddings. Every
command below was run on it, then
[`scripts/verify_real_model.py`](scripts/verify_real_model.py) checked the outputs against the original
with torch 2.14.1 and transformers 5.18 on CPU:

```bash
M=./Qwen2.5-0.5B-Instruct; O=./out; R=examples/llama_tp_rules.yaml
ckpt convert $M --to dcp --dcp-ranks 4 -o $O/dcp
ckpt convert $O/dcp -o $O/rt.safetensors
ckpt convert $O/dcp --to hf --max-shard-size 500MB -o $O/rt_hf
ckpt reshard $M -o $O/sh500 --max-shard-size 500MB
ckpt reshard $M -o $O/sh3 --num-shards 3
ckpt reshard $M --tp 2 --rules $R -o $O/tp2
ckpt reshard $O/tp2 -o $O/tp2_merged.safetensors
ckpt reshard $M --tp 2 --rules $R --dtype fp16 -o $O/tp2_fp16
ckpt convert $M -o $O/fp16.safetensors --dtype fp16
ckpt convert $M -o $O/fp32.safetensors --dtype fp32
ckpt convert $O/fp32.safetensors -o $O/bf16.safetensors --dtype bf16
python scripts/verify_real_model.py $M $O "2" --logits
```

| Check | Result |
|---|---|
| DCP → safetensors, DCP → HF sharded, reshard by 500 MB and into 3 shards, TP2 → merge, fp32 → bf16 | all 290 tensors bit-identical to the original |
| `--dtype fp16`, `--dtype fp32` | bit-identical to `t.to(torch.float16)` / `t.float()` |
| TP2 split, with and without `--dtype fp16` | every tensor of every rank equals the torch slicing of the GQA rules |
| logits (fp32, 18-token prompt) of the DCP round trip, the 500 MB reshard and the TP2 merge | identical to the original's (`torch.equal`, max \|diff\| 0) |
| `ckpt diff` original vs round trip, DCP, TP2 merge | `RESULT: EQUAL`, exit 0; vs fp16: exit 1 |

Time and peak RSS on Linux, from WSL2 ext4 with a warm page cache. The machine (an i9-13900H) was
running fuzz jobs in the background at the time, so these numbers are pessimistic:

| Command | Time | Peak RSS |
|---|---:|---:|
| `inspect --summary` | < 0.01 s | 11 MiB |
| `convert` → DCP (4 ranks) | 3.1 s | 261 MiB |
| `--verify inspect` DCP (CRC of every chunk) | 0.16 s | 947 MiB¹ |
| `convert` DCP → safetensors | 0.8 s | 266 MiB |
| `convert` DCP → HF sharded | 1.2 s | 266 MiB |
| `reshard` into 3 shards | 0.8 s | 266 MiB |
| `reshard --tp 2` (split) | 4.6 s | 260 MiB |
| `reshard` TP2 → one file (merge) | 2.8 s | 955 MiB¹ |
| `convert --dtype fp16` | 3.3 s | 285 MiB |
| `convert --dtype fp32` | 4.0 s | 303 MiB |
| `diff` original vs round trip | 2.1 s | 531 MiB |

¹ The mapped pages of the files read stay resident (CRC of everything, all ranks of a tensor at
once). They count in RSS, but they are clean page cache that the kernel can drop.

On WSL2, read checkpoints from the Linux file system, not from a Windows drive such as `/mnt/c`: memory-mapped
reads through drvfs run at about 20 MB/s, so `--dtype fp16` took 53 s instead of 3.3 s.

### Platforms

| Platform | Status |
|---|---|
| Linux x86_64 | CI: fmt, clippy, unit, CLI and hostile-input tests, and the DCP/dtype/TP pytest suites. Local (WSL2 Ubuntu): all pytest suites, fuzzing and the real-model checks |
| Windows x86_64 | CI: fmt, clippy, unit, CLI and hostile-input tests (MSVC). The cross-built `x86_64-pc-windows-gnu` binary (MinGW, no extra DLLs) ran the same commands natively on Windows 11, on NTFS, for Qwen2.5-0.5B and Qwen2.5-1.5B. Every command succeeded with the expected exit codes. Peak working set is about the model size (0.95 GiB for 0.5B, 2.9 GiB for 1.5B), because on Windows the mapped pages are not released after each tensor; see the note below. For Qwen2.5-0.5B, all 38 output files (DCP shards and metadata, HF and resharded shard sets, TP2 ranks in bf16 and fp16, and dtype conversions) are byte-identical to the Linux outputs of the same commands, and the safetensors outputs also match macOS |
| macOS arm64 | CI-tested on GitHub's `macos-latest` runners: fmt, clippy, unit, CLI and hostile-input tests. The release binary and `cargo install --git` ran the same commands natively on an Apple Silicon machine (M5 Pro, macOS 27, Rust 1.99) for Qwen2.5-0.5B: every command succeeded with the expected exit codes, the DCP round trip and the TP2 split + merge are bit-identical to the original (`diff` exit 0), and `rt.safetensors` and `tp2_merged.safetensors` have the same SHA-256. Peak RSS is about the model size (1.0 GiB for 0.5B) for the same reason as on Windows; see the note below |

**Memory on Windows and macOS.** On Linux, `ckpt` tells the kernel it is done with each tensor's source
pages (`madvise(MADV_DONTNEED)`), so RSS stays near the largest tensor. Windows has no equivalent for
file mappings, and on macOS `madvise` accepts `MADV_DONTNEED` and `MADV_FREE` on a file mapping but does
not drop the pages from the resident set (measured on macOS 27 with both `MAP_SHARED` and
`MAP_PRIVATE`). On both, the peak working set therefore grows to about the size of the files read: 2.9
GiB to convert the 3.1 GB Qwen2.5-1.5B on Windows, 1.0 GiB to convert Qwen2.5-0.5B on macOS. These are
clean, file-backed pages that the OS trims under memory pressure, not private allocations, but they
show up as the process's memory use.

## Benchmarks

The checkpoint was an FSDP2 DCP of about 121M fp32 params (461 MiB, GPT-2 style with 10 layers,
d=768, vocab 32000) written by 4 gloo ranks. Measured with `scripts/bench.py` on an Intel i9-13900H,
WSL2, ext4, warm page cache, at the initial release. These tables were not re-measured after the
hardening for 0.1; the Qwen2.5-0.5B timings in [Testing](#real-models) are current:

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
  end-to-end suites against real torch, Megatron and DeepSpeed run on Linux only. See
  [Platforms](#platforms).

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
