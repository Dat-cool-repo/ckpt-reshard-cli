# Design note: S3 / GCS support

Status: **not implemented.** This note records how it is meant to fit into the existing code.

## Reads

Every reader in `ckpt` already works on byte ranges of a memory-mapped file (`MappedFile` in
`src/ckpt.rs`):

| Format | What is read | Access pattern |
|---|---|---|
| safetensors | 8-byte length, then the JSON header, then tensor byte ranges | 2 small reads, then ranges |
| DCP | `.metadata` (one pickle), then each chunk's `(file, offset, length)` | one small file, then ranges |
| torch.save / DCP `torch_save` chunks / Megatron / DeepSpeed | zip end-of-central-directory and central directory from the tail, `data.pkl`, then storage entries by offset | tail read, small read, then ranges |

So the plan is to replace the `Mmap` inside `MappedFile` with a small trait:

```rust
trait Source: Send + Sync {
    fn len(&self) -> u64;
    fn read_range(&self, range: std::ops::Range<u64>) -> anyhow::Result<bytes::Bytes>;
}
```

- **Local files** keep using `mmap` (zero-copy, as today).
- **Remote objects** use the [`object_store`](https://crates.io/crates/object_store) crate (S3, GCS,
  Azure, HTTP) with ranged GETs and a small block cache:
  - headers, `.metadata`, zip central directories and `data.pkl` are cached;
  - tensor bytes are streamed and **not** cached, so memory stays bounded by the largest tensor.
- **Prefetching:** the writers process tensors in a known order, so a prefetcher can issue the next
  tensor's ranged GETs in parallel and hide latency. DCP chunks of one tensor can be fetched
  concurrently, too.
- **Listing:** format detection currently checks for marker files (`.metadata`,
  `model.safetensors.index.json`, `latest_checkpointed_iteration.txt`, `latest`, ...). These become
  `HEAD` requests or one prefix `LIST`.

## Writes

The writers (`src/writer.rs`, `src/dcp_write.rs`, `src/tp.rs`) are sequential. Each writes one output
file at a time through a `Write` and needs only the final size at the end. The one exception is the
safetensors header, which is computed up front from the declared tensors and so needs no seek.
Remote outputs therefore map onto **multipart uploads**: one upload per output file, with parts of
8–64 MiB flushed as the writer produces bytes. The current atomic `*.tmp-ckpt` + rename becomes
"complete the multipart upload last".

## CLI surface

URLs would be accepted anywhere a path is today, with credentials from the standard environment
(`AWS_*`, `GOOGLE_APPLICATION_CREDENTIALS`, ...):

```bash
ckpt inspect s3://bucket/run1/step_1000
ckpt convert gs://bucket/megatron/iter_0010000 -o s3://bucket/hf/step_10000
```

`--verify` works unchanged, because CRC32 is computed over the streamed bytes.
