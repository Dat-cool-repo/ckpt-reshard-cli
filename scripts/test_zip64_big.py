#!/usr/bin/env python3
"""Real zip64 test: one ~4.2 GiB tensor -> DCP chunk archive > 4 GiB, checked by torch, zipfile and ckpt.

  1. writes big.safetensors with a single uint8 tensor of 4.5e9 elements (a cheap deterministic pattern)
  2. ckpt convert big.safetensors --to dcp --dcp-ranks 1  (-> __0_0.distcp is one zip64 torch.save archive:
     data/0 is > 4 GiB, so the entries after it and the central directory sit beyond 4 GiB as well)
  3. python zipfile: archive structure + CRC of every entry (testzip)
  4. torch.load(mmap=True, weights_only=True) of the chunk, compared slice by slice with the pattern
  5. ckpt inspect --verify (our CRC check) + ckpt diff --verify DCP vs the source
  6. deletes everything

Env: CKPT_BIN, CKPT_BIG_DIR (default $CKPT_DATA_DIR/scratch/zip64, CKPT_DATA_DIR defaults to <repo>/data).
Needs ~8.5 GB free disk.
"""

import json
import os
import pathlib
import shutil
import struct
import subprocess
import sys
import time
import zipfile

import numpy as np
import torch

N = 4_500_000_000  # elements (uint8) -> 4.19 GiB
BLOCK = 1 << 26


def pattern(start, n):
    i = np.arange(start, start + n, dtype=np.uint64)
    return ((i * 2654435761) >> np.uint64(13)).astype(np.uint8)


def main():
    root = pathlib.Path(__file__).resolve().parent.parent
    bin_ = os.environ.get("CKPT_BIN") or str(pathlib.Path(os.environ.get("CARGO_TARGET_DIR", root / "target")) / "release" / "ckpt")
    data = pathlib.Path(os.environ.get("CKPT_DATA_DIR") or root / "data")
    d = pathlib.Path(os.environ.get("CKPT_BIG_DIR") or data / "scratch" / "zip64")
    shutil.rmtree(d, ignore_errors=True)
    d.mkdir(parents=True)
    try:
        t0 = time.time()
        src = d / "big.safetensors"
        hdr = json.dumps({"big": {"dtype": "U8", "shape": [N], "data_offsets": [0, N]}}).encode()
        hdr += b" " * ((8 - len(hdr) % 8) % 8)
        with open(src, "wb") as f:
            f.write(struct.pack("<Q", len(hdr)) + hdr)
            for s in range(0, N, BLOCK):
                f.write(pattern(s, min(BLOCK, N - s)).tobytes())
        print(f"wrote {src} ({src.stat().st_size / 2**30:.2f} GiB) in {time.time() - t0:.0f}s", flush=True)

        t0 = time.time()
        out = d / "dcp"
        r = subprocess.run(["/usr/bin/time", "-f", "%e s %M KB", bin_, "convert", src, "--to", "dcp", "-o", out],
                           capture_output=True, text=True)
        assert r.returncode == 0, r.stderr
        print("ckpt convert:", r.stderr.strip().splitlines()[-1], flush=True)
        chunk = out / "__0_0.distcp"
        print(f"chunk archive {chunk.stat().st_size / 2**30:.2f} GiB", flush=True)

        t0 = time.time()
        z = zipfile.ZipFile(chunk)
        infos = {i.filename: i for i in z.infolist()}
        data = infos["archive/data/0"]
        assert data.file_size == N and data.compress_size == N
        after = [i for i in z.infolist() if i.header_offset > 2**32]
        assert after, "expected entries whose local header lies beyond 4 GiB"
        bad = z.testzip()
        assert bad is None, f"zipfile CRC check failed for {bad}"
        print(f"zipfile: {len(infos)} entries, {len(after)} beyond 4 GiB, testzip OK ({time.time() - t0:.0f}s)", flush=True)

        t0 = time.time()
        t = torch.load(chunk, mmap=True, weights_only=True)
        assert t.dtype == torch.uint8 and t.shape == (N,)
        for s in range(0, N, BLOCK):
            n = min(BLOCK, N - s)
            if not np.array_equal(t[s:s + n].numpy(), pattern(s, n)):
                raise AssertionError(f"torch.load data mismatch in block at {s}")
        del t
        print(f"torch.load(mmap=True): all {N} bytes match ({time.time() - t0:.0f}s)", flush=True)

        t0 = time.time()
        r = subprocess.run([bin_, "inspect", "--verify", "--summary", out], capture_output=True, text=True)
        assert r.returncode == 0 and "CRC32 OK" in r.stdout, r.stdout + r.stderr
        r = subprocess.run([bin_, "--verify", "diff", out, src], capture_output=True, text=True)
        assert r.returncode == 0 and "RESULT: EQUAL" in r.stdout, r.stdout + r.stderr
        print(f"ckpt --verify inspect + diff: OK ({time.time() - t0:.0f}s)", flush=True)
        print("ZIP64 BIG TEST PASSED")
    finally:
        shutil.rmtree(d, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
