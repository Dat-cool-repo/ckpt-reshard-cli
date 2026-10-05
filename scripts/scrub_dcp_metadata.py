#!/usr/bin/env python3
"""Strip the absolute save path from PyTorch DCP `.metadata` files.

torch.distributed.checkpoint records the `checkpoint_id` it was saved to (an absolute path on the
machine that generated it) in `Metadata.storage_meta`. That leaks local directory names into
fixtures, so the fixture generators call this to replace it with the directory's basename.
Nothing else in the metadata changes (it is re-pickled with the original protocol).

Only run this on checkpoints you generated yourself: it unpickles the file with Python's pickle.

Usage: scrub_dcp_metadata.py DIR_OR_METADATA [...]   (directories are searched recursively)
"""

import os
import pathlib
import pickle
import sys


def scrub(meta: pathlib.Path) -> bool:
    raw = meta.read_bytes()
    proto = raw[1] if raw[:1] == b"\x80" else 2
    md = pickle.loads(raw)  # trusted input: a checkpoint this repo's scripts just wrote
    sm = getattr(md, "storage_meta", None)
    cid = getattr(sm, "checkpoint_id", None)
    if cid is None:
        return False
    base = pathlib.PurePath(str(cid)).name
    new = type(cid)(base) if isinstance(cid, os.PathLike) else base
    if str(cid) == str(new):
        return False
    sm.checkpoint_id = new
    tmp = meta.with_name(meta.name + ".tmp")
    tmp.write_bytes(pickle.dumps(md, protocol=proto))
    os.replace(tmp, meta)
    print(f"{meta}: checkpoint_id -> {new!r}", file=sys.stderr)
    return True


def scrub_tree(*roots) -> int:
    n = 0
    for r in map(pathlib.Path, roots):
        metas = [r] if r.is_file() else sorted(r.rglob(".metadata"))
        n += sum(scrub(m) for m in metas)
    return n


if __name__ == "__main__":
    import torch.distributed.checkpoint  # noqa: F401  (classes referenced by the pickle)

    if len(sys.argv) < 2:
        sys.exit(__doc__)
    scrub_tree(*sys.argv[1:])
