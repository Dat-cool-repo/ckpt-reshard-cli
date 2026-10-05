#!/usr/bin/env python3
"""Build the seed corpora for the fuzz targets from the committed fixtures (stdlib only).

    python3 fuzz/make_seeds.py OUT_DIR      # writes OUT_DIR/<target>/<seed>

The first byte of most targets' input selects a variant (allowlist, fixture file, flags); see the
targets in fuzz/fuzz_targets/.
"""
import hashlib
import sys
import zipfile
from pathlib import Path

FIX = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "tiny"


def put(out: Path, target: str, data: bytes) -> None:
    d = out / target
    d.mkdir(parents=True, exist_ok=True)
    (d / hashlib.sha1(data).hexdigest()).write_bytes(data)


def allow_of(p: Path) -> int:
    s = str(p)
    if "megatron" in s:
        return 1
    if "deepspeed" in s:
        return 2
    return 0


def main() -> None:
    out = Path(sys.argv[1])
    pts = sorted(p for p in FIX.rglob("*") if p.suffix in (".pt", ".distcp"))
    metas = sorted(FIX.rglob(".metadata"))
    for p in pts:
        b = p.read_bytes()
        put(out, "torch_save", bytes([allow_of(p)]) + b)
        try:
            with zipfile.ZipFile(p) as z:
                for n in z.namelist():
                    if n.endswith("data.pkl"):
                        put(out, "pickle", bytes([allow_of(p)]) + z.read(n))
        except zipfile.BadZipFile:
            pass
    for p in metas:
        put(out, "dcp_metadata", p.read_bytes())
        put(out, "pickle", bytes([allow_of(p)]) + p.read_bytes())
    for p in sorted(FIX.rglob("*.safetensors")):
        put(out, "safetensors", p.read_bytes())
    put(out, "hf_index", (FIX / "hf_sharded/model.safetensors.index.json").read_bytes())

    dcp = []
    for d in ["dcp_fsdp", "dcp_fsdp_st", "dcp_2d", "megatron/torch_dist/iter_0000010"]:
        dcp += [f"{d}/.metadata"] + [f"{d}/__{r}_0.distcp" for r in range(4)]
    for i, rel in enumerate(dcp):
        b = (FIX / rel).read_bytes()
        put(out, "dcp_dir", bytes([i]) + b)
        if i >= 15:
            put(out, "dcp_dir", bytes([i | 0x80]) + b)

    mg = [f"megatron/legacy/iter_0000010/mp_rank_0{t}_00{p}/model_optim_rng.pt" for t in range(2) for p in range(2)]
    for i, rel in enumerate(mg):
        b = (FIX / rel).read_bytes()
        for flags in (0, 0x40, 0x80, 0xC0):
            put(out, "megatron", bytes([i | flags]) + b)

    g2 = FIX / "deepspeed/zero2/global_step1"
    g3 = FIX / "deepspeed/zero3/global_step1"
    ds = [g2 / "mp_rank_00_model_states.pt"]
    ds += [g2 / f"bf16_zero_pp_rank_{r}_mp_rank_00_optim_states.pt" for r in range(3)]
    ds += [g3 / f"zero_pp_rank_{r}_mp_rank_00_model_states.pt" for r in range(3)]
    ds += [g3 / f"bf16_zero_pp_rank_{r}_mp_rank_00_optim_states.pt" for r in range(3)]
    for i, p in enumerate(ds):
        put(out, "deepspeed", bytes([i]) + p.read_bytes())
    for t in sorted(out.iterdir()):
        print(t.name, len(list(t.iterdir())))


if __name__ == "__main__":
    main()
