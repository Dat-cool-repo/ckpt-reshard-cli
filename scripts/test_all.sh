#!/usr/bin/env bash
# Build + run every test (Rust unit/integration on tiny committed fixtures, then the end-to-end
# pytest suites against real torch.distributed DCP, megatron-core and DeepSpeed checkpoints).
# Run on Linux (or WSL). CPU only: torch comes from the CPU wheel index; megatron-core and deepspeed
# (DS_BUILD_OPS=0) are pure-python installs that pull no CUDA packages. Needs `uv` and a C++ compiler
# (DeepSpeed JIT-builds one small CPU comm op).
#
# Env: VENV (default <repo>/.venv, or ~/venvs/ckpt-reshard-cli on a WSL /mnt/<drive> checkout),
#      CKPT_DATA_DIR / CKPT_FIXTURES (see scripts/env.sh). Generated fixtures are reused if present.
set -euo pipefail
HERE=$(cd "$(dirname "$0")/.." && pwd)
source "$HERE/scripts/env.sh"
cd "$HERE"
cargo build --release
cargo test --release
case "$HERE" in
  /mnt/[a-z]/*) VENV=${VENV:-$HOME/venvs/ckpt-reshard-cli} ;;
  *) VENV=${VENV:-$HERE/.venv} ;;
esac
if [ ! -x "$VENV/bin/python" ]; then
  uv venv "$VENV" --python 3.12
  VIRTUAL_ENV="$VENV" uv pip install torch --index-url https://download.pytorch.org/whl/cpu
  VIRTUAL_ENV="$VENV" uv pip install safetensors transformers pytest numpy
fi
if ! "$VENV/bin/python" -c "import megatron.core, deepspeed" 2>/dev/null; then
  VIRTUAL_ENV="$VENV" uv pip install megatron-core
  DS_BUILD_OPS=0 VIRTUAL_ENV="$VENV" uv pip install deepspeed
fi
source "$VENV/bin/activate"
export TORCH_EXTENSIONS_DIR=${TORCH_EXTENSIONS_DIR:-$VENV/torch_extensions}  # DeepSpeed's CPU comm op
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
mkdir -p "$CKPT_FIXTURES"
[ -f "$CKPT_FIXTURES/reference/model.safetensors" ] || python scripts/make_fixtures.py --out "$CKPT_FIXTURES"
[ -f "$CKPT_FIXTURES/megatron/reference/logits.safetensors" ] || python scripts/make_megatron_fixtures.py --out "$CKPT_FIXTURES/megatron"
[ -f "$CKPT_FIXTURES/deepspeed/zero3/reference.safetensors" ] || python scripts/make_deepspeed_fixtures.py --out "$CKPT_FIXTURES/deepspeed"
python -m pytest -q -p no:cacheprovider tests/test_roundtrip.py tests/test_formats.py
# the real >4 GiB zip64 test writes ~8.5 GB: CKPT_BIG_TESTS=1 scripts/test_all.sh (or run scripts/test_zip64_big.py)
