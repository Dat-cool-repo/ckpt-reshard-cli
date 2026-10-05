# Source me (Linux/WSL/macOS): common env for building and testing ckpt-reshard-cli.
# Every variable can be overridden by exporting it before sourcing.
#
#   CKPT_DATA_DIR      generated fixtures, benchmark data, big-test scratch  (default: <repo>/data, git-ignored)
#   CKPT_FIXTURES      end-to-end pytest fixtures                            (default: $CKPT_DATA_DIR/fixtures)
#   CARGO_TARGET_DIR   build dir (default: <repo>/target; on a WSL /mnt/<drive> checkout,
#                      ~/target/ckpt-reshard-cli because building on drvfs is slow)
#   CKPT_BIN           the release binary used by the Python tests and scripts
[ -f "$HOME/.cargo/env" ] && source "$HOME/.cargo/env"
CKPT_REPO=$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)
export CKPT_REPO
case "$CKPT_REPO" in
  /mnt/[a-z]/*) _ckpt_default_target=$HOME/target/ckpt-reshard-cli ;;
  *) _ckpt_default_target=$CKPT_REPO/target ;;
esac
export CARGO_TARGET_DIR=${CARGO_TARGET_DIR:-$_ckpt_default_target}
unset _ckpt_default_target
export CKPT_DATA_DIR=${CKPT_DATA_DIR:-$CKPT_REPO/data}
export CKPT_FIXTURES=${CKPT_FIXTURES:-$CKPT_DATA_DIR/fixtures}
export CKPT_BIN=${CKPT_BIN:-$CARGO_TARGET_DIR/release/ckpt}
