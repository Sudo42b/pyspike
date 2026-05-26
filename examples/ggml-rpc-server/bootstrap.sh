#!/usr/bin/env bash
# Run on the remote host (owner@100.103.189.26) after deploy.sh rsyncs the
# pyspike repo to ~/Desktop/pyspike-rpc/. Installs apt sysdeps, uv, and
# builds + installs pyspike in-tree.
#
# Usage:
#   bash ~/Desktop/pyspike-rpc/examples/ggml-rpc-server/bootstrap.sh
#
# Assumes /opt/riscv/bin/riscv64-unknown-elf-gcc and existing gtx_spike are
# already in place (audited 2026-05-26). Sudo password is prompted once.

set -euo pipefail

# Resolve repo root = parent of examples/ggml-rpc-server/
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
RISCV="${RISCV:-/opt/riscv}"

log() { printf '\n\033[1;36m[bootstrap]\033[0m %s\n' "$*"; }

# Non-interactive sudo: if BOOTSTRAP_SUDO_PASS is set, feed it; else fall
# through to normal interactive sudo (terminal must have a TTY).
sudo_() {
    if [ -n "${BOOTSTRAP_SUDO_PASS:-}" ]; then
        echo "$BOOTSTRAP_SUDO_PASS" | sudo -S -p '' "$@"
    else
        sudo "$@"
    fi
}

log "Repo:  $REPO"
log "RISCV: $RISCV"
[ -x "$RISCV/bin/riscv64-unknown-elf-gcc" ] || {
    echo "ERROR: $RISCV/bin/riscv64-unknown-elf-gcc not found" >&2; exit 1; }

# ---------------------------------------------------------------------------
log "Step 1/4 — apt sysdeps (sudo)"
# device-tree-compiler is required by spike fdt; python3-venv for venv; rest
# for general C build (already present on this host but harmless to verify).
sudo_ apt-get update -qq
sudo_ apt-get install -y --no-install-recommends \
    device-tree-compiler \
    python3-venv \
    python3-pip \
    build-essential \
    pkg-config

# ---------------------------------------------------------------------------
log "Step 2/4 — uv (user-level)"
if ! command -v uv >/dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi
# Make uv visible to this shell
export PATH="$HOME/.local/bin:$PATH"
uv --version

# ---------------------------------------------------------------------------
log "Step 3/4 — build & install pyspike (in-tree, isolated venv)"
cd "$REPO"

# setup.py uses setuptools_scm which needs git or SETUPTOOLS_SCM_PRETEND_VERSION.
if [ ! -d .git ]; then
    export SETUPTOOLS_SCM_PRETEND_VERSION="0.0.0+rpc.deploy"
fi

# Isolate everything in a project-local venv so we don't poison /usr.
if [ ! -d .venv ]; then
    uv venv .venv --python 3.12
fi
# shellcheck source=/dev/null
. .venv/bin/activate

# The lockfile pulls torch+CUDA (~2.7 GB). We only need numpy at runtime for
# the RPC server. Install editable without resolving the lock — uv will then
# pick up the minimal direct deps declared in pyproject.toml.
RISCV="$RISCV" UV_LINK_MODE=copy uv pip install -e . --no-deps
# Then install the small runtime deps we actually need.
uv pip install numpy

# ---------------------------------------------------------------------------
log "Step 4/4 — sanity"
python3 -c "
import riscv, importlib
print('riscv :', riscv.__file__)
m = importlib.import_module('riscv.gtx')
print('gtx   :', m.__file__)
print('OK')
"
# Verify the pyspike CLI launcher resolves
command -v pyspike >/dev/null && pyspike --help 2>&1 | head -3 || true

cat <<EOF

==============================================================
 pyspike installed.  Launch the ggml RPC server with:

   cd $REPO
   python3 examples/ggml-rpc-server/pyspike_rpc_server.py \\
       --host 0.0.0.0 --port 50052

 Connect a client (e.g. llama.cpp built with GGML_RPC=ON):

   llama-cli -m model.gguf --rpc 100.103.189.26:50052 ...
==============================================================
EOF
