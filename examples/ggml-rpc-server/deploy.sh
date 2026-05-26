#!/usr/bin/env bash
# Run LOCALLY. Rsync the pyspike repo to owner@100.103.189.26:~/Desktop/pyspike-rpc/
# then invoke bootstrap.sh on the remote.
#
# Usage:
#   bash examples/ggml-rpc-server/deploy.sh           # rsync + bootstrap
#   bash examples/ggml-rpc-server/deploy.sh --no-bootstrap   # rsync only
#
# Requires: ssh key already registered (use sshpass if not — see README).

set -euo pipefail

HOST="${HOST:-owner@100.103.189.26}"
DEST="${DEST:-/home/owner/Desktop/pyspike-rpc}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
DO_BOOTSTRAP=1
[ "${1:-}" = "--no-bootstrap" ] && DO_BOOTSTRAP=0

log() { printf '\n\033[1;36m[deploy]\033[0m %s\n' "$*"; }

log "From   : $REPO"
log "To     : $HOST:$DEST"
log "Bootstrap : $([ $DO_BOOTSTRAP = 1 ] && echo yes || echo no)"

# Ensure target exists
ssh "$HOST" "mkdir -p '$DEST'"

# rsync the working tree. Exclude heavyweight/regen-able stuff.
# Include .git so setuptools_scm has version info.
log "rsync (this may take a few minutes — repo is large)..."
rsync -az --info=progress2 --delete \
    --exclude '.venv/' \
    --exclude 'build/' \
    --exclude 'dist/' \
    --exclude '.pytest_cache/' \
    --exclude '__pycache__/' \
    --exclude '*.pyc' \
    --exclude '.benchmarks/' \
    --exclude '.fresh/' \
    --exclude '.claude/worktrees/' \
    --exclude 'dump_trace_iss/' \
    --exclude '.planning/codebase/' \
    --exclude 'tests/data/' \
    "$REPO/" "$HOST:$DEST/"

log "rsync done. Disk usage on remote:"
ssh "$HOST" "du -sh '$DEST' | awk '{print \"  size: \" \$1}'"

if [ $DO_BOOTSTRAP = 1 ]; then
    log "Running bootstrap.sh on remote (will prompt for sudo password)..."
    ssh -t "$HOST" "bash '$DEST/examples/ggml-rpc-server/bootstrap.sh'"
fi
