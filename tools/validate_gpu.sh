#!/usr/bin/env bash
# tools/validate_gpu.sh — single-command GPU validation after a clean boot.
#
# Usage:
#     bash tools/validate_gpu.sh
#
# What it does:
#   1. Confirms no leaked VRAM / no stuck process on /dev/kfd.
#   2. Runs the full N x k benchmark sweep in the devel docker image.
#   3. Prints a Markdown table ready to paste into README.md.
#
# Pre-conditions: GPU must be in a clean state. If you ever Ctrl+C'd a
# rocMolKit call, the HIP runtime leaks ~2 GB VRAM and reports the GPU
# 100% busy even after the holding process dies. The fix is a reboot or
# `sudo modprobe -r amdgpu && sudo modprobe amdgpu`. This script
# refuses to run when it detects that state.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="${ROCMOLKIT_IMAGE:-ghcr.io/insilicall/rocmolkit:devel}"
DATASET="${ROCMOLKIT_DATASET:-tests/data/druglike_100.smi}"

# --- 1. Pre-flight: GPU must be clean ----------------------------------------
DGPU_CARD=""
for c in /sys/class/drm/card*/device; do
    [ -f "$c/mem_info_vram_total" ] || continue
    total=$(cat "$c/mem_info_vram_total")
    # Pick the first card with > 4 GB VRAM as the dGPU.
    if [ "$total" -gt $((4 * 1024 * 1024 * 1024)) ]; then
        DGPU_CARD="$c"
        break
    fi
done

if [ -z "$DGPU_CARD" ]; then
    echo "ERROR: no discrete GPU with >4 GB VRAM found in /sys/class/drm." >&2
    exit 1
fi

vram_used=$(cat "$DGPU_CARD/mem_info_vram_used")
vram_used_mb=$((vram_used / 1024 / 1024))
busy=$(cat "$DGPU_CARD/gpu_busy_percent" 2>/dev/null || echo 0)
kfd_holders=$(lsof /dev/kfd 2>/dev/null | grep -v COMMAND | wc -l || echo 0)

echo "Pre-flight on $DGPU_CARD:"
echo "  vram_used  = ${vram_used_mb} MB"
echo "  busy%      = ${busy}"
echo "  /dev/kfd   = ${kfd_holders} holders"

# Allow up to 200 MB of incidental VRAM (compositor / desktop occasionally
# touches the dGPU). Bail above that with no holders — that is the leak
# pattern documented in ISSUES.md.
if [ "$kfd_holders" -eq 0 ] && [ "$vram_used_mb" -gt 200 ]; then
    cat >&2 <<EOF

ERROR: GPU is in the leaked state (>200 MB VRAM held with no /dev/kfd
holders). This breaks back-to-back HIP calls; the next benchmark will
hang. Recover with:

    sudo modprobe -r amdgpu && sudo modprobe amdgpu

or reboot, then re-run this script.
EOF
    exit 2
fi

# --- 2. GID resolution (host render/video groups in the container) -----------
RENDER_GID=$(getent group render | cut -d: -f3)
VIDEO_GID=$(getent group video | cut -d: -f3)
if [ -z "$RENDER_GID" ] || [ -z "$VIDEO_GID" ]; then
    echo "ERROR: cannot resolve host render/video groups." >&2
    exit 1
fi

echo
echo "Image:    $IMAGE"
echo "Dataset:  $DATASET"
echo "Repo:     $REPO_ROOT"
echo "GIDs:     render=$RENDER_GID video=$VIDEO_GID"
echo

# --- 3. Run the sweep --------------------------------------------------------
exec docker run --rm \
    --device=/dev/kfd --device=/dev/dri \
    --group-add "$RENDER_GID" --group-add "$VIDEO_GID" \
    --security-opt seccomp=unconfined \
    -e LD_LIBRARY_PATH=/usr/local/lib:/opt/rdkit/lib:/opt/rocm/lib:/opt/rocm-7.2.3/lib/llvm/lib \
    -v "$REPO_ROOT:/work:ro" \
    -w /work \
    --entrypoint /bin/bash \
    "$IMAGE" \
    -c "python3 tools/benchmark.py $DATASET --sweep --timeout 900 ${ROCMOLKIT_BENCH_EXTRA:-}"
