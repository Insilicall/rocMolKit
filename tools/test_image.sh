#!/usr/bin/env bash
# tools/test_image.sh — pull a published rocMolKit image and run the GPU test
# suite against it on the local AMD GPU. Validates the *published* bindings
# (installed in the image), using this checkout only for the test files + data.
#
# Usage:
#     bash tools/test_image.sh                 # default: :devel
#     bash tools/test_image.sh v0.4.1-devel    # a specific tag
#     ROCMOLKIT_NO_PULL=1 bash tools/test_image.sh   # skip pull, use local image
#
# Requires: Docker + an AMD GPU (/dev/kfd, /dev/dri) + ROCm kernel driver.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TAG="${1:-devel}"
IMAGE="ghcr.io/insilicall/rocmolkit:${TAG}"

# Host render/video GIDs so the container can reach the GPU devices.
RENDER_GID="$(getent group render | cut -d: -f3 || true)"
VIDEO_GID="$(getent group video | cut -d: -f3 || true)"

GPU_FLAGS=(--device=/dev/kfd --device=/dev/dri --security-opt seccomp=unconfined)
[ -n "$RENDER_GID" ] && GPU_FLAGS+=(--group-add "$RENDER_GID")
[ -n "$VIDEO_GID" ]  && GPU_FLAGS+=(--group-add "$VIDEO_GID")

if [ -z "${ROCMOLKIT_NO_PULL:-}" ]; then
    echo ">> docker pull ${IMAGE}"
    docker pull "${IMAGE}"
fi

echo ">> pytest tests/ --rocm  (against the image's installed bindings)"
# -w /tmp keeps the mounted source tree (/work/rocmolkit) off sys.path, so
# `import rocmolkit` resolves to the version installed in the image — we are
# testing the published artifact, not the local checkout. The tests still read
# their data from /work/tests/data via __file__.
exec docker run --rm \
    "${GPU_FLAGS[@]}" \
    -e HIP_VISIBLE_DEVICES=0 \
    -v "${REPO_ROOT}:/work" -w /tmp \
    "${IMAGE}" \
    python3 -m pytest /work/tests --rocm -v
