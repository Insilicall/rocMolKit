#!/usr/bin/env bash
# Strip pesado para reduzir o tamanho da imagem runtime.
# Roda no diretório de install antes do COPY final no Dockerfile.

set -euo pipefail

ROOT="${1:-/install}"

echo "strip: processando ${ROOT}"

# Strip todas as .so (mantém símbolos dinâmicos exportados)
find "${ROOT}" -type f -name '*.so*' -exec strip --strip-unneeded {} + 2>/dev/null || true

# Remove .a estáticos (não vão para runtime)
find "${ROOT}" -type f -name '*.a' -delete 2>/dev/null || true

# Remove pkgconfig e cmake (não usados em runtime)
rm -rf "${ROOT}/lib/pkgconfig" "${ROOT}/lib/cmake" 2>/dev/null || true

# Remove __pycache__
find "${ROOT}" -type d -name '__pycache__' -exec rm -rf {} + 2>/dev/null || true

du -sh "${ROOT}"
