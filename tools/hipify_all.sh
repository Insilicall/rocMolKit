#!/usr/bin/env bash
# Roda hipify-perl em toda árvore CUDA do nvMolKit upstream e copia para rocmolkit/.
# Uso:
#   ./tools/hipify_all.sh                  # usa ./upstream
#   ./tools/hipify_all.sh /path/nvMolKit
#
# Convenção de saída:
#   foo.cu  -> rocmolkit/<dir>/foo.hip.cpp
#   foo.cuh -> rocmolkit/<dir>/foo.hip.h
#   foo.cpp -> rocmolkit/<dir>/foo.cpp     (cópia direta, hipify só se referenciar CUDA)
#
# Diretórios processados (relativos ao upstream):
#   nvmolkit/  src/  rdkit_extensions/  tests/  benchmarks/

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
UPSTREAM="${1:-${ROOT}/upstream}"
DEST="${ROOT}/rocmolkit"
LOG="${ROOT}/build/hipify.log"

# hipify-perl: prefere o local do tools/bin (standalone), senão PATH
HIPIFY="${ROOT}/tools/bin/hipify-perl"
if [[ ! -x "${HIPIFY}" ]]; then
    HIPIFY="$(command -v hipify-perl || true)"
fi
if [[ -z "${HIPIFY}" ]]; then
    echo "erro: hipify-perl não encontrado em tools/bin/ nem no PATH" >&2
    exit 2
fi

if [[ ! -d "${UPSTREAM}/src" || ! -d "${UPSTREAM}/nvmolkit" ]]; then
    echo "erro: ${UPSTREAM} não parece ser um clone do nvMolKit" >&2
    exit 3
fi

mkdir -p "${DEST}" "$(dirname "${LOG}")"
: > "${LOG}"

DIRS=(nvmolkit src rdkit_extensions tests benchmarks)

count_hip=0
count_copy=0
count_err=0

for d in "${DIRS[@]}"; do
    src_root="${UPSTREAM}/${d}"
    [[ -d "${src_root}" ]] || continue

    while IFS= read -r -d '' src; do
        rel="${src#${UPSTREAM}/}"
        case "${src}" in
            *.cu)
                out="${DEST}/${rel%.cu}.hip.cpp"
                ;;
            *.cuh)
                out="${DEST}/${rel%.cuh}.hip.h"
                ;;
            *.cpp|*.c|*.cc)
                out="${DEST}/${rel}"
                ;;
            *.h|*.hpp|*.inc)
                out="${DEST}/${rel}"
                ;;
            *)
                continue
                ;;
        esac

        mkdir -p "$(dirname "${out}")"

        # Hipify se contiver CUDA-isms; senão cópia direta.
        if grep -q -E 'cuda[A-Z_]|__global__|__device__|__shared__|__host__|cuBLAS|cuRAND|cuSOLVER|<thrust/|<cub/|cooperative_groups' "${src}" 2>/dev/null; then
            if "${HIPIFY}" --quiet-warnings "${src}" > "${out}" 2>>"${LOG}"; then
                count_hip=$((count_hip + 1))
            else
                echo "FAIL hipify: ${rel}" | tee -a "${LOG}"
                count_err=$((count_err + 1))
            fi
        else
            cp "${src}" "${out}"
            count_copy=$((count_copy + 1))
        fi
    done < <(find "${src_root}" -type f \( \
                -name '*.cu' -o -name '*.cuh' \
                -o -name '*.cpp' -o -name '*.c' -o -name '*.cc' \
                -o -name '*.h' -o -name '*.hpp' -o -name '*.inc' \
             \) -print0)
done

echo ""
echo "===================="
echo "hipify-perl: ${count_hip} arquivos convertidos"
echo "cópia direta: ${count_copy} arquivos"
echo "erros:        ${count_err}"
echo "log:          ${LOG}"
echo "destino:      ${DEST}"
