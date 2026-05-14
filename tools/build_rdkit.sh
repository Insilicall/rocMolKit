#!/usr/bin/env bash
# Build RDKit from source — minimal config that covers what nvMolKit needs:
# GraphMol, DistGeom, DistGeomHelpers, ForceField/MMFF, FileParsers,
# SmilesParse, DataStructs, ChemTransforms, MolStandardize.
#
# No Python bindings here (rdkit-pypi handles Python at runtime). No tests,
# no examples, no draw, no DB connectors. Result lives in /opt/rdkit.
#
# Build time: ~10-15 min on 4 cores. Docker cache makes this run once.
#
# Usage: ./tools/build_rdkit.sh [version-tag]
#   default version: Release_2024_09_6

set -euo pipefail

RDKIT_VERSION="${1:-Release_2024_09_6}"
PREFIX="${RDKIT_PREFIX:-/opt/rdkit}"
JOBS="${JOBS:-$(nproc)}"
SRC="${RDKIT_SRC:-/tmp/rdkit-src}"

echo "=== Building RDKit ${RDKIT_VERSION} (jobs=${JOBS}, prefix=${PREFIX}) ==="

if [[ ! -d "${SRC}/.git" ]]; then
    git clone --depth 1 --branch "${RDKIT_VERSION}" \
        https://github.com/rdkit/rdkit.git "${SRC}"
fi

mkdir -p "${SRC}/build"
cd "${SRC}/build"

cmake .. -GNinja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="${PREFIX}" \
    -DCMAKE_CXX_STANDARD=20 \
    -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
    -DRDK_INSTALL_INTREE=OFF \
    -DRDK_INSTALL_STATIC_LIBS=OFF \
    -DRDK_BUILD_PYTHON_WRAPPERS=OFF \
    -DRDK_BUILD_CPP_TESTS=OFF \
    -DRDK_BUILD_LONG_RUNNING_TESTS=OFF \
    -DRDK_BUILD_INCHI_SUPPORT=OFF \
    -DRDK_BUILD_AVALON_SUPPORT=OFF \
    -DRDK_BUILD_FREESASA_SUPPORT=OFF \
    -DRDK_BUILD_COORDGEN_SUPPORT=ON \
    -DRDK_BUILD_DESCRIPTORS3D=ON \
    -DRDK_BUILD_THREADSAFE_SSS=ON \
    -DRDK_USE_BOOST_SERIALIZATION=ON \
    -DRDK_BUILD_CAIRO_SUPPORT=OFF \
    -DRDK_BUILD_MOLINTERCHANGE_SUPPORT=OFF \
    -DRDK_BUILD_RPATH_SUPPORT=ON \
    -DRDK_BUILD_SWIG_WRAPPERS=OFF \
    -DRDK_BUILD_PGSQL=OFF \
    -DRDK_TEST_MMFF_COMPLIANCE=OFF \
    -DRDK_BUILD_MAEPARSER_SUPPORT=OFF \
    -DRDK_BUILD_FREETYPE_SUPPORT=OFF \
    -DRDK_BUILD_YAEHMOP_SUPPORT=OFF \
    -DRDK_BUILD_XYZ2MOL_SUPPORT=OFF \
    -DRDK_BUILD_QED=ON \
    -DBoost_NO_BOOST_CMAKE=ON

cmake --build . --parallel "${JOBS}"
cmake --install .

echo "=== RDKit ${RDKIT_VERSION} installed to ${PREFIX} ==="
echo "    headers: ${PREFIX}/include/rdkit"
echo "    libs:    ${PREFIX}/lib/libRDKit*.so"

# Sanity check: confirm headers nvMolKit relies on are present.
for h in \
    GraphMol/RDKitBase.h \
    GraphMol/DistGeomHelpers/Embedder.h \
    DistGeom/BoundsMatrix.h \
    ForceField/MMFF/Builder.h ; do
    if [[ ! -f "${PREFIX}/include/rdkit/${h}" ]]; then
        echo "FAIL: missing ${PREFIX}/include/rdkit/${h}" >&2
        exit 1
    fi
done
echo "=== sanity check OK ==="
