# Bisect repros for the gfx1200 state-leak SIGSEGV

These two C++ programs were written to localise the segfault discussed in
`ISSUES.md` ("KEY FINDING — bisect localized the bug"). They are kept in
tree because each new ROCm release should re-run them: the day
`embed_pure_cpp` stops crashing is the day we can drop the
`rocmolkit.safe` subprocess workaround.

## `asyncvec_stress.cpp`

Pure HIP allocator pattern that mirrors `AsyncDeviceVector` —
`hipMallocAsync` / `hipFreeAsync` cycles in a tight loop, no RDKit, no
boost-python.

Result on ROCm 7.2.3 + gfx1200: **20 × 1000 iterations, zero crashes.**

→ Conclusion: the bug is **not** in our HIP allocation lifecycle.

## `embed_pure_cpp.cpp`

Drives `nvMolKit::embedMolecules` directly via the C++ API, with an
`RDKit::ROMol` built through the RDKit C++ headers. **No boost-python,
no Python interpreter.**

Result on ROCm 7.2.3 + gfx1200: **segfaults on every invocation, even on
`CCO`.**

→ Conclusion: the bug lives **inside** `nvMolKit::embedMolecules` or one
of its kernels. Python / boost-python are not implicated.

## How to build

These are not part of the CMake build (intentional — they pull in HIP
and RDKit headers and are not runtime dependencies). Compile by hand
inside the devel image:

```bash
docker run --rm -v $PWD:/work -w /work rocmolkit:devel-local bash -c '
  /opt/rocm/llvm/bin/clang++ -x hip -std=c++20 \
    -isystem /opt/rocm/include -isystem /opt/rdkit/include \
    --rocm-path=/opt/rocm \
    --offload-arch=gfx1200 \
    tests/repro/asyncvec_stress.cpp -o /tmp/asyncvec_stress \
    -L/opt/rocm/lib -lamdhip64
  /tmp/asyncvec_stress
'
```

Replace `asyncvec_stress.cpp` with `embed_pure_cpp.cpp` (and link
`-L/opt/rdkit/lib -lRDKit*`) for the second repro.
