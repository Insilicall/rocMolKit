# rocMolKit — Plano de Port (nvMolKit → ROCm/HIP)

> Port do [NVIDIA-Digital-Bio/nvMolKit](https://github.com/NVIDIA-Digital-Bio/nvMolKit) (Apache 2.0) para AMD GPUs via HIP/ROCm.
> Inspirado na abordagem do [mlxmolkit](https://github.com/guillaume-osmo/mlxmolkit) (Apple Silicon), mas usando **conversão CUDA→HIP automatizada** ao invés de reimplementação.

---

## 1. Princípios

1. **Imagem Docker mínima** (requisito explícito). Multi-stage build, runtime final sem SDK, sem headers, sem doc.
2. **Reaproveitar o máximo do código nvMolKit** via `hipify-perl`. Reescrita só onde for inevitável.
3. **Paridade numérica** com RDKit como gate de aceitação (mesmas tolerâncias usadas pelo nvMolKit nos testes upstream).
4. **ETKDG + MMFF94 primeiro**. Resto entra em fases isoladas.
5. **Apache 2.0** preservada, NOTICE com atribuição clara ao upstream.

---

## 2. Imagem Docker — estratégia "fica leve"

### Problema com a imagem do nvMolKit
A oficial usa `rocm/dev-ubuntu-22.04` (~10–15 GB) como runtime. Isso é desnecessário em produção: traz compilador HIP, headers, doc, profilers, debug symbols.

### Solução: multi-stage com runtime slim

```
┌─────────────────────────────────────────────────┐
│ STAGE 1: builder                                 │
│   FROM rocm/dev-ubuntu-22.04:6.2                 │
│   - hipcc, cmake, RDKit headers, boost-python    │
│   - compila libs e wheel Python                  │
│   - strip --strip-unneeded em todas as .so       │
└─────────────────────────────────────────────────┘
                        ↓ COPY artefatos
┌─────────────────────────────────────────────────┐
│ STAGE 2: runtime                                 │
│   FROM ubuntu:22.04                              │
│   - apt: libpython3.10, libstdc++6, libgomp1     │
│   - copia SELETIVO de /opt/rocm:                 │
│       libamdhip64.so.6                           │
│       libhsa-runtime64.so.1                      │
│       librocblas.so.4                            │
│       librocrand.so.1                            │
│       librocsolver.so.0  (se usado)              │
│       libhipblas.so.2                            │
│   - copia rocMolKit .so + wheel rdkit-pypi       │
│   - SEM /opt/rocm/bin, SEM /opt/rocm/include     │
│   - SEM /opt/rocm/share (doc, samples)           │
└─────────────────────────────────────────────────┘
```

**Alvo de tamanho:** imagem final **< 2 GB** (vs ~12 GB se fosse single-stage). Comparar com:
- `rocm/dev-ubuntu-22.04` ≈ 13 GB
- `rocm/rocm-terminal` ≈ 9 GB
- nosso runtime alvo ≈ 1.5–2 GB

### Variantes
- `rocmolkit:slim` — runtime mínimo (acima)
- `rocmolkit:devel` — para devs, com hipcc + headers
- `rocmolkit:cuda-compat` — opcional, build HIP rodando em NVIDIA (HIP é portátil)

### Trick adicional para encolher
- `--squash` no build final
- `dpkg --purge` agressivo de pacotes apt do estágio runtime
- não instalar `rdkit` via conda (puxa miniforge inteiro); usar `rdkit-pypi` wheel

---

## 3. Estratégia técnica de conversão

### O que é automático (HIPIFY)
| nvMolKit usa | rocMolKit usa | ferramenta |
|---|---|---|
| `cudaMalloc/cudaMemcpy/cudaStream_t` | `hipMalloc/hipMemcpy/hipStream_t` | `hipify-perl` |
| `__global__`, `__device__`, `__shared__` | idem (HIP herda sintaxe) | trivial |
| `cuBLAS` | `hipBLAS` (wrapper) ou `rocBLAS` | `hipify-perl` |
| `cuRAND` | `hipRAND` ou `rocRAND` | `hipify-perl` |
| `Thrust` | `rocThrust` (API ~idêntica) | recompile |
| `CUB` | `hipCUB` / `rocPRIM` | `hipify-perl` |

### O que dá trabalho manual
1. **Wavefront 64 vs warp 32**
   - Kernels com `__shfl_*` assumindo 32 threads precisam parametrização.
   - Adicionar `#define WARP_SIZE warpSize` (intrínseco HIP).
   - Reduções intra-warp viram intra-wavefront → menos blocks por SM.

2. **cooperative_groups** (nvMolKit usa em ETKDG batch optimizer)
   - HIP suporta `hip/hip_cooperative_groups.h` mas com subset menor.
   - Pode exigir reestruturar grid-sync.

3. **Atomics em float64**
   - AMD CDNA (MI100+) suporta. RDNA (gaming) **não**. Detectar em runtime.

4. **CMake**
   - `enable_language(CUDA)` → `find_package(hip REQUIRED)` + `set_source_files_properties(... LANGUAGE HIP)`
   - `CMAKE_CUDA_ARCHITECTURES` → `GPU_TARGETS` (ex: `gfx90a;gfx942;gfx1100`)

5. **boost-python bindings**
   - Funcionam, basta `CXX=hipcc` no build do módulo.
   - ABI de RDKit precisa bater (mesma stdlib).

---

## 4. Fases

| # | Fase | Entregável | Critério de aceitação | Estimativa |
|---|---|---|---|---|
| 0 | **Scaffold** | repo + LICENSE/NOTICE + CI esqueleto + Dockerfile multi-stage | `docker build` passa, imagem < 2 GB | 1–2 dias |
| 1 | **Hipify mecânico** | todos `.cu` → `.hip.cpp`, CMake compila | `cmake --build` sem erros (testes podem falhar) | 3–7 dias **(parte mecânica feita: 161 arquivos, 0 erros)** |
| 2 | **ETKDG funcional** | `EmbedMolecules` retorna conformers válidos | paridade com RDKit em SPICE-100 (RMSD < 0.1 Å vs nvMolKit) | 2–3 sem |
| 3 | **MMFF94 funcional** | `MMFFOptimizeMoleculesConfs` converge | energia final dentro de 1e-3 kcal/mol vs RDKit | 2–3 sem |
| 4 | **Tuning AMD** | benchmark vs RDKit CPU em RX 7900 XTX e/ou MI210 | ≥ 5× speedup em batch ≥ 100 mols | contínuo |
| 5 | **Fingerprints + Similaridade** | Morgan + Tanimoto GPU | match exato com RDKit (bits idênticos) | 2 sem |
| 6 | **Butina clustering** | divide-and-conquer em > 100k mols | resultados idênticos ao nvMolKit | 1–2 sem |
| 7 | **UFF, conformerRMSD, TFD** | resto da API | paridade RDKit | 2 sem cada |

---

## 5. Estrutura de diretórios proposta

```
rocMolKit/
├── LICENSE                    # Apache 2.0 (herdada)
├── NOTICE                     # atribuição nvMolKit + autores rocMolKit
├── README.md
├── PLAN.md                    # este arquivo
├── pyproject.toml             # build via scikit-build-core
├── CMakeLists.txt             # raiz
├── cmake/
│   ├── FindHIP.cmake
│   └── ROCmTargets.cmake      # mapping gfx targets
├── docker/
│   ├── Dockerfile.slim        # runtime mínimo
│   ├── Dockerfile.devel       # com SDK
│   └── runtime-libs.txt       # lista das .so a copiar
├── tools/
│   ├── hipify_all.sh          # roda hipify-perl recursivo
│   ├── numerical_diff.py      # compara saída vs RDKit/nvMolKit
│   └── strip_release.sh       # strip + dpkg cleanup
├── rocmolkit/                 # mirror do dir nvmolkit/
│   ├── embedMolecules.{cpp,hip,py}
│   ├── mmffOptimization.{cpp,hip,py}
│   ├── fingerprints.{cpp,hip,py}
│   ├── similarity.{cpp,hip,py}
│   ├── clustering.{cpp,hip,py}
│   ├── conformerRmsd.{cpp,hip,py}
│   └── ...
├── tests/
│   ├── test_etkdg_parity.py   # vs RDKit
│   ├── test_mmff_parity.py
│   └── data/spice_100.smi
├── benchmarks/
│   └── bench_etkdg_mmff.py    # mesmos 1000 SMILES do mlxmolkit
└── .github/workflows/
    ├── ci.yml                 # build + tests em runner ROCm
    └── docker.yml             # publica imagens slim/devel
```

---

## 6. Validação numérica (gate de qualidade)

Sem isso, port "compila mas é lixo". Plano:

1. **Dataset:** mesmos 1000 SMILES do SPICE-2.0.1 que o mlxmolkit usa (`bench_conformers.py` deles).
2. **Métrica ETKDG:** RMSD entre conformers gerados por rocMolKit vs nvMolKit (referência GPU) e vs RDKit (referência CPU). Critério: 95% das moléculas com RMSD < 0.5 Å após alinhamento.
3. **Métrica MMFF:** energia final |E_roc - E_rdkit| / |E_rdkit| < 1e-4.
4. **CI gate:** falha se métrica regredir.

---

## 7. Hardware alvo — definir com o usuário

| GPU | Arch | Status ROCm | Comentário |
|---|---|---|---|
| MI300X / MI300A | gfx942 | oficial | ideal, mas caro |
| MI250 / MI210 | gfx90a | oficial | sweet spot datacenter |
| RX 7900 XTX/XT | gfx1100 | oficial (desde 6.0) | melhor consumer |
| RX 6900 XT | gfx1030 | não-oficial | funciona com `HSA_OVERRIDE_GFX_VERSION=10.3.0` |
| Radeon VII | gfx906 | EOL no ROCm 6 | evitar |

**Decisão pendente do usuário:** qual GPU AMD você tem/vai usar? Define os `GPU_TARGETS` default do build e o runner do CI.

---

## 8. Riscos

| Risco | Probabilidade | Mitigação |
|---|---|---|
| **CUDA Graphs Conditional em `butina.cu`** (CONFIRMADO Fase 1) | 100% | Adiar Butina para Fase 6; reescrever loop conditional como dispatch CPU-side |
| `cudaCheckError` macro custom não-padrão (CONFIRMADO Fase 1) | 100% | Header `hip_compat.h` com macro equivalente — feito |
| nvMolKit usa CUTLASS / kernels CUDA-only avançados | média | rocPRIM como substituto; reescrever onde necessário |
| ETKDG cooperative-groups não traduz limpo | baixa | hipify converteu OK em `dist_geom_kernels_device` e `bfgs_hessian` (verificado) |
| Float atomics quebram em RDNA | alta | path alternativo com lock-based reduction |
| RDKit ABI incompatível entre conda e pip | média | pinar uma das duas, documentar |
| CI sem GPU AMD disponível | alta | runner self-hosted obrigatório, ou usar HIP-on-CUDA p/ smoke test |

---

## 9. Próximas ações (ordem)

1. ✅ Decidir nome — **rocMolKit** (confirmado)
2. ⏳ Usuário define GPU alvo (item 7)
3. Criar scaffold (Fase 0)
4. Validar build do Dockerfile slim
5. Hipify mecânico (Fase 1)
6. ETKDG (Fase 2)
7. MMFF94 (Fase 3)

---

## 10. O que NÃO vai entrar (escopo declarado)

- Suporte simultâneo NVIDIA via HIP-on-CUDA (possível, mas vai dispersar foco). Decidir depois da Fase 4.
- GUI ou notebook integrado.
- Reescrita em Python puro com PyTorch/ROCm (caminho que o mlxmolkit pegou). É mais lento e perde paridade com nvMolKit.
- Empacotamento conda. Só PyPI wheel inicialmente.
