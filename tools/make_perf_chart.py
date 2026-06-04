"""Render docs/assets/rocmolkit_speedup.png — the GPU-vs-RDKit speedup chart.

Uses InsilicAll's chart styling. Install it first (private repo):

    pip install matplotlib git+https://github.com/insilicall/insilicall-charts.git@v1.0.0

Then from the repo root:

    python3 tools/make_perf_chart.py

Numbers come from tools/bench_features.py (features) and tools/etkdg_bench.py /
mmff_fresh.py (conformers), GPU vs 12-thread RDKit on the same host.
"""

import pathlib

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from insilicall_charts import apply_style, palette, FIGSIZE, add_caption

ROOT = pathlib.Path(__file__).resolve().parent.parent
LOGO = ROOT / "docs" / "assets" / "insilicall-logo.png"
OUT = ROOT / "docs" / "assets" / "rocmolkit_speedup.png"

apply_style("standard")

# (operation, speedup vs 12-thread RDKit)
data = [
    ("Tanimoto similarity", 658),
    ("TFD", 137),
    ("Butina clustering", 48),
    ("Substructure", 12),
    ("Morgan fingerprints", 7),
    ("ETKDG generation", 6),
    ("MMFF94 optimization", 5),
]
data.sort(key=lambda t: t[1])
labels = [d[0] for d in data]
vals = [d[1] for d in data]

fig, ax = plt.subplots(figsize=FIGSIZE["social_16x9"])
colors = [palette["accent_warm"] if v == max(vals) else palette["brand_secondary"] for v in vals]
bars = ax.barh(labels, vals, color=colors, height=0.62, zorder=3)

ax.set_xscale("log")
ax.set_xlim(1, 1100)
ax.set_xticks([1, 10, 100, 1000])
ax.set_xticklabels(["1×", "10×", "100×", "1000×"])
ax.axvline(1, color=palette["ink_muted"], lw=1, ls="--", zorder=2)
ax.set_xlabel("Speedup vs 12-thread RDKit (log scale)")
ax.set_title("rocMolKit — GPU acceleration over multi-threaded RDKit", loc="left", fontweight="bold")

for b, v in zip(bars, vals):
    ax.text(v * 1.08, b.get_y() + b.get_height() / 2, f"{v}×",
            va="center", ha="left", fontweight="bold", color=palette["ink_strong"])

ax.grid(axis="x", which="both", color=palette["divider"], lw=0.6, zorder=0)
for side in ("top", "right", "left"):
    ax.spines[side].set_visible(False)
ax.tick_params(axis="y", length=0)

add_caption(fig, "AMD Radeon RX 9060 XT (RDNA4) + Ryzen 5 7600, ROCm 7.2.3 · "
                 "drug-like molecules · validated bit/precision-exact vs RDKit · github.com/Insilicall/rocMolKit")

logo = mpimg.imread(str(LOGO))
lh, lw = logo.shape[0], logo.shape[1]
fw, fh = fig.get_size_inches()
w = 0.19
h = w * (lh / lw) * (fw / fh)
ax_logo = fig.add_axes([0.965 - w, 0.98 - h, w, h], anchor="NE", zorder=10)
ax_logo.imshow(logo)
ax_logo.axis("off")

fig.savefig(OUT, dpi=200, bbox_inches="tight")
print(f"wrote {OUT}")
