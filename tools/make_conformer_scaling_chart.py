"""Render docs/assets/rocmolkit_conformer_scaling.png — the two-panel scaling figure.

Left:  ETKDG throughput vs batch size (GPU vs single-core / 12-thread RDKit), log-log.
Right: GPU speedup vs batch size — the advantage grows with batch, then holds at scale.

All points are at 100% success and RDKit-validated (generated conformers fall in
the same MMFF energy basins as RDKit). Drug-like molecules, k=4.

InsilicAll chart styling. Install it first (private repo):

    pip install matplotlib git+https://github.com/insilicall/insilicall-charts.git@v1.0.0

Then from the repo root:

    python3 tools/make_conformer_scaling_chart.py
"""

import pathlib

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from insilicall_charts import apply_style, palette, add_caption

ROOT = pathlib.Path(__file__).resolve().parent.parent
LOGO = ROOT / "docs" / "assets" / "insilicall-logo.png"
OUT = ROOT / "docs" / "assets" / "rocmolkit_conformer_scaling.png"

apply_style("standard")

# ETKDG generation, k=4, drug-like, 100% success, RX 9060 XT. Measured (best of N).
N      = [500, 1000, 5000, 10000]
gpu    = [1946, 2460, 4552, 4231]
rd_1c  = 395   # RDKit single-core, flat in batch size
rd_12t = 865   # RDKit 12-thread, flat in batch size
sp_12t = [g / rd_12t for g in gpu]

fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 5.4))

# ---- Left: throughput vs batch size --------------------------------------
axL.plot(N, gpu, "-o", color=palette["brand_secondary"], lw=2.6, ms=8, zorder=4,
         label="rocMolKit (RX 9060 XT GPU)")
axL.plot([N[0], N[-1]], [rd_12t, rd_12t], "--", color=palette["ink_muted"], lw=2.0,
         zorder=3, label="RDKit (12 threads)")
axL.plot([N[0], N[-1]], [rd_1c, rd_1c], ":", color=palette["ink_muted"], lw=1.8,
         zorder=2, label="RDKit (single core)")
axL.set_xscale("log")
axL.set_yscale("log")
axL.set_xlabel("Batch size (molecules)")
axL.set_ylabel("Throughput (conformers / second)")
axL.set_title("Throughput scales with batch size", loc="left", fontweight="bold")
axL.grid(True, which="both", color=palette["divider"], lw=0.6, zorder=0)
for side in ("top", "right"):
    axL.spines[side].set_visible(False)
axL.legend(frameon=False, loc="center right")
axL.annotate("saturates ~4,500 conf/s", xy=(5000, 4552), xytext=(700, 5200),
             color=palette["brand_secondary"], fontweight="bold", fontsize=10,
             arrowprops=dict(arrowstyle="->", color=palette["brand_secondary"], lw=1.4))

# ---- Right: speedup vs batch size ----------------------------------------
bars = axR.bar([str(n) for n in N], sp_12t, color=palette["brand_secondary"],
               width=0.62, zorder=3)
axR.axhline(1, color=palette["ink_muted"], lw=1.2, ls="--", zorder=2)
axR.set_xlabel("Batch size (molecules)")
axR.set_ylabel("Speedup vs 12-thread RDKit")
axR.set_title("GPU advantage grows, then holds at scale", loc="left", fontweight="bold")
axR.set_ylim(0, 6.5)
for side in ("top", "right", "left"):
    axR.spines[side].set_visible(False)
axR.tick_params(axis="y", length=0)
axR.grid(axis="y", color=palette["divider"], lw=0.6, zorder=0)
for b, s in zip(bars, sp_12t):
    axR.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.12, f"{s:.1f}×",
             ha="center", va="bottom", fontweight="bold", color=palette["ink_strong"], fontsize=10)

fig.suptitle("rocMolKit — GPU-accelerated conformer generation on AMD (100% success)",
             x=0.012, ha="left", fontweight="bold", fontsize=16)

add_caption(fig, "ETKDG, k=4, drug-like molecules · AMD Radeon RX 9060 XT (RDNA4) + "
                 "Ryzen 5 7600, ROCm 7.2.3 · 100% success, conformers match RDKit energy basins · "
                 "github.com/Insilicall/rocMolKit")

fig.tight_layout(rect=[0, 0.04, 1, 0.93])

logo = mpimg.imread(str(LOGO))
lh, lw = logo.shape[0], logo.shape[1]
fw, fh = fig.get_size_inches()
w = 0.15
h = w * (lh / lw) * (fw / fh)
ax_logo = fig.add_axes([0.985 - w, 0.99 - h, w, h], anchor="NE", zorder=10)
ax_logo.imshow(logo)
ax_logo.axis("off")

fig.savefig(OUT, dpi=200, bbox_inches="tight")
print(f"wrote {OUT}")
