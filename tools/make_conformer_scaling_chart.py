"""Render docs/assets/rocmolkit_conformer_scaling.png — the two-panel scaling figure.

Left:  throughput vs batch size (GPU vs single-core RDKit), log-log.
Right: speedup vs batch size — the GPU advantage *grows then holds* at scale.

InsilicAll chart styling. Install it first (private repo):

    pip install matplotlib git+https://github.com/insilicall/insilicall-charts.git@v1.0.0

Then from the repo root:

    python3 tools/make_conformer_scaling_chart.py

Numbers come from tools/conformer scaling runs (ETKDG, 1 conformer/molecule),
GPU vs single-core RDKit on the same host.
"""

import pathlib

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from insilicall_charts import apply_style, palette, add_caption

ROOT = pathlib.Path(__file__).resolve().parent.parent
LOGO = ROOT / "docs" / "assets" / "insilicall-logo.png"
OUT = ROOT / "docs" / "assets" / "rocmolkit_conformer_scaling.png"

apply_style("standard")

# ETKDG, 1 conformer/molecule, drug-like set. GPU vs single-core RDKit.
N   = [100, 500, 1000, 3000, 10000]
gpu = [121, 111, 3377, 3467, 3438]
cpu = [303, 299,  298,  301,   301]
sp  = [g / c for g, c in zip(gpu, cpu)]

fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 5.4))

# ---- Left: throughput vs batch size --------------------------------------
axL.plot(N, gpu, "-o", color=palette["brand_secondary"], lw=2.6, ms=8,
         zorder=4, label="rocMolKit (RX 9060 XT GPU)")
axL.plot(N, cpu, "-o", color=palette["ink_muted"], lw=2.2, ms=7,
         zorder=3, label="RDKit (single-core CPU)")
axL.set_xscale("log")
axL.set_yscale("log")
axL.set_xlabel("Batch size (molecules)")
axL.set_ylabel("Throughput (molecules / second)")
axL.set_title("Throughput scales with batch size", loc="left", fontweight="bold")
axL.grid(True, which="both", color=palette["divider"], lw=0.6, zorder=0)
for side in ("top", "right"):
    axL.spines[side].set_visible(False)
axL.legend(frameon=False, loc="center right")
# annotate the saturated plateau
axL.annotate("saturates ~3,400 mol/s", xy=(3000, 3467), xytext=(900, 1500),
             color=palette["brand_secondary"], fontweight="bold", fontsize=10,
             arrowprops=dict(arrowstyle="->", color=palette["brand_secondary"], lw=1.4))

# ---- Right: speedup vs batch size ----------------------------------------
colors = [palette["accent_warm"] if s < 1 else palette["brand_secondary"] for s in sp]
bars = axR.bar([str(n) for n in N], sp, color=colors, width=0.62, zorder=3)
axR.axhline(1, color=palette["ink_muted"], lw=1.2, ls="--", zorder=2)
axR.text(0.02, 1.0, "  break-even (1×)", transform=axR.get_yaxis_transform(),
         va="bottom", ha="left", color=palette["ink_muted"], fontsize=9)
axR.set_xlabel("Batch size (molecules)")
axR.set_ylabel("Speedup vs single-core RDKit")
axR.set_title("GPU advantage grows, then holds at scale", loc="left", fontweight="bold")
axR.set_ylim(0, 13)
for side in ("top", "right", "left"):
    axR.spines[side].set_visible(False)
axR.tick_params(axis="y", length=0)
axR.grid(axis="y", color=palette["divider"], lw=0.6, zorder=0)
for b, s in zip(bars, sp):
    txt = f"{s:.1f}×" if s >= 1 else f"{s:.2f}×"
    axR.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.25, txt,
             ha="center", va="bottom", fontweight="bold",
             color=palette["ink_strong"], fontsize=10)

fig.suptitle("rocMolKit — GPU-accelerated conformer generation on AMD",
             x=0.012, ha="left", fontweight="bold", fontsize=16)

add_caption(fig, "ETKDG, 1 conformer/molecule · AMD Radeon RX 9060 XT (RDNA4) + "
                 "Ryzen 5 7600, ROCm 7.2.3 · drug-like molecules · "
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
