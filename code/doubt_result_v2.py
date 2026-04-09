import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── Data ──────────────────────────────────────────────────────────────────────
labels = ["Web", "Prog.", "IoT", "Fin.", "App."]

data = {
    "DeepSeek": {
        "origin": [26.1, 21.6, 43.2, 44.8, 26.5],
        "optim": [5.2, 7.2, 13.5, 8.2, 5.9],
    },
    "GPT-4o": {
        "origin": [12.9, 15.9, 50.0, 18.3, 27.3],
        "optim": [2.8, 1.9, 12.5, 5.0, 12.1],
    },
    "Claude": {
        "origin": [11.9, 6.7, 11.3, 7.3, 5.0],
        "optim": [5.0, 4.8, 8.1, 4.5, 5.0],
    },
    "Gemini": {
        "origin": [9.6, 26.2, 17.2, 11.8, 13.3],
        "optim": [4.3, 11.9, 13.8, 5.9, 10.0],
    },
}

COLOR_ORIGIN = "#4E79A7"
COLOR_OPTIM = "#76B7A0"

FS = 12.5  # 基准字号

plt.rcParams.update(
    {
        "font.family": "Arial",
        "font.size": FS,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": True,
        "axes.spines.bottom": True,
        "axes.linewidth": 0.7,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)

n_domains = len(labels)
x = np.arange(n_domains)

bar_w = 0.38
gap = 0.05
offsets = np.array([-bar_w / 2 - gap / 2, bar_w / 2 + gap / 2])

fig, axes = plt.subplots(
    1,
    4,
    figsize=(10.5, 2.8),
    sharey=False,
    facecolor="white",
)
fig.subplots_adjust(
    wspace=0.16,  # 子图间距进一步收窄
    left=0.07,
    right=0.98,
    top=0.82,
    bottom=0.18,  # 底部留白压缩
)

SPINE_COLOR = "#AAAAAA"
GRID_COLOR = "#E8E8E8"
TICK_COLOR = "#444444"
LABEL_COLOR = "#222222"

for ax, (model, d) in zip(axes, data.items()):

    origin_vals = d["origin"]
    optim_vals = d["optim"]

    bars_o = ax.bar(
        x + offsets[0],
        origin_vals,
        width=bar_w,
        color=COLOR_ORIGIN,
        linewidth=0,
        zorder=3,
    )
    bars_p = ax.bar(
        x + offsets[1],
        optim_vals,
        width=bar_w,
        color=COLOR_OPTIM,
        linewidth=0,
        zorder=3,
    )

    # ── value labels ──────────────────────────────────────────────────────────
    for bar, val in zip(bars_o, origin_vals):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.6,
            f"{val:.0f}",
            ha="center",
            va="bottom",
            fontsize=FS * 0.70,
            color=COLOR_ORIGIN,
            fontweight="bold",
        )
    for bar, val in zip(bars_p, optim_vals):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.6,
            f"{val:.0f}",
            ha="center",
            va="bottom",
            fontsize=FS * 0.70,
            color=COLOR_OPTIM,
            fontweight="bold",
        )

    # ── x-axis ────────────────────────────────────────────────────────────────
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=FS * 0.90, color=TICK_COLOR, rotation=0)
    ax.set_xlim(-0.65, n_domains - 0.35)

    # ── y-axis ────────────────────────────────────────────────────────────────
    ax.set_ylim(0, 60)
    ax.yaxis.set_major_locator(plt.MultipleLocator(10))
    ax.tick_params(axis="y", labelsize=FS * 0.90, colors=TICK_COLOR, pad=2)

    if ax is axes[0]:
        ax.set_ylabel(
            "Risk proportion (%)", fontsize=FS * 0.92, color=LABEL_COLOR, labelpad=4
        )
    else:
        ax.set_ylabel("")
        ax.tick_params(axis="y", labelleft=False)

    # ── grid ──────────────────────────────────────────────────────────────────
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color=GRID_COLOR, linewidth=0.55, zorder=0)
    ax.set_facecolor("white")

    # ── spines & ticks ────────────────────────────────────────────────────────
    for sp in ["left", "bottom"]:
        ax.spines[sp].set_color(SPINE_COLOR)
        ax.spines[sp].set_linewidth(0.7)
    ax.tick_params(axis="x", length=3.0, width=0.7, colors=TICK_COLOR)
    ax.tick_params(axis="y", length=3.0, width=0.7, colors=TICK_COLOR)

    # ── panel title ───────────────────────────────────────────────────────────
    ax.set_title(model, fontsize=FS * 1.05, fontweight="bold", color=LABEL_COLOR, pad=5)

# ── shared legend ─────────────────────────────────────────────────────────────
patch_o = mpatches.Patch(color=COLOR_ORIGIN, label="Origin")
patch_p = mpatches.Patch(color=COLOR_OPTIM, label="Optim")
fig.legend(
    handles=[patch_o, patch_p],
    loc="upper center",
    ncol=2,
    frameon=False,
    fontsize=FS,
    handlelength=1.1,
    handleheight=0.9,
    columnspacing=1.2,
    bbox_to_anchor=(0.54, 1.03),
)

for ext in ("pdf", "png"):
    path = f"doubt_result.{ext}"
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"Saved → {path}")

plt.show()
