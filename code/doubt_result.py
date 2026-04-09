import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── Data ──────────────────────────────────────────────────────────────────────
labels = ["Web", "Program", "IoT", "Finance", "Application"]
ORDER = [4, 3, 2, 1, 0]

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

# ── Style ─────────────────────────────────────────────────────────────────────
COLOR_ORIGIN = "#D85A30"  # coral-red  (origin)
COLOR_OPTIM = "#1D9E75"  # teal-green (optim)
BG_COLOR = "#FAFAF9"
GRID_COLOR = "#E5E4DF"

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 8.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": False,
        "axes.spines.bottom": True,
        "xtick.direction": "out",
        "ytick.direction": "out",
    }
)

n_groups = len(labels)  # 5 domains
bar_h = 0.32  # height of each bar
gap = 0.06  # gap between the pair
group_pad = 0.52  # padding between groups

# y positions for each group centre
centres = np.arange(n_groups) * (2 * bar_h + gap + group_pad)
y_origin = centres + bar_h / 2 + gap / 2
y_optim = centres - bar_h / 2 - gap / 2

fig, axes = plt.subplots(
    1,
    4,
    figsize=(14, 3.6),
    facecolor="white",
    sharey=True,
)
fig.subplots_adjust(wspace=0.18, left=0.08, right=0.97, top=0.88, bottom=0.12)

for ax, (model, d) in zip(axes, data.items()):

    origin_vals = [d["origin"][i] for i in ORDER]
    optim_vals = [d["optim"][i] for i in ORDER]

    # ── bars ──────────────────────────────────────────────────────────────────
    bars_o = ax.barh(
        y_origin,
        origin_vals,
        height=bar_h,
        color=COLOR_ORIGIN,
        alpha=0.88,
        linewidth=0,
        zorder=3,
    )
    bars_p = ax.barh(
        y_optim,
        optim_vals,
        height=bar_h,
        color=COLOR_OPTIM,
        alpha=0.88,
        linewidth=0,
        zorder=3,
    )

    # ── data labels ───────────────────────────────────────────────────────────
    for bar, val in zip(bars_o, origin_vals):
        ax.text(
            bar.get_width() + 0.6,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.1f}%",
            va="center",
            ha="left",
            fontsize=7.2,
            color=COLOR_ORIGIN,
            fontweight="500",
        )
    for bar, val in zip(bars_p, optim_vals):
        ax.text(
            bar.get_width() + 0.6,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.1f}%",
            va="center",
            ha="left",
            fontsize=7.2,
            color=COLOR_OPTIM,
            fontweight="500",
        )

    # ── y-axis labels (only on first subplot) ─────────────────────────────────
    ax.set_yticks(centres)
    if ax is axes[0]:
        ax.set_yticklabels(labels, fontsize=8.5)
        ax.tick_params(axis="y", labelleft=True)
    else:
        ax.tick_params(axis="y", labelleft=False)

    # ── x-axis ────────────────────────────────────────────────────────────────
    ax.set_xlim(0, 62)
    ax.set_xlabel("Risk proportion (%)", fontsize=8.5, labelpad=4)
    ax.xaxis.set_major_locator(plt.MultipleLocator(10))
    ax.xaxis.set_minor_locator(plt.MultipleLocator(5))
    ax.tick_params(axis="x", labelsize=7.8)

    # ── grid ──────────────────────────────────────────────────────────────────
    ax.set_axisbelow(True)
    ax.xaxis.grid(True, color=GRID_COLOR, linewidth=0.6, zorder=0)
    ax.xaxis.grid(
        True, which="minor", color=GRID_COLOR, linewidth=0.3, linestyle=":", zorder=0
    )
    ax.set_facecolor("white")

    # ── subplot title ─────────────────────────────────────────────────────────
    ax.set_title(model, fontsize=10, fontweight="500", pad=7, color="#2C2C2A")

    # ── spine style ───────────────────────────────────────────────────────────
    ax.spines["bottom"].set_color("#C0BEB5")
    ax.spines["bottom"].set_linewidth(0.6)
    ax.tick_params(axis="y", length=0)

# ── shared legend ─────────────────────────────────────────────────────────────
patch_o = mpatches.Patch(color=COLOR_ORIGIN, alpha=0.88, label="Origin")
patch_p = mpatches.Patch(color=COLOR_OPTIM, alpha=0.88, label="Optim")
fig.legend(
    handles=[patch_o, patch_p],
    loc="upper center",
    ncol=2,
    frameon=False,
    fontsize=9,
    handlelength=1.2,
    handleheight=0.9,
    columnspacing=1.4,
    bbox_to_anchor=(0.54, 1.02),
)

out_path = "doubt_result.pdf"
plt.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
print(f"Saved → {out_path}")

out_png = "doubt_result.png"
plt.savefig(out_png, dpi=300, bbox_inches="tight", facecolor="white")

plt.show()
