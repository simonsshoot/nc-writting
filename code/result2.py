import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ── Global style (matching doubt_result_v2) ──────────────────────────────────
FS = 12.5
plt.rcParams.update(
    {
        "font.family": "Arial",
        "font.size": FS,
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

SPINE_COLOR = "#AAAAAA"
GRID_COLOR = "#E8E8E8"
TICK_COLOR = "#444444"
LABEL_COLOR = "#222222"

# ── Data ─────────────────────────────────────────────────────────────────────
categories = ["App.", "Fin.", "IoT", "Prog.", "Web"]
models = ["DeepSeek", "GPT-4o", "Claude", "Gemini"]

data_need = {
    "DeepSeek": [69.44, 50.00, 83.33, 60.16, 48.57],
    "GPT-4o": [60.71, 46.83, 43.33, 53.91, 51.43],
    "Claude": [72.62, 38.53, 86.67, 77.78, 62.86],
    "Gemini": [61.90, 34.13, 66.67, 54.69, 42.86],
}

data_new = {
    "DeepSeek": [51.44, 43.16, 65.38, 20.33, 56.00],
    "GPT-4o": [14.34, 19.57, 75.00, 52.04, 57.14],
    "Claude": [58.42, 71.15, 75.81, 68.57, 56.67],
    "Gemini": [30.06, 25.00, 46.43, 40.79, 41.38],
}

# ── Layout ───────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 4, figsize=(10.5, 5.6), sharey=True, facecolor="white")
fig.subplots_adjust(wspace=0.12, hspace=0.30, top=0.90, bottom=0.08, left=0.07, right=0.98)

bar_w = 0.38
gap = 0.05
offsets = np.array([-bar_w / 2 - gap / 2, bar_w / 2 + gap / 2])
x = np.arange(len(categories))


def plot_bar(ax, val1, val2, title, color1, color2, label1, label2):
    bars1 = ax.bar(
        x + offsets[0], val1, width=bar_w,
        label=label1, color=color1, linewidth=0, zorder=3,
    )
    bars2 = ax.bar(
        x + offsets[1], val2, width=bar_w,
        label=label2, color=color2, linewidth=0, zorder=3,
    )

    ax.set_title(title, fontsize=FS * 1.05, fontweight="bold", color=LABEL_COLOR, pad=5)
    ax.set_xticks(x)
    ax.set_ylim(0, 115)
    ax.yaxis.set_major_locator(plt.MultipleLocator(20))

    # Grid
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color=GRID_COLOR, linewidth=0.55, zorder=0)
    ax.set_facecolor("white")

    # Spines – full box
    for sp in ["left", "bottom", "top", "right"]:
        ax.spines[sp].set_visible(True)
        ax.spines[sp].set_color(SPINE_COLOR)
        ax.spines[sp].set_linewidth(0.8)
    ax.tick_params(axis="x", length=3.0, width=0.7, colors=TICK_COLOR)
    ax.tick_params(axis="y", length=3.0, width=0.7, colors=TICK_COLOR, labelsize=FS * 0.90)

    # Value labels with smart stagger
    for i in range(len(categories)):
        v1, v2 = val1[i], val2[i]
        x1 = x[i] + offsets[0]
        x2 = x[i] + offsets[1]
        diff = abs(v1 - v2)

        if diff < 5:
            if v1 >= v2:
                off1, off2 = 3.5, -1.0
            else:
                off1, off2 = -1.0, 3.5
        elif diff < 10:
            if v1 >= v2:
                off1, off2 = 2.5, 0.5
            else:
                off1, off2 = 0.5, 2.5
        else:
            off1, off2 = 1.2, 1.2

        ax.text(
            x1, v1 + off1, f"{v1:.1f}",
            ha="center", va="bottom",
            fontsize=FS * 0.70, color=color1, fontweight="bold",
        )
        ax.text(
            x2, v2 + off2, f"{v2:.1f}",
            ha="center", va="bottom",
            fontsize=FS * 0.70, color=color2, fontweight="bold",
        )

    return bars1, bars2


# ── Draw subplots ────────────────────────────────────────────────────────────
for i, model in enumerate(models):
    # Row 1: Need vs No Need
    need_vals = data_need[model]
    noneed_vals = [100 - v for v in need_vals]
    plot_bar(
        axes[0, i], need_vals, noneed_vals,
        f"{model}",
        "#2E5A88", "#A8DADC", "Need", "No Need",
    )
    axes[0, i].tick_params(axis="x", labelbottom=False)

    # Row 2: New vs Reused
    new_vals = data_new[model]
    reused_vals = [100 - v for v in new_vals]
    plot_bar(
        axes[1, i], new_vals, reused_vals,
        f"{model}",
        "#D85C27", "#FFB385", "New", "Reused",
    )
    axes[1, i].set_xticklabels(
        categories, fontsize=FS * 0.90, color=TICK_COLOR, rotation=0,
        fontweight="bold",
    )

# ── Row labels ───────────────────────────────────────────────────────────────
axes[0, 0].set_ylabel("Need vs No Need (%)", fontsize=FS * 0.92, color=LABEL_COLOR, labelpad=4)
axes[1, 0].set_ylabel("New vs Reused (%)", fontsize=FS * 0.92, color=LABEL_COLOR, labelpad=4)

# ── Legends (one per row) ────────────────────────────────────────────────────
axes[0, -1].legend(
    loc="upper right", frameon=True, edgecolor=SPINE_COLOR,
    fancybox=False, fontsize=FS * 0.90,
)
axes[1, -1].legend(
    loc="upper right", frameon=True, edgecolor=SPINE_COLOR,
    fancybox=False, fontsize=FS * 0.90,
)

for ext in ("pdf", "png"):
    path = f"temp_reuse_result.{ext}"
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"Saved → {path}")
