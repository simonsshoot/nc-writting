import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── Global style (unified with main_result_final.py) ─────────────────────────
FS = 12.5
plt.rcParams.update({
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
})

SPINE_COLOR = "#AAAAAA"
GRID_COLOR = "#E8E8E8"
TICK_COLOR = "#444444"
LABEL_COLOR = "#222222"

# Colors from main_result_final.py
COLOR_BLUE = "#4E79A7"
COLOR_GREEN = "#76B7A0"
COLOR_RED = "#B05252"
COLOR_BLUE_MUTED = "#9BB2C8"
COLOR_GREEN_MUTED = "#AED1C2"
COLOR_RED_MUTED = "#CFA0A0"


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 1: Doubt Result (Evolution Auditing)
# ═══════════════════════════════════════════════════════════════════════════════
labels_doubt = ["Web", "Prog.", "IoT", "Fin.", "App."]

data_doubt = {
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

n_domains = len(labels_doubt)
x_doubt = np.arange(n_domains)
bar_w = 0.38
gap = 0.05
offsets = np.array([-bar_w / 2 - gap / 2, bar_w / 2 + gap / 2])

fig, axes = plt.subplots(1, 4, figsize=(10.5, 2.8), sharey=False, facecolor="white")
fig.subplots_adjust(wspace=0.16, left=0.07, right=0.98, top=0.82, bottom=0.18)

for ax, (model, d) in zip(axes, data_doubt.items()):
    origin_vals = d["origin"]
    optim_vals = d["optim"]

    bars_o = ax.bar(x_doubt + offsets[0], origin_vals, width=bar_w,
                    color=COLOR_RED, linewidth=0, zorder=3)
    bars_p = ax.bar(x_doubt + offsets[1], optim_vals, width=bar_w,
                    color=COLOR_BLUE, linewidth=0, zorder=3)

    for bar, val in zip(bars_o, origin_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.8,
                f"{val:.1f}", ha="center", va="bottom",
                fontsize=FS * 0.70, color=COLOR_RED, fontweight="bold")
    for bar, val in zip(bars_p, optim_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.8,
                f"{val:.1f}", ha="center", va="bottom",
                fontsize=FS * 0.70, color=COLOR_BLUE, fontweight="bold")

    ax.set_xticks(x_doubt)
    ax.set_xticklabels(labels_doubt, fontsize=FS * 0.90, color=TICK_COLOR,
                       rotation=0, fontweight="bold")
    ax.set_xlim(-0.65, n_domains - 0.35)
    ax.set_ylim(0, 60)
    ax.yaxis.set_major_locator(plt.MultipleLocator(10))
    ax.tick_params(axis="y", labelsize=FS * 0.90, colors=TICK_COLOR, pad=2)

    if ax is axes[0]:
        ax.set_ylabel("Risk proportion (%)", fontsize=FS * 0.92,
                      color=LABEL_COLOR, labelpad=4)
    else:
        ax.set_ylabel("")
        ax.tick_params(axis="y", labelleft=False)

    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color=GRID_COLOR, linewidth=0.55, zorder=0)
    ax.set_facecolor("white")

    for sp in ["left", "bottom", "top", "right"]:
        ax.spines[sp].set_visible(True)
        ax.spines[sp].set_color(SPINE_COLOR)
        ax.spines[sp].set_linewidth(0.8)
    ax.tick_params(axis="x", length=3.0, width=0.7, colors=TICK_COLOR)
    ax.tick_params(axis="y", length=3.0, width=0.7, colors=TICK_COLOR)

    ax.set_title(model, fontsize=FS * 1.05, fontweight="bold",
                 color=LABEL_COLOR, pad=5)

patch_o = mpatches.Patch(color=COLOR_RED, label="Origin")
patch_p = mpatches.Patch(color=COLOR_BLUE, label="Optimized")
leg = fig.legend(handles=[patch_o, patch_p], loc="upper center", ncol=2,
           frameon=True, edgecolor=SPINE_COLOR, fancybox=False,
           fontsize=FS, handlelength=1.1, handleheight=0.9,
           columnspacing=1.2, bbox_to_anchor=(0.54, 1.06))
for text in leg.get_texts():
    text.set_fontweight("bold")

for ext in ("pdf", "png"):
    fig.savefig(f"doubt_result.{ext}", dpi=300, bbox_inches="tight", facecolor="white")
    print(f"Saved → doubt_result.{ext}")
plt.close()


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 2: Reuse Result (Tool Usage & Reuse)
# ═══════════════════════════════════════════════════════════════════════════════
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

x_reuse = np.arange(len(categories))

fig2, axes2 = plt.subplots(2, 4, figsize=(10.5, 5.6), sharey=True, facecolor="white")
fig2.subplots_adjust(wspace=0.12, hspace=0.30, top=0.90, bottom=0.08,
                     left=0.07, right=0.98)


def plot_bar(ax, val1, val2, title, color1, color2, label1, label2):
    bars1 = ax.bar(x_reuse + offsets[0], val1, width=bar_w,
                   label=label1, color=color1, linewidth=0, zorder=3)
    bars2 = ax.bar(x_reuse + offsets[1], val2, width=bar_w,
                   label=label2, color=color2, linewidth=0, zorder=3)

    ax.set_title(title, fontsize=FS * 1.05, fontweight="bold",
                 color=LABEL_COLOR, pad=5)
    ax.set_xticks(x_reuse)
    ax.set_ylim(0, 115)
    ax.yaxis.set_major_locator(plt.MultipleLocator(20))

    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color=GRID_COLOR, linewidth=0.55, zorder=0)
    ax.set_facecolor("white")

    for sp in ["left", "bottom", "top", "right"]:
        ax.spines[sp].set_visible(True)
        ax.spines[sp].set_color(SPINE_COLOR)
        ax.spines[sp].set_linewidth(0.8)
    ax.tick_params(axis="x", length=3.0, width=0.7, colors=TICK_COLOR)
    ax.tick_params(axis="y", length=3.0, width=0.7, colors=TICK_COLOR,
                   labelsize=FS * 0.90)

    for i in range(len(categories)):
        v1, v2 = val1[i], val2[i]
        x1 = x_reuse[i] + offsets[0]
        x2 = x_reuse[i] + offsets[1]
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

        ax.text(x1, v1 + off1, f"{v1:.1f}", ha="center", va="bottom",
                fontsize=FS * 0.70, color=color1, fontweight="bold")
        ax.text(x2, v2 + off2, f"{v2:.1f}", ha="center", va="bottom",
                fontsize=FS * 0.70, color=color2, fontweight="bold")

    return bars1, bars2


for i, model in enumerate(models):
    # Row 1: Need vs No Need
    need_vals = data_need[model]
    noneed_vals = [100 - v for v in need_vals]
    plot_bar(axes2[0, i], need_vals, noneed_vals, f"{model}",
             COLOR_BLUE, COLOR_BLUE_MUTED, "Need", "No Need")
    axes2[0, i].tick_params(axis="x", labelbottom=False)

    # Row 2: New vs Reused
    new_vals = data_new[model]
    reused_vals = [100 - v for v in new_vals]
    plot_bar(axes2[1, i], new_vals, reused_vals, f"{model}",
             COLOR_RED, COLOR_RED_MUTED, "New", "Reused")
    axes2[1, i].set_xticklabels(categories, fontsize=FS * 0.90,
                                color=TICK_COLOR, rotation=0, fontweight="bold")

axes2[0, 0].set_ylabel("Need vs No Need (%)", fontsize=FS * 0.92,
                        color=LABEL_COLOR, labelpad=4)
axes2[1, 0].set_ylabel("New vs Reused (%)", fontsize=FS * 0.92,
                        color=LABEL_COLOR, labelpad=4)

axes2[0, -1].legend(loc="upper right", frameon=True, edgecolor=SPINE_COLOR,
                    fancybox=False, fontsize=FS * 0.90)
axes2[1, -1].legend(loc="upper right", frameon=True, edgecolor=SPINE_COLOR,
                    fancybox=False, fontsize=FS * 0.90)

for ext in ("pdf", "png"):
    path = f"temp_reuse_result.{ext}"
    fig2.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"Saved → {path}")
plt.close()
