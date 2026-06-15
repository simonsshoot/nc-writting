import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

matplotlib.use("MacOSX")

# ── Data (order: CoTSafe, ShieldAgent, AGrail, GuardAgent, EVOLVE) ────────────
methods = ["CoTSafe", "ShieldAgent", "AGrail", "GuardAgent", "EVOLVE"]
method_labels = ["CSafe", "Shield", "AGrail", "Guard", "EVOLVE"]

attacks = ["Multilingual", "Jailbroken", "PAP", "ReneLLM"]

perturbed = {
    "Multilingual": [0.82, 0.92, 0.86, 0.92, 0.92],
    "Jailbroken": [0.72, 0.92, 1.00, 0.94, 1.00],
    "PAP": [0.84, 0.94, 0.92, 0.82, 0.94],
    "ReneLLM": [0.64, 0.90, 0.88, 0.90, 0.94],
}

original = {
    "Multilingual": [0.88, 0.96, 0.88, 0.96, 0.96],
    "Jailbroken": [0.88, 0.96, 0.88, 0.96, 0.96],
    "PAP": [0.88, 0.96, 0.88, 0.96, 0.96],
    "ReneLLM": [0.88, 0.96, 0.88, 0.96, 0.96],
}

# ── Style ─────────────────────────────────────────────────────────────────────
FS = 14
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
VAL_COLOR = "#111111"

COLOR_PERTURBED = "#B05252"
COLOR_ORIGINAL = "#4E79A7"
COLOR_PERTURBED_MUT = "#CFA0A0"
COLOR_ORIGINAL_MUT = "#9BB2C8"

# ── Layout ────────────────────────────────────────────────────────────────────
n = len(methods)
bar_w = 0.30
gap = 0.06
offsets = np.array([-bar_w / 2 - gap / 2, bar_w / 2 + gap / 2])
x = np.arange(n)

fig, axes = plt.subplots(1, 4, figsize=(16.0, 3.8), dpi=150, facecolor="white")
fig.subplots_adjust(wspace=0.18, top=0.82, bottom=0.15, left=0.05, right=0.98)

for col, attack in enumerate(attacks):
    ax = axes[col]
    ax.set_facecolor("white")
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color=GRID_COLOR, linewidth=0.55, zorder=0)

    p_vals = perturbed[attack]
    o_vals = original[attack]

    for i in range(n):
        is_evolve = i == n - 1
        c_p = COLOR_PERTURBED if is_evolve else COLOR_PERTURBED_MUT
        c_o = COLOR_ORIGINAL if is_evolve else COLOR_ORIGINAL_MUT

        ax.bar(
            x[i] + offsets[0],
            p_vals[i],
            width=bar_w,
            color=c_p,
            linewidth=0,
            zorder=2,
        )
        ax.bar(
            x[i] + offsets[1],
            o_vals[i],
            width=bar_w,
            color=c_o,
            linewidth=0,
            zorder=2,
        )

        if is_evolve:
            ax.bar(
                x[i] + offsets[0],
                p_vals[i],
                width=bar_w,
                color="none",
                edgecolor="#444444",
                linewidth=0.9,
                zorder=3,
            )
            ax.bar(
                x[i] + offsets[1],
                o_vals[i],
                width=bar_w,
                color="none",
                edgecolor="#444444",
                linewidth=0.9,
                zorder=3,
            )

    # trend lines
    x_p = x + offsets[0]
    x_o = x + offsets[1]
    ax.plot(
        x_p,
        p_vals,
        color=COLOR_PERTURBED,
        linewidth=1.8,
        linestyle="-",
        marker="",
        alpha=0.90,
        zorder=5,
    )
    ax.plot(
        x_o,
        o_vals,
        color=COLOR_ORIGINAL,
        linewidth=1.8,
        linestyle="--",
        marker="",
        alpha=0.90,
        zorder=5,
    )

    # value labels
    for i in range(n):
        is_evolve = i == n - 1
        fw = "bold" if is_evolve else "normal"
        col_txt = "#000000" if is_evolve else VAL_COLOR

        ax.text(
            x[i] + offsets[0],
            p_vals[i] + 0.01,
            f"{p_vals[i]:.2f}",
            ha="center",
            va="bottom",
            fontsize=FS * 0.95,
            color=col_txt,
            fontweight=fw,
            zorder=6,
        )
        ax.text(
            x[i] + offsets[1],
            o_vals[i] + 0.01,
            f"{o_vals[i]:.2f}",
            ha="center",
            va="bottom",
            fontsize=FS * 0.95,
            color=col_txt,
            fontweight=fw,
            zorder=6,
        )

    # x-axis
    ax.set_xticks(x)
    ax.set_xticklabels(method_labels, fontsize=FS * 0.92, ha="center", color=TICK_COLOR)
    for lbl in ax.get_xticklabels():
        if lbl.get_text() == "EVOLVE":
            lbl.set_fontweight("bold")
    ax.set_xlim(-0.55, n - 0.45)

    # y-axis
    ax.set_ylim(0.4, 1.15)
    ax.set_yticks([0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    ax.tick_params(axis="y", labelsize=FS * 0.90, colors=TICK_COLOR, pad=2)
    if col != 0:
        ax.tick_params(axis="y", labelleft=False)

    for sp in ["left", "bottom", "top", "right"]:
        ax.spines[sp].set_visible(True)
        ax.spines[sp].set_color(SPINE_COLOR)
        ax.spines[sp].set_linewidth(0.8)
    ax.tick_params(axis="x", length=3.0, width=0.7, colors=TICK_COLOR)
    ax.tick_params(axis="y", length=3.0, width=0.7, colors=TICK_COLOR)

    ax.set_title(attack, fontsize=FS * 1.0, color=LABEL_COLOR, pad=8, fontweight="bold")

# ── Legend ─────────────────────────────────────────────────────────────────────
handles = [
    mpatches.Patch(color=COLOR_PERTURBED, label="Perturb"),
    mpatches.Patch(color=COLOR_ORIGINAL, label="Origin"),
]
fig.legend(
    handles=handles,
    loc="upper center",
    ncol=2,
    frameon=True,
    edgecolor=SPINE_COLOR,
    fancybox=False,
    fontsize=FS * 0.92,
    handlelength=1.1,
    handleheight=0.85,
    columnspacing=1.5,
    handletextpad=0.5,
    bbox_to_anchor=(0.5, 0.98),
)

for ext in ("pdf", "png"):
    fig.savefig(
        f"attack_result.{ext}", bbox_inches="tight", dpi=300 if ext == "png" else 150
    )
plt.close(fig)
print("Saved attack_result.pdf / .png")
