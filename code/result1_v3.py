import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── Data ──────────────────────────────────────────────────────────────────────
methods = ["CoTSafe", "ShieldAgent", "AGrail", "GuardAgent", "EVOLVE"]
method_labels = ["CSafe", "Shield", "AGrail", "Guard", "EVOLVE"]

data = {
    "DeepSeek": {
        "RJudge": [
            (0.8091, 0.7815), (0.7863, 0.8318), (0.5403, 0.7601),
            (0.6935, 0.6086), (0.9107, 0.9065),
        ],
        "AsseBench": [
            (0.6316, 0.7722), (0.8046, 0.8095), (0.8678, 0.8587),
            (0.7259, 0.7485), (0.8868, 0.8690),
        ],
        "AgentHarm": [
            (0.7666, 0.6915), (0.5286, 0.1791), (0.6000, 0.4964),
            (0.7714, 0.7281), (0.8800, 0.8719),
        ],
    },
    "GPT-4o": {
        "RJudge": [
            (0.6786, 0.5804), (0.7863, 0.8318), (0.5494, 0.4910),
            (0.6848, 0.5455), (0.9002, 0.9123),
        ],
        "AsseBench": [
            (0.4663, 0.5781), (0.8046, 0.8095), (0.6924, 0.7482),
            (0.6184, 0.6336), (0.8633, 0.8869),
        ],
        "AgentHarm": [
            (0.5857, 0.4801), (0.5286, 0.1791), (0.5874, 0.3009),
            (0.5829, 0.3362), (0.8758, 0.9035),
        ],
    },
    "Claude": {
        "RJudge": [
            (0.8634, 0.8309), (0.7863, 0.8318), (0.7916, 0.7987),
            (0.6856, 0.5489), (0.8841, 0.8841),
        ],
        "AsseBench": [
            (0.7466, 0.7746), (0.8046, 0.8095), (0.8152, 0.7981),
            (0.7143, 0.7675), (0.8802, 0.8758),
        ],
        "AgentHarm": [
            (0.6543, 0.4718), (0.5286, 0.1791), (0.5543, 0.3741),
            (0.7600, 0.7467), (0.8812, 0.8994),
        ],
    },
    "Gemini": {
        "RJudge": [
            (0.8021, 0.7703), (0.7863, 0.8318), (0.7846, 0.7072),
            (0.6968, 0.6042), (0.9255, 0.9240),
        ],
        "AsseBench": [
            (0.6242, 0.6107), (0.8046, 0.8095), (0.8849, 0.8692),
            (0.7234, 0.7684), (0.8874, 0.8834),
        ],
        "AgentHarm": [
            (0.7771, 0.7547), (0.5286, 0.1791), (0.5457, 0.1846),
            (0.8333, 0.7297), (0.8886, 0.9107),
        ],
    },
}

backbones = ["DeepSeek", "GPT-4o", "Claude", "Gemini"]
benchmarks = ["RJudge", "AsseBench", "AgentHarm"]

# ── Color scheme ──────────────────────────────────────────────────────────────
COLORS = {
    "acc": "#4E79A7", "f1": "#76B7A0",
    "acc_m": "#9BB2C8", "f1_m": "#AED1C2",
}

# ── Global style ─────────────────────────────────────────────────────────────
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

n = len(methods)
bar_w = 0.28
gap = 0.05
offsets = np.array([-bar_w / 2 - gap / 2, bar_w / 2 + gap / 2])
x = np.arange(n)


def generate(scheme_key, colors):
    COLOR_ACC = colors["acc"]
    COLOR_F1 = colors["f1"]
    COLOR_ACC_MUTED = colors["acc_m"]
    COLOR_F1_MUTED = colors["f1_m"]

    fig, axes = plt.subplots(
        4, 3, figsize=(12.0, 10.0), dpi=150, facecolor="white",
    )
    fig.subplots_adjust(
        hspace=0.18, wspace=0.12, top=0.92, bottom=0.05, left=0.07, right=0.985
    )

    # Row labels
    for row, bb in enumerate(backbones):
        axes[row, 0].annotate(
            bb, xy=(0, 0.5), xycoords="axes fraction",
            xytext=(-38, 0), textcoords="offset points",
            ha="center", va="center", rotation=90,
            fontsize=FS * 1.0, fontweight="bold", color=LABEL_COLOR,
        )

    # Draw subplots
    for row, bb in enumerate(backbones):
        for col, bench in enumerate(benchmarks):
            ax = axes[row, col]
            ax.set_facecolor("white")
            ax.set_axisbelow(True)
            ax.yaxis.grid(True, color=GRID_COLOR, linewidth=0.55, zorder=0)

            pairs = data[bb][bench]
            acc_vals = [p[0] for p in pairs]
            f1_vals = [p[1] for p in pairs]

            for i in range(n):
                is_evolve = i == n - 1
                c_acc = COLOR_ACC if is_evolve else COLOR_ACC_MUTED
                c_f1 = COLOR_F1 if is_evolve else COLOR_F1_MUTED

                ax.bar(x[i] + offsets[0], acc_vals[i], width=bar_w,
                       color=c_acc, linewidth=0, zorder=3)
                ax.bar(x[i] + offsets[1], f1_vals[i], width=bar_w,
                       color=c_f1, linewidth=0, zorder=3)

                if is_evolve:
                    ax.bar(x[i] + offsets[0], acc_vals[i], width=bar_w,
                           color="none", edgecolor="#555555", linewidth=0.8, zorder=4)
                    ax.bar(x[i] + offsets[1], f1_vals[i], width=bar_w,
                           color="none", edgecolor="#555555", linewidth=0.8, zorder=4)

            # Value labels
            for i in range(n):
                is_evolve = i == n - 1
                fw = "bold" if is_evolve else "normal"
                xa = x[i] + offsets[0]
                xf = x[i] + offsets[1]
                diff = abs(acc_vals[i] - f1_vals[i])

                if diff < 0.025:
                    if acc_vals[i] >= f1_vals[i]:
                        acc_off, f1_off = 0.040, -0.010
                    else:
                        acc_off, f1_off = -0.010, 0.040
                elif diff < 0.06:
                    if acc_vals[i] >= f1_vals[i]:
                        acc_off, f1_off = 0.028, 0.008
                    else:
                        acc_off, f1_off = 0.008, 0.028
                else:
                    acc_off, f1_off = 0.016, 0.016

                ax.text(xa, acc_vals[i] + acc_off, f"{acc_vals[i]:.3f}",
                        ha="center", va="bottom",
                        fontsize=FS * 0.72, color=COLOR_ACC, fontweight=fw)
                ax.text(xf, f1_vals[i] + f1_off, f"{f1_vals[i]:.3f}",
                        ha="center", va="bottom",
                        fontsize=FS * 0.72, color=COLOR_F1, fontweight=fw)

            # x-axis
            ax.set_xticks(x)
            ax.set_xticklabels(
                method_labels, fontsize=FS * 0.88, rotation=0,
                ha="center", color=TICK_COLOR,
            )
            for lbl in ax.get_xticklabels():
                if lbl.get_text() == "EVOLVE":
                    lbl.set_fontweight("bold")
            ax.set_xlim(-0.55, n - 0.45)

            # y-axis
            ax.set_ylim(0, 1.14)
            ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
            ax.tick_params(axis="y", labelsize=FS * 0.82, colors=TICK_COLOR, pad=2)

            if col != 0:
                ax.tick_params(axis="y", labelleft=False)

            # Spines – full box
            for sp in ["left", "bottom", "top", "right"]:
                ax.spines[sp].set_visible(True)
                ax.spines[sp].set_color(SPINE_COLOR)
                ax.spines[sp].set_linewidth(0.8)
            ax.tick_params(axis="x", length=3.0, width=0.7, colors=TICK_COLOR)
            ax.tick_params(axis="y", length=3.0, width=0.7, colors=TICK_COLOR)

            if row == 0:
                ax.set_title(
                    bench, fontsize=FS * 1.0, color=LABEL_COLOR,
                    pad=8, fontweight="bold",
                )

    # Legend
    patch_acc = mpatches.Patch(color=COLOR_ACC, label="Acc")
    patch_f1 = mpatches.Patch(color=COLOR_F1, label="F1")
    fig.legend(
        handles=[patch_acc, patch_f1],
        loc="upper center", ncol=2,
        frameon=True, edgecolor=SPINE_COLOR, fancybox=False,
        fontsize=FS, handlelength=1.1, handleheight=0.9,
        columnspacing=1.2, bbox_to_anchor=(0.54, 0.99),
    )

    for ext in ("pdf", "png"):
        path = f"main_results{scheme_key}.{ext}"
        fig.savefig(path, bbox_inches="tight", dpi=300 if ext == "png" else 150)
    plt.close(fig)
    print(f"Saved main_results{scheme_key}.pdf / .png")


# ── Generate ─────────────────────────────────────────────────────────────────
generate("1", COLORS)

print("All done!")
