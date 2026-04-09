import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ── Data ──────────────────────────────────────────────────────────────────────
methods = ["CoTSafe", "ShieldAgent", "AGrail", "GuardAgent", "EVOLVE"]
method_labels = ["CSafe", "Shield", "AGrail", "Guard", "EVOLVE"]

# Structure: data[backbone][benchmark] = { method: (Acc, F1) }
data = {
    "DeepSeek": {
        "RJudge": [
            (0.8091, 0.7815),
            (0.7863, 0.8318),
            (0.5403, 0.7601),
            (0.6935, 0.6086),
            (0.9107, 0.9065),
        ],
        "AsseBench": [
            (0.6316, 0.7722),
            (0.8046, 0.8095),
            (0.8678, 0.8587),
            (0.7259, 0.7485),
            (0.8868, 0.8690),
        ],
        "AgentHarm": [
            (0.7666, 0.6915),
            (0.5286, 0.1791),
            (0.6000, 0.4964),
            (0.7714, 0.7281),
            (0.8800, 0.8719),
        ],
    },
    "GPT-4o": {
        "RJudge": [
            (0.6786, 0.5804),
            (0.7863, 0.8318),
            (0.5494, 0.4910),
            (0.6848, 0.5455),
            (0.9002, 0.9123),
        ],
        "AsseBench": [
            (0.4663, 0.5781),
            (0.8046, 0.8095),
            (0.6924, 0.7482),
            (0.6184, 0.6336),
            (0.8633, 0.8869),
        ],
        "AgentHarm": [
            (0.5857, 0.4801),
            (0.5286, 0.1791),
            (0.5874, 0.3009),
            (0.5829, 0.3362),
            (0.8758, 0.9035),
        ],
    },
    "Claude": {
        "RJudge": [
            (0.8634, 0.8309),
            (0.7863, 0.8318),
            (0.7916, 0.7987),
            (0.6856, 0.5489),
            (0.8841, 0.8841),
        ],
        "AsseBench": [
            (0.7466, 0.7746),
            (0.8046, 0.8095),
            (0.8152, 0.7981),
            (0.7143, 0.7675),
            (0.8802, 0.8758),
        ],
        "AgentHarm": [
            (0.6543, 0.4718),
            (0.5286, 0.1791),
            (0.5543, 0.3741),
            (0.7600, 0.7467),
            (0.8812, 0.8994),
        ],
    },
    "Gemini": {
        "RJudge": [
            (0.8021, 0.7703),
            (0.7863, 0.8318),
            (0.7846, 0.7072),
            (0.6968, 0.6042),
            (0.9255, 0.9240),
        ],
        "AsseBench": [
            (0.6242, 0.6107),
            (0.8046, 0.8095),
            (0.8849, 0.8692),
            (0.7234, 0.7684),
            (0.8874, 0.8834),
        ],
        "AgentHarm": [
            (0.7771, 0.7547),
            (0.5286, 0.1791),
            (0.5457, 0.1846),
            (0.8333, 0.7297),
            (0.8886, 0.9107),
        ],
    },
}

backbones = ["DeepSeek", "GPT-4o", "Claude", "Gemini"]
benchmarks = ["RJudge", "AsseBench", "AgentHarm"]

# ── Palette (soft, academic — inspired by reference image) ────────────────────
# Acc bars: muted blue-grey family
# F1  bars: muted rose/coral family
# EVOLVE gets a slightly warmer highlight tone
COLOR_ACC = [
    "#9BB5C8",  # CoTSafe    – steel blue-grey
    "#A8C4A2",  # ShieldAgent – sage green
    "#C4B09A",  # AGrail      – warm tan
    "#B3A8C8",  # GuardAgent  – lavender
    "#E8A89C",  # EVOLVE      – salmon rose  (highlighted)
]
COLOR_F1 = [
    "#C8D8E4",  # CoTSafe
    "#C8DCC4",  # ShieldAgent
    "#DDD0C0",  # AGrail
    "#D4CDE4",  # GuardAgent
    "#F0C8C0",  # EVOLVE
]

EDGE = "white"
BAR_WIDTH = 0.40  # width of each single bar
GROUP_GAP = 0.01  # gap between Acc/F1 within a method
INTER_GROUP_GAP = 0.12  # gap between method groups (controls tick spacing)
FONTSIZE = 9.5

# ── Layout ────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(
    4,
    3,
    figsize=(11.2, 9.6),
    dpi=150,
    facecolor="white",
)
fig.subplots_adjust(
    hspace=0.10, wspace=0.06, top=0.965, bottom=0.06, left=0.07, right=0.985
)

# ── Row labels (backbone names) ──────────────────────────────────────────────
for row, bb in enumerate(backbones):
    axes[row, 0].annotate(
        bb,
        xy=(0, 0.5),
        xycoords="axes fraction",
        xytext=(-36, 0),
        textcoords="offset points",
        ha="center",
        va="center",
        rotation=90,
        fontsize=12.0,
        fontweight="bold",
        color="#333333",
    )

n = len(methods)
GROUP_WIDTH = 2 * BAR_WIDTH + GROUP_GAP
x_centers = np.arange(n) * (GROUP_WIDTH + INTER_GROUP_GAP)  # one tick per method

# ── Draw every subplot ────────────────────────────────────────────────────────
for row, bb in enumerate(backbones):
    for col, bench in enumerate(benchmarks):
        ax = axes[row, col]
        ax.set_facecolor("#F8F8FA")
        ax.grid(axis="y", color="white", linewidth=1.0, zorder=0)
        ax.set_axisbelow(True)

        pairs = data[bb][bench]  # list of (Acc, F1) per method
        acc_vals = [p[0] for p in pairs]
        f1_vals = [p[1] for p in pairs]

        # Each group: Acc bar left, F1 bar right, centered on x_centers
        offset_acc = -(BAR_WIDTH / 2 + GROUP_GAP / 2)
        offset_f1 = BAR_WIDTH / 2 + GROUP_GAP / 2
        close_thresh = 0.030

        for i in range(n):
            xa = x_centers[i] + offset_acc
            xf = x_centers[i] + offset_f1
            is_evolve = i == n - 1

            # Acc bar
            ba = ax.bar(
                xa,
                acc_vals[i],
                BAR_WIDTH,
                color=COLOR_ACC[i],
                edgecolor=EDGE,
                linewidth=0.6,
                zorder=3,
            )
            # F1 bar
            bf = ax.bar(
                xf,
                f1_vals[i],
                BAR_WIDTH,
                color=COLOR_F1[i],
                edgecolor=EDGE,
                linewidth=0.6,
                zorder=3,
            )

            # Value labels on top (stagger when two values are close)
            acc_off = 0.018
            f1_off = 0.006
            if abs(acc_vals[i] - f1_vals[i]) < close_thresh:
                acc_off = 0.032
                f1_off = -0.006
            if is_evolve:
                acc_off += 0.008
                f1_off -= 0.004
            ax.text(
                xa,
                acc_vals[i] + acc_off,
                f"{acc_vals[i]:.3f}",
                ha="center",
                va="bottom",
                fontsize=9.6,
                color="#444444",
                fontweight="bold" if is_evolve else "normal",
            )
            ax.text(
                xf,
                f1_vals[i] + f1_off,
                f"{f1_vals[i]:.3f}",
                ha="center",
                va="bottom",
                fontsize=9.6,
                color="#444444",
                fontweight="bold" if is_evolve else "normal",
            )

            # Highlight EVOLVE with a subtle outline
            if is_evolve:
                for b, c in [(ba, COLOR_ACC[i]), (bf, COLOR_F1[i])]:
                    for rect in b:
                        rect.set_edgecolor("#888888")
                        rect.set_linewidth(1.0)

        # Axes formatting
        ax.set_xticks(x_centers)
        ax.set_xticklabels(
            method_labels,
            fontsize=10.8,
            rotation=0,
            ha="center",
            color="#444444",
            fontweight="bold",
        )
        ax.set_xlim(
            x_centers[0] - GROUP_WIDTH / 2 - 0.10,
            x_centers[-1] + GROUP_WIDTH / 2 + 0.10,
        )
        ax.set_ylim(0, 1.10)
        ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(
            ["0", "0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=FONTSIZE, color="#555555"
        )

        if col == 0:
            ax.set_ylabel("")
        else:
            ax.set_ylabel("")
            ax.set_yticklabels([])

        if row == len(backbones) - 1:
            ax.set_xlabel("")

        if row == 0:
            ax.set_title(
                bench,
                fontsize=11.6,
                color="#333333",
                pad=6,
                fontweight="bold",
            )

        # Spine cleanup
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        for spine in ["left", "bottom"]:
            ax.spines[spine].set_color("#CCCCCC")
            ax.spines[spine].set_linewidth(0.8)

        ax.tick_params(axis="both", length=0)



plt.savefig("main_results.pdf", bbox_inches="tight", dpi=150)
plt.savefig("main_results.png", bbox_inches="tight", dpi=150)
print("Saved main_results.pdf and main_results.png")
