import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FixedFormatter

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["axes.unicode_minus"] = False

# ============================================================
# 数据
# ============================================================
domains = ["Application", "Program", "IoT", "Finance", "Web"]
data_fig1 = {
    "GuardAgent": [0.4409, 0.3014, 0.3636, 0.8052, 0.3],
    "AGrail": [0.4316, 0.3765, 0.5517, 0.6667, 0.0],
    "CoTSafe": [0.6667, 0.3659, 0.5714, 0.5882, 0.6667],
    "EVOLVE": [0.9393, 0.9286, 0.8333, 0.8593, 0.8649],
}

labels_fig2 = ["Accuracy", "Precision", "Recall", "F1-score"]
data_fig2 = {
    "w/ AnalysisAgent": [0.7846, 0.5409, 0.7679, 0.6355],
    "w/ FusionAgent": [0.8932, 0.8800, 0.8963, 0.8881],
    "w/ AuditorAgent": [0.8581, 0.8733, 0.8528, 0.8630],
    "Full EVOLVE": [0.9107, 0.8982, 0.9148, 0.9065],
}

# ============================================================
# 共享样式变量
# ============================================================
EVOLVE_COLOR = "#1b9e77"
BASELINE_COLOR = "#bdbdbd"
ABLATION_COLORS = ["#F1C40F", "#3498DB", "#2ECC71", "#E74C3C"]
ABLATION_MARKERS = ["p", "s", "^", "D"]

LABEL_FONTSIZE = 16
TICK_FONTSIZE = 14
LEGEND_FONTSIZE = 13
ANNOT_FONTSIZE = 12
SPINE_LW = 1.4

# ============================================================
# 画布参数：两张独立图，保证相同尺寸与边距
# ============================================================
FIGSIZE = (6.2, 4.9)
MARGINS = dict(left=0.12, right=0.98, bottom=0.17, top=0.90)

# ────────────────────────────────────────────────
# 图 1：Baseline Comparison (F1 across domains)
# ────────────────────────────────────────────────
fig1, ax1 = plt.subplots(figsize=FIGSIZE)
fig1.subplots_adjust(**MARGINS)
x1 = np.arange(len(domains)) * 0.8

for method, values in data_fig1.items():
    if method == "EVOLVE":
        ax1.plot(
            x1,
            values,
            marker="o",
            linewidth=2.8,
            markersize=10,
            color=EVOLVE_COLOR,
            label="EVOLVE",
            zorder=3,
        )
        evolve_labels = [0.939, 0.859, 0.833, 0.929, 0.865]
        for ix, val in enumerate(evolve_labels):
            ax1.annotate(
                f"{val:.3f}",
                xy=(x1[ix], values[ix]),
                xytext=(0, 8),
                textcoords="offset points",
                ha="center",
                fontsize=ANNOT_FONTSIZE,
                fontweight="bold",
                color=EVOLVE_COLOR,
            )
    else:
        ax1.plot(
            x1,
            values,
            marker="o",
            linestyle="--",
            linewidth=1.6,
            markersize=8,
            color=BASELINE_COLOR,
            alpha=0.9,
            label=method,
            zorder=2,
        )

ax1.set_ylabel("F1", fontsize=LABEL_FONTSIZE, fontweight="bold")
ax1.set_xticks(x1)
ax1.set_xticklabels(domains, fontsize=TICK_FONTSIZE)
ax1.tick_params(axis="y", labelsize=TICK_FONTSIZE)
ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)
ax1.spines["left"].set_linewidth(SPINE_LW)
ax1.spines["bottom"].set_linewidth(SPINE_LW)
ax1.grid(axis="y", linestyle="--", alpha=0.3)
ax1.legend(frameon=False, fontsize=LEGEND_FONTSIZE)
ax1.set_title("")

# ────────────────────────────────────────────────
# 图 2：Ablation Study（非线性纵轴）
# ────────────────────────────────────────────────
fig2, ax2 = plt.subplots(figsize=FIGSIZE)
fig2.subplots_adjust(**MARGINS)
x2 = np.arange(len(labels_fig2))

gamma = 4.0
ax2.set_yscale(
    "function",
    functions=(
        lambda y: np.power(np.clip(y, 1e-9, None), gamma),
        lambda y: np.power(np.clip(y, 1e-9, None), 1 / gamma),
    ),
)

for i, (mode_name, values) in enumerate(data_fig2.items()):
    is_full = "Full" in mode_name
    ax2.plot(
        x2,
        values,
        label=mode_name,
        color=ABLATION_COLORS[i],
        marker=ABLATION_MARKERS[i],
        markersize=10,
        linewidth=2.8 if is_full else 2.0,
        linestyle="-" if is_full else "--",
        zorder=10 if is_full else 5,
        alpha=1.0 if is_full else 0.85,
    )

    for ix, val in enumerate(values):
        custom_offsets = {
            ("w/ FusionAgent", 1): (10, 8),
            ("w/ AuditorAgent", 1): (-10, -6),
            ("w/ AnalysisAgent", 1): (0, 10),
        }
        dx, dy = custom_offsets.get((mode_name, ix), (0, 8 if is_full else -13))
        ax2.annotate(
            f"{val:.3f}",
            xy=(ix, val),
            xytext=(dx, dy),
            textcoords="offset points",
            ha="center",
            fontsize=ANNOT_FONTSIZE,
            fontweight="bold",
            color=ABLATION_COLORS[i],
        )

ax2.set_xticks(x2)
ax2.set_xticklabels(labels_fig2, fontsize=TICK_FONTSIZE)
ax2.tick_params(axis="y", labelsize=TICK_FONTSIZE)

tick_values = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
ax2.yaxis.set_major_locator(FixedLocator(tick_values))
ax2.yaxis.set_major_formatter(FixedFormatter([f"{v:.1f}" for v in tick_values]))
ax2.set_ylim(0.5, 0.95)

ax2.set_ylabel(
    "Ablation Performance Metrics",
    fontsize=LABEL_FONTSIZE,
    fontweight="bold",
    labelpad=8,
)
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)
ax2.spines["left"].set_linewidth(SPINE_LW)
ax2.spines["bottom"].set_linewidth(SPINE_LW)
ax2.grid(True, linestyle=":", which="both", color="grey", alpha=0.4)

legend2 = ax2.legend(
    loc="lower left", frameon=True, fontsize=LEGEND_FONTSIZE, edgecolor="black", ncol=1
)
legend2.get_frame().set_linewidth(1.2)
ax2.set_title("")

# ============================================================
# 输出
# ============================================================
fig1.savefig("detail_domains.png", dpi=300, bbox_inches="tight")
fig1.savefig(
    "detail_domains.pdf",
    format="pdf",
    bbox_inches="tight",
    facecolor="white",
    edgecolor="none",
)
fig2.savefig("detail_ablation.png", dpi=300, bbox_inches="tight")
fig2.savefig(
    "detail_ablation.pdf",
    format="pdf",
    bbox_inches="tight",
    facecolor="white",
    edgecolor="none",
)
plt.show()
