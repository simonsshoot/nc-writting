import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FixedLocator, FixedFormatter

# Global style: gentle, academic, Nature MI-like
plt.rcParams.update(
    {
        "font.family": "Times New Roman",
        "axes.unicode_minus": False,
        "axes.titleweight": "bold",
        "axes.labelweight": "bold",
        "axes.edgecolor": "#2A2A2A",
        "axes.linewidth": 1.2,
        "xtick.color": "#2A2A2A",
        "ytick.color": "#2A2A2A",
        "text.color": "#2A2A2A",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
    }
)


def get_ablation_data():
    labels = ["Accuracy", "Precision", "Recall", "F1-score"]
    dict_data = {
        "w/ AnalysisAgent": [0.7846, 0.5409, 0.7679, 0.6355],
        "w/ FusionAgent": [0.8932, 0.8800, 0.8963, 0.8881],
        "w/ AuditorAgent": [0.8581, 0.8733, 0.8528, 0.8630],
        "Full EVOLVE": [0.9107, 0.8982, 0.9148, 0.9065],
    }
    return labels, dict_data


def draw_nonlinear_ablation():
    labels, dict_data = get_ablation_data()
    x = np.arange(len(labels))

    # Softer, desaturated academic palette
    colors = ["#C9A830", "#4A8DB8", "#4DA882", "#C0605E"]
    markers = ["p", "s", "^", "D"]

    fig, ax = plt.subplots(figsize=(9, 7))

    # Nonlinear y scale to stretch high scores
    gamma = 4.0

    def forward(y):
        return np.power(y, gamma)

    def inverse(y):
        return np.power(y, 1 / gamma)

    ax.set_yscale("function", functions=(forward, inverse))

    # Plot lines
    # Per-point annotation offsets (dx, dy) in points to avoid overlap
    # Series order: 0=AnalysisAgent, 1=FusionAgent, 2=AuditorAgent, 3=Full EVOLVE
    # Metric order: 0=Accuracy, 1=Precision, 2=Recall, 3=F1-score
    offsets = {
        (0, 0): (0, -20), (0, 1): (0, -20), (0, 2): (0, -20), (0, 3): (0, -20),
        (1, 0): (0, -20), (1, 1): (24, 10),  (1, 2): (24, -8), (1, 3): (0, -20),
        (2, 0): (0, -20), (2, 1): (-24, -20), (2, 2): (0, -20), (2, 3): (0, -20),
        (3, 0): (0, 13),  (3, 1): (0, 13),  (3, 2): (0, 13),  (3, 3): (0, 13),
    }

    for i, (mode_name, data) in enumerate(dict_data.items()):
        is_full = "Full" in mode_name
        ax.plot(
            x,
            data,
            label=mode_name,
            color=colors[i],
            marker=markers[i],
            markersize=10,
            linewidth=3.0 if is_full else 2.2,
            linestyle="-" if is_full else "--",
            zorder=10 if is_full else 5,
            alpha=0.95 if is_full else 0.85,
        )

        # Value annotations
        for ix, val in enumerate(data):
            dx, dy = offsets[(i, ix)]
            ax.annotate(
                f"{val:.3f}",
                xy=(ix, val),
                xytext=(dx, dy),
                textcoords="offset points",
                ha="center",
                fontsize=16.5,
                fontweight="bold",
                color=colors[i],
            )

    # X axis
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=15, fontweight="bold")

    # Y axis ticks
    tick_values = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    ax.yaxis.set_major_locator(FixedLocator(tick_values))
    ax.yaxis.set_major_formatter(FixedFormatter([f"{v:.1f}" for v in tick_values]))
    plt.yticks(fontsize=13.5, fontweight="bold")
    plt.ylim(0.5, 0.95)

    # Labels
    ax.set_ylabel("Ablation Performance Metrics", fontsize=17, labelpad=10)

    # Subtle grid
    ax.grid(True, linestyle=":", linewidth=0.9, color="#B9B9B9", alpha=0.5)

    # Spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.2)
    ax.spines["bottom"].set_linewidth(1.2)

    # Legend
    legend = ax.legend(
        loc="lower left",
        frameon=True,
        fontsize=11.5,
        edgecolor="#4A4A4A",
    )
    legend.get_frame().set_linewidth(1.0)
    legend.get_frame().set_alpha(0.95)

    plt.tight_layout()
    plt.savefig("ablation.png", dpi=300, bbox_inches="tight")
    plt.savefig(
        "ablation_study_temp.pdf",
        format="pdf",
        bbox_inches="tight",
        facecolor="white",
        edgecolor="none",
    )
    plt.show()


if __name__ == "__main__":
    draw_nonlinear_ablation()
