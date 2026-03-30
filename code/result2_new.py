import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

categories = ["Application", "Finance", "IoT", "Program", "Web"]
models = {
    "DeepSeek": {
        "origin": [26.06, 21.62, 43.24, 44.78, 26.47],
        "optim": [5.21, 7.21, 13.51, 8.20, 5.88],
    },
    "GPT-4o": {
        "origin": [12.89, 15.89, 50.00, 18.33, 27.27],
        "optim": [2.79, 1.87, 12.50, 5.00, 12.12],
    },
    "Claude": {
        "origin": [11.88, 6.73, 11.29, 7.35, 5.00],
        "optim": [4.95, 4.81, 8.06, 4.49, 5.00],
    },
    "Gemini": {
        "origin": [9.57, 26.19, 17.24, 11.76, 13.33],
        "optim": [4.26, 11.90, 13.79, 5.88, 10.00],
    },
}

COLOR_ORIGIN = "#E04040"
COLOR_OPTIM = "#2EAA6E"
LW = 8  # outer line width (pt)
HOLLOW_RATIO = 0.2  # inner white line width = LW * HOLLOW_RATIO
ORIGIN_OFFSET = -0.13  # vertical offset from category center
OPTIM_OFFSET = 0.13

fig, axes = plt.subplots(1, 4, figsize=(22, 6.0), facecolor="white")
fig.patch.set_facecolor("white")
y_pos = np.arange(len(categories), dtype=float)

for ax_idx, (model_name, data) in enumerate(models.items()):
    ax = axes[ax_idx]
    ax.set_facecolor("#F7F7F7")

    origin = data["origin"]
    optim = data["optim"]

    x_max = max(55, int(np.ceil(max(origin) / 10) * 10) + 5)

    for i in range(len(categories)):
        yo = y_pos[i] + ORIGIN_OFFSET
        yp = y_pos[i] + OPTIM_OFFSET

        # origin bar — hollow tube: thick colored outer + thin white inner
        ax.plot(
            [0, origin[i]],
            [yo, yo],
            color=COLOR_ORIGIN,
            linewidth=LW,
            solid_capstyle="round",
            zorder=3,
        )
        ax.plot(
            [0, origin[i]],
            [yo, yo],
            color="white",
            linewidth=LW * HOLLOW_RATIO,
            solid_capstyle="butt",
            zorder=4,
        )

        # optim bar — hollow tube
        ax.plot(
            [0, optim[i]],
            [yp, yp],
            color=COLOR_OPTIM,
            linewidth=LW,
            solid_capstyle="round",
            zorder=3,
        )
        ax.plot(
            [0, optim[i]],
            [yp, yp],
            color="white",
            linewidth=LW * HOLLOW_RATIO,
            solid_capstyle="butt",
            zorder=4,
        )

        # Data labels at right end of each bar
        ax.text(
            origin[i] + x_max * 0.012,
            yo,
            f"{origin[i]:.1f}%",
            va="center",
            ha="left",
            fontsize=14,
            color=COLOR_ORIGIN,
            fontweight="500",
            zorder=5,
        )
        ax.text(
            optim[i] + x_max * 0.012,
            yp,
            f"{optim[i]:.1f}%",
            va="center",
            ha="left",
            fontsize=14,
            color=COLOR_OPTIM,
            fontweight="500",
            zorder=5,
        )

    # Grid
    ax.xaxis.grid(True, color="#DDDDDD", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)

    # x axis
    x_ticks = np.arange(0, x_max + 1, 10)
    ax.set_xticks(x_ticks)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x)}%"))
    ax.tick_params(axis="x", labelsize=12, colors="#555555", length=3)
    ax.set_xlim(0, x_max + x_max * 0.18)  # extra right padding for labels

    # y axis
    ax.set_yticks(y_pos)
    ax.set_yticklabels(categories, fontsize=14, color="#333333")
    ax.tick_params(axis="y", length=0)
    ax.set_ylim(-0.6, len(categories) - 0.4)

    # Spines — keep only bottom
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.spines["bottom"].set_visible(True)
    ax.spines["bottom"].set_color("#CCCCCC")
    ax.spines["bottom"].set_linewidth(0.8)

    ax.set_title(model_name, fontsize=14, fontweight="600", color="#222222", pad=12)

# Shared legend
legend_elements = [
    mpatches.Patch(facecolor=COLOR_ORIGIN, label="origin"),
    mpatches.Patch(facecolor=COLOR_OPTIM, label="optim"),
]
fig.legend(
    handles=legend_elements,
    loc="upper left",
    fontsize=14,
    frameon=False,
    bbox_to_anchor=(0.01, 1.05),
    ncol=2,
    handlelength=1.0,
    handleheight=0.85,
    columnspacing=1.2,
)

plt.tight_layout(pad=2.2, w_pad=3.5)
plt.savefig("doubt_result.png", dpi=180, bbox_inches="tight", facecolor="white")
print("Saved: doubt_result.png")
plt.savefig(
    "doubt_result.pdf",
    format="pdf",
    bbox_inches="tight",
    facecolor="white",
    edgecolor="none",
)
