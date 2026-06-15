import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

matplotlib.use("MacOSX")

# ── Data ──────────────────────────────────────────────────────────────────────
methods = ["CoTSafe", "ShieldAgent", "AGrail", "GuardAgent", "EVOLVE"]
method_labels = ["CSafe", "Shield", "AGrail", "Guard", "EVOLVE"]

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

PALETTE = {
    "DeepSeek": dict(acc="#4E79A7", f1="#76B7A0", acc_m="#9BB2C8", f1_m="#AED1C2"),
    "GPT-4o": dict(acc="#C97B4B", f1="#B05252", acc_m="#DDB593", f1_m="#CFA0A0"),
    "Claude": dict(acc="#7E6BAD", f1="#5B84B0", acc_m="#B8B0D0", f1_m="#A5B8CC"),
    "Gemini": dict(acc="#B5953B", f1="#6A9E6A", acc_m="#D4C08A", f1_m="#A8C3A8"),
}

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
VAL_COLOR = "#111111"

n = len(methods)
bar_w = 0.22
gap = 0.06
offsets = np.array([-bar_w / 2 - gap / 2, bar_w / 2 + gap / 2])
x = np.arange(n)

# ── 碰撞检测参数（数据坐标系）────────────────────────────────────────────────
LBL_W = 0.38
LBL_H = 0.075  # ↑ 0.062 → 0.075，字体占用高度估算放宽
BASE = 0.015  # ↑ 0.013 → 0.015，柱顶到标注底部的固定间距


def resolve_overlaps(labels, max_iter=120):  # ↑ 迭代次数也加多
    for _ in range(max_iter):
        moved = False
        for i in range(len(labels)):
            for j in range(i + 1, len(labels)):
                li, lj = labels[i], labels[j]
                dx = abs(li["x"] - lj["x"])
                dy = abs(li["y"] - lj["y"])
                if dx < LBL_W and dy < LBL_H:
                    overlap = LBL_H - dy
                    push = overlap / 2 + 0.012  # ↑ 冗余量 0.004 → 0.012
                    if li["y"] <= lj["y"]:
                        li["y"] -= push
                        lj["y"] += push
                    else:
                        li["y"] += push
                        lj["y"] -= push
                    moved = True
        for l in labels:
            floor = l["bar_top"] + BASE
            if l["y"] < floor:
                l["y"] = floor
        if not moved:
            break
    return labels


def generate(scheme_key):
    fig, axes = plt.subplots(
        4,
        3,
        figsize=(12.0, 10.5),
        dpi=150,
        facecolor="white",
    )
    fig.subplots_adjust(
        hspace=0.20, wspace=0.12, top=0.88, bottom=0.07, left=0.07, right=0.985
    )

    for row, bb in enumerate(backbones):
        axes[row, 0].annotate(
            bb,
            xy=(0, 0.5),
            xycoords="axes fraction",
            xytext=(-38, 0),
            textcoords="offset points",
            ha="center",
            va="center",
            rotation=90,
            fontsize=FS * 1.0,
            fontweight="bold",
            color=LABEL_COLOR,
        )

    for row, bb in enumerate(backbones):
        pal = PALETTE[bb]
        COLOR_ACC = pal["acc"]
        COLOR_F1 = pal["f1"]
        COLOR_ACC_MUT = pal["acc_m"]
        COLOR_F1_MUT = pal["f1_m"]

        for col, bench in enumerate(benchmarks):
            ax = axes[row, col]
            ax.set_facecolor("white")
            ax.set_axisbelow(True)
            ax.yaxis.grid(True, color=GRID_COLOR, linewidth=0.55, zorder=0)

            pairs = data[bb][bench]
            acc_vals = [p[0] for p in pairs]
            f1_vals = [p[1] for p in pairs]

            x_acc = x + offsets[0]
            x_f1 = x + offsets[1]

            # ── Bars ──────────────────────────────────────────────────────
            for i in range(n):
                is_evolve = i == n - 1
                c_acc = COLOR_ACC if is_evolve else COLOR_ACC_MUT
                c_f1 = COLOR_F1 if is_evolve else COLOR_F1_MUT

                ax.bar(
                    x[i] + offsets[0],
                    acc_vals[i],
                    width=bar_w,
                    color=c_acc,
                    linewidth=0,
                    zorder=2,
                )
                ax.bar(
                    x[i] + offsets[1],
                    f1_vals[i],
                    width=bar_w,
                    color=c_f1,
                    linewidth=0,
                    zorder=2,
                )

                if is_evolve:
                    ax.bar(
                        x[i] + offsets[0],
                        acc_vals[i],
                        width=bar_w,
                        color="none",
                        edgecolor="#444444",
                        linewidth=0.9,
                        zorder=3,
                    )
                    ax.bar(
                        x[i] + offsets[1],
                        f1_vals[i],
                        width=bar_w,
                        color="none",
                        edgecolor="#444444",
                        linewidth=0.9,
                        zorder=3,
                    )

            # ── Trend lines ───────────────────────────────────────────────
            ax.plot(
                x_acc,
                acc_vals,
                color=COLOR_ACC,
                linewidth=1.8,
                linestyle="-",
                marker="",
                alpha=0.90,
                zorder=5,
            )
            ax.plot(
                x_f1,
                f1_vals,
                color=COLOR_F1,
                linewidth=1.8,
                linestyle="--",
                marker="",
                alpha=0.90,
                zorder=5,
            )

            # ── 构建所有标注初始位置，然后全局去重叠 ──────────────────────
            labels = []
            for i in range(n):
                labels.append(
                    dict(
                        x=x_acc[i],
                        y=acc_vals[i] + BASE,
                        bar_top=acc_vals[i],
                        val=acc_vals[i],
                        idx=i,
                        kind="acc",
                    )
                )
                labels.append(
                    dict(
                        x=x_f1[i],
                        y=f1_vals[i] + BASE,
                        bar_top=f1_vals[i],
                        val=f1_vals[i],
                        idx=i,
                        kind="f1",
                    )
                )

            labels = resolve_overlaps(labels)

            # ── 绘制标注 ──────────────────────────────────────────────────
            for l in labels:
                i = l["idx"]
                is_evolve = i == n - 1
                fw = "bold" if is_evolve else "normal"
                col_txt = "#000000" if is_evolve else VAL_COLOR

                ax.text(
                    l["x"],
                    l["y"],
                    f"{l['val']:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=FS * 0.95,
                    color=col_txt,
                    fontweight=fw,
                    zorder=6,
                )

            # ── x-axis ────────────────────────────────────────────────────
            ax.set_xticks(x)
            ax.set_xticklabels(
                method_labels,
                fontsize=FS * 0.88,
                ha="center",
                color=TICK_COLOR,
            )
            for lbl in ax.get_xticklabels():
                if lbl.get_text() == "EVOLVE":
                    lbl.set_fontweight("bold")
            ax.set_xlim(-0.55, n - 0.45)

            # ── y-axis ────────────────────────────────────────────────────
            ax.set_ylim(0, 1.22)
            ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
            ax.tick_params(axis="y", labelsize=FS * 0.82, colors=TICK_COLOR, pad=2)
            if col != 0:
                ax.tick_params(axis="y", labelleft=False)

            for sp in ["left", "bottom", "top", "right"]:
                ax.spines[sp].set_visible(True)
                ax.spines[sp].set_color(SPINE_COLOR)
                ax.spines[sp].set_linewidth(0.8)
            ax.tick_params(axis="x", length=3.0, width=0.7, colors=TICK_COLOR)
            ax.tick_params(axis="y", length=3.0, width=0.7, colors=TICK_COLOR)

            if row == 0:
                ax.set_title(
                    bench,
                    fontsize=FS * 1.0,
                    color=LABEL_COLOR,
                    pad=8,
                    fontweight="bold",
                )

    # ── Legend ────────────────────────────────────────────────────────────────
    handles, labels_leg = [], []
    for bb in backbones:
        pal = PALETTE[bb]
        handles.append(mpatches.Patch(color=pal["acc"]))
        labels_leg.append(f"{bb} Acc")
        handles.append(mpatches.Patch(color=pal["f1"]))
        labels_leg.append(f"{bb} F1")

    fig.legend(
        handles=handles,
        labels=labels_leg,
        loc="upper center",
        ncol=4,
        frameon=True,
        edgecolor=SPINE_COLOR,
        fancybox=False,
        fontsize=FS * 0.88,
        handlelength=1.1,
        handleheight=0.85,
        columnspacing=1.0,
        handletextpad=0.5,
        bbox_to_anchor=(0.54, 0.995),
    )

    for ext in ("pdf", "png"):
        path = f"main_results{scheme_key}.{ext}"
        fig.savefig(path, bbox_inches="tight", dpi=300 if ext == "png" else 150)
    plt.close(fig)
    print(f"Saved main_results{scheme_key}.pdf / .png")


generate("1")
print("All done!")
