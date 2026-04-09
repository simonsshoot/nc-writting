import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# --- 1. 读取并清洗数据 ---
file_path = "main_result.xlsx"
df = pd.read_excel(file_path, header=[0, 1])

# 处理 llm 列的 'none' 问题并向下填充
df.iloc[:, 0] = df.iloc[:, 0].replace(["none", "None"], np.nan).ffill()

llms_list = ["DeepSeek", "GPT-4o", "Claude", "Gemini"]
benchmarks = ["Rjudge", "Assebench", "Agentharm"]
metrics = ["Accuracy", "Precision", "Recall", "F1"]
frameworks = ["CoTSafe", "ShieldAgent", "AGrail", "GuardAgent", "EVOLVE"]

# --- 2. 全局样式 ---
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.linewidth": 0.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "#CCCCCC",
        "grid.color": "#EBEBEB",
        "grid.linewidth": 0.5,
        "xtick.major.size": 0,
        "ytick.major.size": 2,
        "ytick.major.width": 0.5,
        "legend.frameon": False,
    }
)

# 4 种指标的颜色（与原热力图的蓝绿基调呼应）
METRIC_COLORS = ["#2B7BB9", "#27A87A", "#E08B2A", "#C0415A"]

# --- 3. 布局：3 行（benchmark）× 4 列（LLM） ---
fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(22, 12))
plt.subplots_adjust(
    left=0.05, right=0.98, bottom=0.08, top=0.94, wspace=0.04, hspace=0.15
)

n_fw = len(frameworks)
n_met = len(metrics)
group_w = 0.75
bar_w = group_w / n_met
offsets = np.linspace(-(group_w - bar_w) / 2, (group_w - bar_w) / 2, n_met)
x = np.arange(n_fw)

for row_idx, bench in enumerate(benchmarks):
    for col_idx, llm in enumerate(llms_list):
        ax = axes[row_idx, col_idx]

        # 筛选数据
        current_df = df[df.iloc[:, 0].astype(str).str.lower() == llm.lower()]
        current_df = current_df.set_index(current_df.columns[1])
        bench_key = bench.lower()
        try:
            plot_data = current_df.loc[frameworks, bench_key][metrics].astype(float)
        except KeyError:
            ax.axis("off")
            continue

        # 绘制分组柱状图
        ax.set_axisbelow(True)
        ax.yaxis.grid(True, linestyle="--", alpha=0.7)
        ax.set_facecolor("white")

        all_vals = []
        for mi, metric in enumerate(metrics):
            vals = plot_data[metric].values
            all_vals.extend(vals)
            ax.bar(
                x + offsets[mi],
                vals,
                width=bar_w * 0.88,
                color=METRIC_COLORS[mi],
                alpha=0.88,
                linewidth=0,
                zorder=3,
            )

        # y 轴范围：留出上下空间
        ymin = max(0.0, min(all_vals) - 0.06)
        ymax = min(1.02, max(all_vals) + 0.08)
        ax.set_ylim(ymin, ymax)

        # --- 坐标轴简化（与原热力图保持一致的显示逻辑）---
        ax.set_xticks(x)
        if row_idx == 2:
            ax.set_xticklabels(frameworks, rotation=25, ha="right", fontsize=8.5)
        else:
            ax.set_xticklabels([])

        ax.tick_params(axis="y", labelsize=8.5)

        if col_idx == 0:
            ax.set_ylabel(bench, fontsize=13, fontweight="bold", labelpad=15)
        else:
            ax.set_yticklabels([])
            ax.set_ylabel("")

        if row_idx == 0:
            ax.set_title(llm, fontsize=15, fontweight="bold", pad=20)

        ax.spines["left"].set_color("#CCCCCC")
        ax.spines["bottom"].set_color("#CCCCCC")

# --- 4. 共享图例（指标颜色说明）---
patches = [
    mpatches.Patch(color=METRIC_COLORS[i], label=metrics[i]) for i in range(n_met)
]
fig.legend(
    handles=patches,
    loc="upper center",
    ncol=n_met,
    bbox_to_anchor=(0.5, 0.995),
    columnspacing=1.6,
    handlelength=1.2,
    handleheight=0.9,
    fontsize=11,
)

# --- 5. 保存 ---
plt.savefig("main_result.png", dpi=300, bbox_inches="tight", pad_inches=0.1)
plt.savefig(
    "main_result.pdf",
    format="pdf",
    bbox_inches="tight",
    facecolor="white",
    edgecolor="none",
)
print("Saved: main_result.png / main_result.pdf")
