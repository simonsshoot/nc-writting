import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

# --- 1. 读取并清洗数据 ---
file_path = "main_result.xlsx"
df = pd.read_excel(file_path, header=[0, 1])

# 处理 llm 列的 'none' 问题并向下填充，解决多余 ShieldAgent 问题
df.iloc[:, 0] = df.iloc[:, 0].replace(["none", "None"], np.nan).ffill()

llms_list = ["DeepSeek", "GPT-4o", "Claude", "Gemini"]
benchmarks = ["Rjudge", "Assebench", "Agentharm"]
metrics = ["Accuracy", "Precision", "Recall", "F1"]
frameworks = ["CoTSafe", "ShieldAgent", "AGrail", "GuardAgent", "EVOLVE"]

# --- 2. 设置非线性映射与配色 ---
# gamma 越高，高分段 (0.6-1.0) 的颜色变化越剧烈，区分度越大
gamma_val = 3
norm = mcolors.PowerNorm(gamma=gamma_val, vmin=0.0, vmax=1.0)
cmap = "GnBu"  # 蓝绿配色

# --- 3. 开始绘图 ---
fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(22, 12))
"""
left=0.06 # 左侧留出6%的空间用于y轴标签
right=0.88 # 右侧留出12%的空间用于颜色条
bottom=0.08 # 底部留出8%的空间用于x轴标签
top=0.92 # 顶部留出8%的空间用于子图标题
wspace=0.08 # 列间距为子图宽度的8%
hspace=0.1 # 行间距为子图高度的10%
"""
plt.subplots_adjust(
    left=0.05, right=0.98, bottom=0.08, top=0.94, wspace=0.04, hspace=0.15
)

for row_idx, bench in enumerate(benchmarks):
    for col_idx, llm in enumerate(llms_list):
        ax = axes[row_idx, col_idx]

        # 筛选数据
        current_df = df[df.iloc[:, 0].astype(str).str.lower() == llm.lower()]
        current_df = current_df.set_index(current_df.columns[1])

        bench_key = bench.lower()
        try:
            # 确保按指定 frameworks 顺序读取
            plot_data = current_df.loc[frameworks, bench_key][metrics].astype(float)
        except KeyError:
            continue

        # 绘制热力图
        # cbar_ax: 我们稍后统一处理图例，或者在最后一列处理
        im = sns.heatmap(
            plot_data,
            annot=True,
            fmt=".4f",
            cmap=cmap,
            norm=norm,
            ax=ax,
            cbar=(col_idx == 3),
            annot_kws={"size": 10},
        )

        # --- 修复 Colorbar 重叠问题 ---
        if col_idx == 3:
            cbar = ax.collections[0].colorbar
            # 手动设置刻度，避开被压缩的 0.2, 0.4 段，重点展示高分段
            ticks = [0, 0.6, 0.8, 0.9, 1.0]
            cbar.set_ticks(ticks)
            cbar.set_ticklabels([f"{t:.1f}" for t in ticks])
            cbar.ax.tick_params(labelsize=10)

        # --- 坐标轴简化 ---
        if col_idx == 0:
            ax.set_ylabel(f"{bench}", fontsize=13, fontweight="bold", labelpad=15)
            ax.set_yticklabels(frameworks, rotation=0)
        else:
            ax.set_ylabel("")
            ax.set_yticklabels([])

        if row_idx == 2:
            ax.set_xticklabels(metrics, rotation=0)
        else:
            ax.set_xticklabels([])

        if row_idx == 0:
            ax.set_title(f"{llm}", fontsize=15, fontweight="bold", pad=20)
        else:
            ax.set_title("")

# 保存
plt.savefig("main_result.png", dpi=300, bbox_inches="tight", pad_inches=0.1)
plt.savefig(
    "main_result.pdf",  # PDF文件名
    format="pdf",  # 明确指定格式
    bbox_inches="tight",  # 裁剪白边
    facecolor="white",  # 背景色
    edgecolor="none",  # 无边框
)
plt.show()
