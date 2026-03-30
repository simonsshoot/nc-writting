import matplotlib.pyplot as plt
import numpy as np

# Use slightly larger fonts to match body text size.
plt.rcParams.update(
    {
        "font.size": 14,
        "axes.titlesize": 15,
        "axes.labelsize": 13,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
        "legend.fontsize": 13,
    }
)

# --- 1. 数据准备 (基于上传图片提取的百分比数据) ---
categories = ["Application", "Finance", "IoT", "Program", "Web"]
models = ["DeepSeek", "GPT-4o", "Claude", "Gemini"]

# Need / No Need 数据 (%)
data_need = {
    "DeepSeek": [69.44, 50.00, 83.33, 60.16, 48.57],
    "GPT-4o": [60.71, 46.83, 43.33, 53.91, 51.43],
    "Claude": [72.62, 38.53, 86.67, 77.78, 62.86],
    "Gemini": [61.90, 34.13, 66.67, 54.69, 42.86],
}

# New / Reused 数据 (%)
data_new = {
    "DeepSeek": [51.44, 43.16, 65.38, 20.33, 56.00],
    "GPT-4o": [14.34, 19.57, 75.00, 52.04, 57.14],
    "Claude": [58.42, 71.15, 75.81, 68.57, 56.67],
    "Gemini": [30.06, 25.00, 46.43, 40.79, 41.38],
}

# --- 2. 绘图配置 ---
fig, axes = plt.subplots(2, 4, figsize=(24, 12), sharey=True)
plt.subplots_adjust(wspace=0.08, hspace=0.15, top=0.94, left=0.05, right=0.98)


def plot_bar(ax, cat_labels, val1, val2, title, color1, color2, label1, label2):
    x = np.arange(len(cat_labels))
    width = 0.4

    rects1 = ax.bar(
        x - width / 2, val1, width, label=label1, color=color1, edgecolor="white"
    )
    rects2 = ax.bar(
        x + width / 2, val2, width, label=label2, color=color2, edgecolor="white"
    )

    ax.set_title(title, fontsize=16, fontweight="bold", pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(cat_labels, rotation=30, fontsize=16)
    ax.set_ylim(0, 110)
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    ax.tick_params(axis="y", labelsize=14)

    # 添加数值标签
    for rect in rects1 + rects2:
        height = rect.get_height()
        ax.annotate(
            f"{height:.1f}%",
            xy=(rect.get_x() + rect.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=14,
        )


# --- 3. 循环生成子图 ---
for i, model in enumerate(models):
    # 第一行：Need vs No Need
    need_vals = data_need[model]
    noneed_vals = [100 - v for v in need_vals]  # 计算对应的 No Need 比例
    plot_bar(
        axes[0, i],
        categories,
        need_vals,
        noneed_vals,
        f"{model}: Need vs No Need",
        "#2E5A88",
        "#A8DADC",
        "Need",
        "No Need",
    )
    axes[0, i].tick_params(axis="x", labelbottom=False)

    # 第二行：New vs Reused
    new_vals = data_new[model]
    reused_vals = [
        100 - v for v in new_vals
    ]  # 注：图片中部分数据不加和为100，此处以New数据为准计算对比
    plot_bar(
        axes[1, i],
        categories,
        new_vals,
        reused_vals,
        f"{model}: New vs Reused",
        "#D85C27",
        "#FFB385",
        "New",
        "Reused",
    )

# --- 4. 全局修饰 ---
axes[0, 0].set_ylabel("Percentage (%)", fontsize=14, fontweight="bold")
axes[1, 0].set_ylabel("Percentage (%)", fontsize=14, fontweight="bold")
axes[0, 3].legend(loc="upper right")
axes[1, 3].legend(loc="upper right")

# 保存并显示
plt.savefig("reuse_result.png", dpi=300, bbox_inches="tight")
plt.savefig(
    "reuse_result.pdf",  # PDF文件名
    format="pdf",  # 明确指定格式
    bbox_inches="tight",  # 裁剪白边
    facecolor="white",  # 背景色
    edgecolor="none",  # 无边框
)
plt.show()
