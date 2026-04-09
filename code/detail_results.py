import numpy as np
import matplotlib.pyplot as plt

# ======================
# 1. 数据（从你图片手动整理）
# ======================
domains = ["Application", "Program", "IoT", "Finance", "Web"]

data = {
    "GuardAgent": [0.4409, 0.3014, 0.3636, 0.8052, 0.3],
    "AGrail": [0.4316, 0.3765, 0.5517, 0.6667, 0.0],
    "CoTSafe": [0.6667, 0.3659, 0.5714, 0.5882, 0.6667],
    "EVOLVE": [0.9393, 0.9286, 0.8333, 0.8593, 0.8649],
}

# ======================
# 2. 画图
# ======================
plt.figure(figsize=(7, 4))

# 缩短横轴间距
x = np.arange(len(domains)) * 0.8

main_color = "#1b9e77"  # EVOLVE 绿色
baseline_color = "#bdbdbd"

for method, values in data.items():
    if method == "EVOLVE":
        plt.plot(
            x,
            values,
            marker="o",
            linewidth=2.8,
            markersize=11,
            color=main_color,
            label="EVOLVE",
            zorder=3,
        )
    else:
        plt.plot(
            x,
            values,
            marker="o",
            linestyle="--",
            linewidth=1.6,
            markersize=9,
            color=baseline_color,
            alpha=0.9,
            label=method,
            zorder=2,
        )

# ======================
# 3. 美化（论文风）
# ======================
plt.ylabel("F1", fontsize=12)
plt.xticks(x, domains, fontsize=11)
plt.yticks(fontsize=11)

# 去掉上/右边框
ax = plt.gca()
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# 轻网格
plt.grid(axis="y", linestyle="--", alpha=0.3)

# 图例（右侧）
plt.legend(frameon=False, fontsize=10)

# 防止 warning
plt.subplots_adjust(left=0.08, right=0.98, top=0.88, bottom=0.15)

plt.show()
