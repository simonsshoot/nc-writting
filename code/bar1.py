import matplotlib.pyplot as plt
import numpy as np
import matplotlib
matplotlib.use('MacOSX')  # 或者尝试使用其他后端，如 'Agg', 'MacOSX', 'Qt5Agg' 等
matplotlib.rcParams['font.family'] = 'Arial'


# 数据
datasets = ['SHS', 'LSAT', 'GRE', 'Codenames', 'Essentiality', 'Physics']
dp_v3_scores = [77.32, 88.69, 87.95, 68.24, 83.58, 69.81]
dp_r1_scores = [90.72, 95.24, 92.17, 70.59, 88.60, 83.02]
ours_scores = [87.63, 91.07, 89.76, 68.24, 89.55, 84.91]

# 误差线（根据图中的误差线估算）
dp_v3_errors = [4, 5, 3, 6, 5, 5]
dp_r1_errors = [2, 3, 2, 2, 3, 2]
ours_errors = [3, 5, 3, 4, 5, 3]


# # 数据
# datasets = ['SHS', 'LSAT', 'GRE', 'Codenames', 'Essentiality', 'Physics']
# gpt_4o_scores = [64.94, 78.57, 70.78, 69.41, 65.67, 52.83]
# gpt_o1_scores = [81.44, 76.79, 75.30, 78.82, 74.63, 75.47]
# ours_scores = [75.26, 86.90, 84.94, 75.29, 74.63, 69.81]
#
# # 误差线（根据图中的误差线估算）
# gpt_4o_errors = [6, 3, 4, 5, 5, 5]
# gpt_o1_errors = [3, 2, 2, 2, 3, 3]
# ours_errors = [5, 2, 3, 3, 4, 3]

# percentage_correct_data = [
#     [64.94, 81.44, 75.26],  # 子图 1
#     [78.57, 76.79, 86.9],  # 子图 2
#     [70.78, 75.3, 84.94],  # 子图 3
#     [69.41, 78.82, 75.29],  # 子图 4
#     [65.67, 74.63, 74.63],  # 子图 5
#     [52.83, 75.47, 69.81]  # 子图 6
# ]
# # 为每个子图分别设置不同的误差条数据a
# error_data = [
#     [6, 3, 5],  # 子图 1
#     [3, 2, 2],  # 子图 2
#     [4, 2, 3],  # 子图 3
#     [5, 2, 3],  # 子图 4
#     [5, 3, 4],  # 子图 5
#     [5, 3, 3]  # 子图 6
# ]

# 创建图表
fig, axes = plt.subplots(2, 3, figsize=(10,5))
fig.subplots_adjust(hspace=0.35, wspace=0.3)

# 颜色设置
# colors = ['#C0C0C0', '#7B9CC4', '#D99999']  # 灰色、蓝色、红色
colors = ['#B8D4B8', '#7B9CC4', '#E6B566']  # 淡绿色、蓝色（不变）、橙黄色
bar_width = 0.5

# 绘制每个子图
for idx, ax in enumerate(axes.flat):
    dataset = datasets[idx]
    scores = [dp_v3_scores[idx], ours_scores[idx], dp_r1_scores[idx]]
    errors = [dp_v3_errors[idx], ours_errors[idx], dp_r1_errors[idx]]
    # scores = [gpt_4o_scores[idx], ours_scores[idx], gpt_o1_scores[idx]]
    # errors = [gpt_4o_errors[idx], ours_errors[idx], gpt_o1_errors[idx]]
    # labels = ['GPT-4o',  'Ours','GPT-o1']
    labels = ['DP-v3',  'Ours','DP-r1']

    # 绘制柱状图
    x_pos = np.arange(len(labels))
    bars = ax.bar(x_pos, scores, bar_width,
                  color=colors,
                  edgecolor='black',
                  linewidth=1.2,
                  yerr=errors,
                  capsize=5,
                  error_kw={'linewidth': 1.5, 'ecolor': 'black'})

    # 在柱子上方添加数值标签
    for i, (bar, score) in enumerate(zip(bars, scores)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height + errors[i] + 2,
                f'{score:.2f}%',
                ha='center', va='bottom', fontsize=12, fontweight='bold')

    # 设置标题和标签
    ax.set_title(dataset, fontsize=15, fontweight='bold', pad=10)
    ax.set_ylabel('Accuracy (%)', fontsize=14)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, fontsize=14)

    # 设置y轴范围
    ax.set_ylim(0, 120)

    # ax.set_yticks(np.arange(0, 121, 20))
    # ax.set_yticks(np.arange(0, 101, 20))  # 设置 y 轴的位置
    ax.set_yticklabels([f'{i}' for i in np.arange(0, 121, 20)], fontsize=14)

    # 设置网格
    ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.5)
    ax.set_axisbelow(True)

    # 设置边框
    for spine in ax.spines.values():
        spine.set_linewidth(1.2)
        spine.set_color('black')

# 调整布局
plt.tight_layout()
plt.savefig('model_comparison.pdf')
plt.show()