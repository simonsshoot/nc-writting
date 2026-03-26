import numpy as np
import matplotlib.pyplot as plt

# 1. 整理实验数据
domains = ['Application', 'Finance', 'IoT', 'Program', 'Web']
models = ['DeepSeek', 'GPT-4o', 'Claude', 'Gemini']

data = {
    'DeepSeek': {
        'origin': [26.06, 21.62, 43.24, 44.78, 26.47],
        'optim':   [ 5.21,  7.21, 13.51,  8.20,  5.88]
    },
    'GPT-4o': {
        'origin': [12.89, 15.89, 50.00, 18.33, 27.27],
        'optim':   [ 2.79,  1.87, 12.50,  5.00, 12.12]
    },
    'Claude': {
        'origin': [11.88,  6.73, 11.29,  7.35,  5.00],
        'optim':   [ 4.95,  4.81,  8.06,  4.49,  5.00]
    },
    'Gemini': {
        'origin': [ 9.57, 26.19, 17.24, 11.76, 13.33],
        'optim':   [ 4.26, 11.90, 13.79,  5.88, 10.00]
    }
}

# 2. 参数
n_domains = len(domains)
# 均匀角度位置（每个扇区的中间角度）
angles = np.linspace(0, 2 * np.pi, n_domains, endpoint=False)

# 颜色配置
color_origin = '#A8DADC' # 浅青色（外层）
color_optim  = '#457B9D' # 深蓝色（里层）

# 选择拉伸策略：把 [0,10,max_val] 映射到 [0, 10*stretch, max_val]
stretch = 2.0  # 把 0-10 区间放大为 0-20；可以调整为 1.5 / 3.0 等试试
# 计算全局最大值（用于映射）
all_vals = []
for m in models:
    all_vals += data[m]['origin'] + data[m]['optim']
max_val = max(all_vals)

def scale_r(r):
    # 分段线性映射： 0 -> 0, 10 -> 10*stretch, max_val -> max_val
    return np.interp(r, [0.0, 10.0, max_val], [0.0, 10.0 * stretch, max_val])

# 3. 绘图
fig, axes = plt.subplots(1, 4, figsize=(20, 6), subplot_kw={'projection': 'polar'})
plt.subplots_adjust(wspace=0.4)

for i, model_name in enumerate(models):
    ax = axes[i]

    origin_vals = np.array(data[model_name]['origin'])
    optim_vals  = np.array(data[model_name]['optim'])

    # 宽度与间隔（角度上保留一定间隙）
    width = (2 * np.pi) / n_domains * 0.8

    # 将值映射到变换后的半径
    origin_scaled = scale_r(origin_vals)
    optim_scaled  = scale_r(optim_vals)

    # 绘制两层柱状（origin 在外层，optim 在内层）
    bars_origin = ax.bar(angles, origin_scaled, width=width,
                         bottom=0.0, color=color_origin, edgecolor='white', linewidth=1, alpha=0.9,
                         label='Origin')
    bars_optim = ax.bar(angles, optim_scaled, width=width,
                        bottom=0.0, color=color_optim, edgecolor='white', linewidth=1, alpha=0.95,
                        label='Optim')

    # 设置 12 点为起点，顺时针方向
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    # 设置角度刻度标签
    ax.set_xticks(angles)
    ax.set_xticklabels(domains, fontsize=10, fontweight='bold')

    # 隐藏极坐标边框（去掉外层闭合圆圈）
    try:
        ax.spines['polar'].set_visible(False)
    except Exception:
        # 兼容性：某些 matplotlib 版本的极坐标边框处理不同
        ax.set_frame_on(False)

    # 设置径向刻度：我们要显示原始数值，但位置需要映射
    # 选取合适的原始刻度（确保包含 0 和 10）
    raw_ticks = np.unique(np.concatenate([
        np.array([0, 5, 10]),
        np.linspace(0, max_val, 5)
    ]))
    raw_ticks = np.sort(raw_ticks)
    # 只保留 <= max_val 的刻度并尽量不太多
    raw_ticks = raw_ticks[raw_ticks <= max_val]

    tick_positions = scale_r(raw_ticks)
    ax.set_yticks(tick_positions)
    # 显示原始数值作为标签（便于读数），格式化为不多于2位小数
    ax.set_yticklabels([f'{t:.0f}' if t >= 1 else f'{t:.2f}' for t in raw_ticks])

    # 设置当前半径显示范围为变换后的最大值
    ax.set_ylim(0, scale_r(max_val))

    # 网格样式
    ax.yaxis.grid(True, linestyle='--', alpha=0.4)
    ax.xaxis.grid(False)

    # 仅在最后一个子图添加图例
    if i == len(models) - 1:
        ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.05))

plt.show()