import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FixedLocator, FixedFormatter

# 设置全局字体
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.unicode_minus'] = False

def get_ablation_data():
    labels = ['Accuracy', 'Precision', 'Recall', 'F1-score']
    # 原始数据
    dict_data = {
        'w/ AnalysisAgent': [0.7846, 0.5409, 0.7679, 0.6355],
        'w/ FusionAgent': [0.8932, 0.8800, 0.8963, 0.8881],
        'w/ AuditorAgent': [0.8581, 0.8733, 0.8528, 0.8630],
        'Full DEFEND': [0.9107, 0.8982, 0.9148, 0.9065]
    }
    return labels, dict_data

def draw_nonlinear_ablation():
    labels, dict_data = get_ablation_data()
    x = np.arange(len(labels))
    
    # 学术配色
    colors = ['#F1C40F', '#3498DB', '#2ECC71', '#E74C3C'] 
    markers = ['p', 's', '^', 'D']
    
    fig, ax = plt.subplots(figsize=(9, 7))

    # --- 核心修改：非线性纵轴映射 ---
    # 使用幂函数转换：f(y) = y^gamma。gamma越大，高分段拉伸越明显
    gamma = 4.0
    def forward(y): return np.power(y, gamma)
    def inverse(y): return np.power(y, 1/gamma)
    
    # 应用非线性缩放
    ax.set_yscale('function', functions=(forward, inverse))

    # 绘制折线
    for i, (mode_name, data) in enumerate(dict_data.items()):
        is_full = 'Full' in mode_name
        line = ax.plot(x, data, 
                        label=mode_name, 
                        color=colors[i], 
                        marker=markers[i], 
                        markersize=11, 
                        linewidth=3 if is_full else 2.5,
                        linestyle='-' if is_full else '--',
                        zorder=10 if is_full else 5, # 确保Full线在最上层
                        alpha=1.0 if is_full else 0.8)
        
        # 数值标注：只为高分段添加标注，避免重叠
        for ix, val in enumerate(data):
            ax.annotate(f'{val:.3f}', 
                        xy=(ix, val), 
                        xytext=(0, 8 if is_full else -12), 
                        textcoords="offset points",
                        ha='center', fontsize=12, 
                        fontweight='bold',
                        color=colors[i])

    # --- 坐标轴与刻度美化 ---
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=16, fontweight='bold')

    # 手动设置刻度点，展现非线性效果
    # 我们可以设置在 0.5 到 0.8 之间稀疏，0.8 到 1.0 之间密集
    tick_values = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    ax.yaxis.set_major_locator(FixedLocator(tick_values))
    ax.yaxis.set_major_formatter(FixedFormatter([f'{v:.1f}' for v in tick_values]))
    
    plt.yticks(fontsize=14, fontweight='bold')
    plt.ylabel('Ablation Performance Metrics', fontsize=18, fontweight='bold', labelpad=10)
    
    # 限制显示范围
    plt.ylim(0.5, 0.95)

    # 图例与装饰
    legend = plt.legend(loc='lower left', frameon=True, fontsize=12, edgecolor='black', ncol=1)
    legend.get_frame().set_linewidth(1.2)
    
    plt.grid(True, linestyle=':', which='both', color='grey', alpha=0.4)
    
    # 隐藏上方和右侧框架
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)

    plt.tight_layout()
    plt.savefig('ablation.png', dpi=300, bbox_inches='tight')
    plt.savefig('ablation_study.pdf', format='pdf', bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.show()

if __name__ == '__main__':
    draw_nonlinear_ablation()