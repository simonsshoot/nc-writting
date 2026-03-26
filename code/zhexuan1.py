import os
import matplotlib.pyplot as plt
import numpy as np

plt.switch_backend('pgf')
plt.rc('font', family='Times New Roman')

def auto_label(rects, fontofzi, family_zi, flag=False):
    ax = plt.gca()  # Get current axis
    for rect in rects:
        height = round(rect.get_height(), 1)
        _dis = -2
        ax.annotate(
            '{}'.format(height),
            xy=(rect.get_x() + rect.get_width() / 2, height),
            xytext=(0, _dis),
            textcoords="offset points",
            ha='center',
            va='bottom',
            fontsize=20,
            family=family_zi,
        )

def getdata():
    GPT = [35.61, 32.80, 52.21, 42.36]
    QWEN = [38.12, 34.95, 47.83, 40.65]
    LLAVA = [45.47, 42.45, 45.67, 38.12]
    GEMINI = [38.23, 33.76, 49.43, 41.45]
    
    labels = ['GAR $\downarrow$', 'QAR $\downarrow$', 'GS $\\uparrow$', 'QS $\\uparrow$']
    dict_data = {'GPT-4o': np.array(GPT), 'Qwen-3': np.array(QWEN), 'LLaVA': np.array(LLAVA), 'Gemini': np.array(GEMINI)}
    return labels, dict_data

def draw_fig_adv():
    family_zi = 'Times New Roman'
    colors = ['#9ECDF0', "#8CD8B5", "#FCE3A4", '#CCA9D5']
    labels, dict_data = getdata()
    index = np.arange(len(labels))
    ALPHA = 1.0
    FONTSIZE = 28
    fontofzi = FONTSIZE
    width = 0.2

    plt.figure(figsize=(7, 6))
    rects = []

    for ind, bitname in enumerate(dict_data):
        data = dict_data[bitname]
        rect = plt.bar(
            index + ind * width - width * len(dict_data) / 2 + width / 2,
            data,
            color=colors[ind % len(colors)],
            label=bitname,
            width=width,
            alpha=ALPHA
        )
        rects.append(rect)

    plt.xticks(ticks=index, labels=labels, fontsize=FONTSIZE, family=family_zi, fontweight='bold')
    my_y_ticks = np.arange(0,61,10)
    plt.yticks(my_y_ticks, fontsize=FONTSIZE, family=family_zi, fontweight='bold')
    # Make the y-axis label text bold
    # plt.ylabel('Metrics (\%)', fontsize=FONTSIZE, family=family_zi, fontweight='bold', labelpad=5)
    # plt.xlabel('Metrics', fontsize=FONTSIZE, family=family_zi, fontweight='bold', labelpad=0)

    for rect in rects:
        auto_label(rect, fontofzi, family_zi)

    legend_properties = {'family': family_zi, 'size': 20}
    plt.legend(loc='lower right', frameon=True, ncol=2, prop=legend_properties)
    plt.grid(True, linestyle='--', which='major', color='grey', alpha=0.5)
    plt.tight_layout()

    plt.savefig('./table1.svg', bbox_inches='tight', dpi=300)
    plt.savefig('./table1.png', bbox_inches='tight')
    plt.savefig('./table1.pdf', format='pdf', bbox_inches='tight')

if __name__ == '__main__':
    draw_fig_adv()
