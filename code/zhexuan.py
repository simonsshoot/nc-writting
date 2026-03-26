import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import os
import pandas as pd

plt.rc('font', family='Times New Roman', size=25)  # 设置字体字号
plt.rcParams['mathtext.fontset'] = 'stix'
plt.figure(figsize=(7,6))

#背景横线
plt.grid(axis="y",zorder = 1,linestyle = '--')
plt.grid(axis="x",zorder = 1,linestyle = '--')

x_labels = ['1', '2', '3', '4', '5', '6']
tick_step=2

ticks = np.arange(len(x_labels)) * tick_step


method_name = 'zhexian1'
# method_name = 'UFLD'
# 读取Excel文件
df = pd.read_excel(f'{method_name}.xlsx')
# 转置数据
df = df.set_index('Model').T

models = ['GAR_n', 'GS_n', 'GAR_k', 'GS_k']
data = []
for model in models:
    model_data = df[model].tolist()
    # print(model_data)
    data.append(model_data)
# mobi_as = ans[3]
# clp_h = [90.45,47.56,25.79,9.5]

# dense_as = ans[4]
# slice_h = [0.01,2.78,9.15,32.8]

# google_as = ans[5]
# deepmufl_mu100_h = [0.21, 2.24,5.24,23.46]

line_1, = plt.plot(ticks,data[0],color='#3498db',lw=3.0,label='GAR$_n$',marker = 'x',zorder = 10, linestyle=':')
line_2, = plt.plot(ticks,data[1],color='#3498db',lw=2.0,label='GS$_n$',marker = 'x',zorder = 10,linestyle='-')
line_3, = plt.plot(ticks,data[2],color='#e74c3c',lw=3.0,label='GAR$_k$',marker = 'v',zorder = 10,linestyle=':')
line_4, = plt.plot(ticks,data[3],color='#e74c3c',lw=2.0,label='GS$_k$',marker = 'v',zorder = 10,linestyle='-')
# line_5, = plt.plot(ticks,data[4],color='#a4b9c7',lw=2.0,label=models[4],marker = '.',zorder = 10,linestyle='-')
# line_6, = plt.plot(ticks,data[5],color='#e67e22',lw=2.0,label=models[5],marker = '*',zorder = 10,linestyle='-')
# line_7, = plt.plot(ticks,data[6],color='#e67e22',lw=2.0,label=models[6],marker = 'h',zorder = 10,linestyle='-')

plt.axvline(x=ticks[3], color='#3498db', linestyle='dashdot', linewidth=2)
plt.axvline(x=ticks[1], color='#e74c3c', linestyle='dashdot', linewidth=2)

# line_zhexian1r, = plt.plot(ticks,zhexian1r,color='#a4b9c7',lw=3.0,label='FP-HIT',marker = 's',zorder = 10)
# line_nc_h, = plt.plot(ticks,nc_h,color='#f7d8e9',lw=3.0,label='NC-HIT',marker = '+',zorder = 10)
# line_anp_h, = plt.plot(ticks,anp_h,color='#badc58',lw=3.0,label='ANP-HIT',marker = 'p',zorder = 10)
# line_clp_h, = plt.plot(ticks,clp_h,color='#d7bde2',lw=3.0,label='CLP-HIT',marker = ',',zorder = 10)
# line_deepmufl_mu100_h, = plt.plot(ticks,deepmufl_mu100_h,color='#fdebd0',lw=3.0,label='deepmufl-HIT',marker = 'h',zorder = 10)
# line_slice_h, = plt.plot(ticks,slice_h,color='#d2aa7f',lw=3.0,label='SLICE-HIT',marker = '_',zorder = 10)

#横坐标的操作
plt.xticks(ticks, x_labels,family='Times New Roman', rotation=0, fontsize=24, fontweight='bold')

#纵坐标的操作
my_y_ticks = np.arange(20, 61, 5)
plt.yticks(my_y_ticks, family='Times New Roman', fontsize=24, fontweight='bold')

#标题
# plt.xlabel("Layer",fontsize = 17)
# plt.ylabel("Neuron Sensitivity",fontsize = 17)
# plt.ylabel("Neuron Uncertainty",fontsize = 17)
# plt.xlabel(r'Rotation Angle ($\circ$)', fontweight='bold', fontsize=25, family='Times New Roman')
# plt.xlabel(r'k/n', fontweight='bold', fontsize=25, family='Times New Roman')
# plt.ylabel(r'Metrics (%)', fontweight='bold', family='Times New Roman')
# plt.title("Black-Box Attack Defense")

#纵坐标换为百分数
#plt.gca().yaxis.set_major_formatter(FuncFormatter(to_percent))

#图注

# plt.legend(handles=[line_fp_w, line_nc_w,line_anp_w,line_clp_w,line_deepmufl_mu100_w,line_slice_w],labels=['FP-WJI','NC-WJI','ANP-WJI','CLP-WJI','deepmufl-WJI','SLICE-WJI'],loc='upper center',fontsize=20)
# plt.legend(handles=[line_fp_h, line_nc_h,line_anp_h,line_clp_h,line_deepmufl_mu100_h,line_slice_h],labels=['FP-HIT','NC-HIT','ANP-HIT','CLP-HIT','deepmufl-HIT','SLICE-HIT'],loc='upper right',fontsize=20)
plt.legend(ncol=2, fontsize=20, loc='lower right')

# plt.show()
# plt.savefig("defence-lr0.0001.pdf", format="pdf", bbox_inches="tight")

fname = os.path.join('./', 'zhexian1.svg')
fname_png = os.path.join('./', 'zhexian1.png')
fname_pdf = os.path.join('./', 'zhexian1.pdf')
plt.savefig(fname, bbox_inches='tight', dpi=300)
plt.savefig(fname_png, bbox_inches='tight')
plt.savefig(fname_pdf, format="pdf", bbox_inches="tight")