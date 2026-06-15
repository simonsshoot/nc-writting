import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ── Global style (matching main_result_final.py) ──────────────────────────────
FS = 12.5
plt.rcParams.update({
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
})

SPINE_COLOR = "#AAAAAA"
GRID_COLOR = "#E8E8E8"
TICK_COLOR = "#444444"
LABEL_COLOR = "#222222"

# ═══════════════════════════════════════════════════════════════════════════════
# Figure 1: Detail Domains (5 domains × 4 methods)
# ═══════════════════════════════════════════════════════════════════════════════
domains = ["App.", "Prog.", "IoT", "Fin.", "Web"]
methods_dom = ["GuardAgent", "AGrail", "CoTSafe", "EVOLVE"]

data_dom = {
    "GuardAgent": [0.4409, 0.3014, 0.3636, 0.8052, 0.3],
    "AGrail":     [0.4316, 0.3765, 0.5517, 0.6667, 0.0],
    "CoTSafe":    [0.6667, 0.3659, 0.5714, 0.5882, 0.6667],
    "EVOLVE":     [0.9393, 0.9286, 0.8333, 0.8593, 0.8649],
}

# Blue gradient from main_result_final.py (#4E79A7): lightest → darkest
colors_dom = ["#9BB2C8", "#7A9DB8", "#4E79A7", "#2E5A87"]

n_methods = len(methods_dom)
n_domains = len(domains)
x = np.arange(n_domains)
bar_w = 0.18
offsets = np.array([-(1.5*bar_w + 0.02*1.5),
                    -(0.5*bar_w + 0.02*0.5),
                    (0.5*bar_w + 0.02*0.5),
                    (1.5*bar_w + 0.02*1.5)])

fig, ax = plt.subplots(figsize=(5.2, 3.2), facecolor="white")
fig.subplots_adjust(left=0.12, right=0.97, top=0.82, bottom=0.13)

for i, method in enumerate(methods_dom):
    vals = data_dom[method]
    bars = ax.bar(x + offsets[i], vals, width=bar_w,
                  color=colors_dom[i], linewidth=0, zorder=3, label=method)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{val:.3f}", ha="center", va="bottom",
                fontsize=FS * 0.82, color=LABEL_COLOR, rotation=90)

ax.set_xticks(x)
ax.set_xticklabels(domains, fontsize=FS * 0.90, color=TICK_COLOR, fontweight="bold")
ax.set_ylim(0, 1.18)
ax.yaxis.set_major_locator(plt.MultipleLocator(0.2))
ax.set_ylabel("F1", fontsize=FS * 0.92, color=LABEL_COLOR, labelpad=4)

# Grid & spines
ax.set_axisbelow(True)
ax.yaxis.grid(True, color=GRID_COLOR, linewidth=0.55, zorder=0)
ax.set_facecolor("white")
for sp in ["left", "bottom"]:
    ax.spines[sp].set_visible(True)
    ax.spines[sp].set_color(SPINE_COLOR)
    ax.spines[sp].set_linewidth(0.7)
for sp in ["top", "right"]:
    ax.spines[sp].set_visible(False)
ax.tick_params(axis="x", length=3.0, width=0.7, colors=TICK_COLOR)
ax.tick_params(axis="y", length=3.0, width=0.7, colors=TICK_COLOR, labelsize=FS * 0.90)

# Legend
ax.legend(loc="upper center", ncol=4, frameon=True, edgecolor=SPINE_COLOR,
          fancybox=False, fontsize=FS * 0.72, bbox_to_anchor=(0.5, 1.18),
          columnspacing=0.8, handlelength=1.0)

for ext in ("pdf", "png"):
    fig.savefig(f"detail_domains_bar.{ext}", dpi=300, bbox_inches="tight", facecolor="white")
    print(f"Saved → detail_domains_bar.{ext}")
plt.close()

# ═══════════════════════════════════════════════════════════════════════════════
# Figure 2: Ablation Study (4 metrics × 4 variants)
# ═══════════════════════════════════════════════════════════════════════════════
metrics = ["Acc.", "Prec.", "Rec.", "F1"]
variants = ["w/o Analysis.", "w/o Fusion.", "w/o Auditor.", "Full EVOLVE"]

data_abl = {
    "w/o Analysis.": [0.7846, 0.5409, 0.7679, 0.6355],
    "w/o Fusion.":   [0.8932, 0.8800, 0.8963, 0.8881],
    "w/o Auditor.":  [0.8581, 0.8733, 0.8528, 0.8630],
    "Full EVOLVE":   [0.9107, 0.8982, 0.9148, 0.9065],
}

# Red gradient from main_result_final.py (#B05252): lightest → darkest
colors_abl = ["#CFA0A0", "#C07878", "#B05252", "#8A3333"]

n_variants = len(variants)
n_metrics = len(metrics)
x2 = np.arange(n_metrics)

fig2, ax2 = plt.subplots(figsize=(5.2, 3.2), facecolor="white")
fig2.subplots_adjust(left=0.12, right=0.97, top=0.82, bottom=0.13)

for i, variant in enumerate(variants):
    vals = data_abl[variant]
    bars = ax2.bar(x2 + offsets[i], vals, width=bar_w,
                   color=colors_abl[i], linewidth=0, zorder=3, label=variant)
    for bar, val in zip(bars, vals):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                 f"{val:.3f}", ha="center", va="bottom",
                 fontsize=FS * 0.82, color=LABEL_COLOR, rotation=90)

ax2.set_xticks(x2)
ax2.set_xticklabels(metrics, fontsize=FS * 0.90, color=TICK_COLOR, fontweight="bold")
ax2.set_ylim(0.45, 1.08)
ax2.yaxis.set_major_locator(plt.MultipleLocator(0.1))
ax2.set_ylabel("Score", fontsize=FS * 0.92, color=LABEL_COLOR, labelpad=4)

# Grid & spines
ax2.set_axisbelow(True)
ax2.yaxis.grid(True, color=GRID_COLOR, linewidth=0.55, zorder=0)
ax2.set_facecolor("white")
for sp in ["left", "bottom"]:
    ax2.spines[sp].set_visible(True)
    ax2.spines[sp].set_color(SPINE_COLOR)
    ax2.spines[sp].set_linewidth(0.7)
for sp in ["top", "right"]:
    ax2.spines[sp].set_visible(False)
ax2.tick_params(axis="x", length=3.0, width=0.7, colors=TICK_COLOR)
ax2.tick_params(axis="y", length=3.0, width=0.7, colors=TICK_COLOR, labelsize=FS * 0.90)

# Legend
ax2.legend(loc="upper center", ncol=4, frameon=True, edgecolor=SPINE_COLOR,
           fancybox=False, fontsize=FS * 0.72, bbox_to_anchor=(0.5, 1.18),
           columnspacing=0.8, handlelength=1.0)

for ext in ("pdf", "png"):
    fig2.savefig(f"ablation_study_bar.{ext}", dpi=300, bbox_inches="tight", facecolor="white")
    print(f"Saved → ablation_study_bar.{ext}")
plt.close()
