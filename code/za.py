# import matplotlib
# import matplotlib.pyplot as plt
# import matplotlib.patches as mpatches
# import matplotlib.ticker as ticker
# import numpy as np

# matplotlib.rcParams["font.family"] = "DejaVu Sans"
# matplotlib.rcParams["axes.spines.top"] = False
# matplotlib.rcParams["axes.spines.right"] = False
# matplotlib.rcParams["axes.linewidth"] = 0.8
# matplotlib.rcParams["xtick.major.size"] = 0
# matplotlib.rcParams["ytick.major.size"] = 3

# # ── Data  (AgentHarm) ────────────────────────────────────────────────────────
# methods = ["CoTSafe", "ShieldAgent", "AGrail", "GuardAgent", "EVOLVE"]
# metrics = ["Acc", "Prec", "Rec", "F1"]

# data = {
#     "DeepSeek": {
#         "CoTSafe": [0.7666, 0.9007, 0.5978, 0.6915],
#         "ShieldAgent": [0.5286, 0.6923, 0.1029, 0.1791],
#         "AGrail": [0.6000, 0.8214, 0.3557, 0.4964],
#         "GuardAgent": [0.7714, 0.8992, 0.6114, 0.7281],
#         "EVOLVE": [0.8800, 0.9346, 0.8171, 0.8719],
#     },
#     "GPT-4o": {
#         "CoTSafe": [0.5857, 0.6442, 0.3829, 0.4801],
#         "ShieldAgent": [0.5286, 0.6923, 0.1029, 0.1791],
#         "AGrail": [0.5874, 1.0000, 0.1771, 0.3009],
#         "GuardAgent": [0.5829, 0.8222, 0.2114, 0.3362],
#         "EVOLVE": [0.8758, 1.0000, 0.8240, 0.9035],
#     },
# }

# # ── Palette — muted, academic ────────────────────────────────────────────────
# colors = {
#     "CoTSafe": "#6E9EC8",
#     "ShieldAgent": "#6BBAA7",
#     "AGrail": "#C8A96E",
#     "GuardAgent": "#B88BAF",
#     "EVOLVE": "#C0544F",
# }
# edge_colors = {m: ("#8B2E2B" if m == "EVOLVE" else "#4a4a4a") for m in methods}
# alphas = {m: (1.00 if m == "EVOLVE" else 0.80) for m in methods}
# linewidths = {m: (1.1 if m == "EVOLVE" else 0.45) for m in methods}

# # ── Layout params ────────────────────────────────────────────────────────────
# n_metrics = len(metrics)  # 4 groups on x-axis
# n_methods = len(methods)  # 5 bars per group
# bar_w = 0.13
# group_gap = 0.18
# x_centers = np.arange(n_metrics) * (n_methods * bar_w + group_gap)

# # ── Figure ───────────────────────────────────────────────────────────────────
# fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharey=False)
# fig.subplots_adjust(wspace=0.36, left=0.07, right=0.98, top=0.87, bottom=0.12)

# # 删除了 titles 变量
# for ax, backbone in zip(axes, ["DeepSeek", "GPT-4o"]):
#     bdata = data[backbone]

#     for mi, method in enumerate(methods):
#         vals = bdata[method]
#         offset = (mi - (n_methods - 1) / 2) * bar_w
#         xpos = x_centers + offset

#         ax.bar(
#             xpos,
#             vals,
#             width=bar_w,
#             color=colors[method],
#             alpha=alphas[method],
#             edgecolor=edge_colors[method],
#             linewidth=linewidths[method],
#             zorder=3,
#         )

#     # x-axis
#     ax.set_xticks(x_centers)
#     ax.set_xticklabels(metrics, fontsize=11.5)
#     ax.set_ylabel("Score", fontsize=11, labelpad=5)
#     ax.set_ylim(0.0, 1.08)
#     ax.yaxis.set_major_locator(ticker.MultipleLocator(0.2))
#     ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.1))
#     ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))
#     ax.tick_params(axis="y", labelsize=9)

#     # grid
#     ax.grid(
#         axis="y", which="major", linestyle="--", linewidth=0.5, alpha=0.55, zorder=0
#     )
#     ax.grid(axis="y", which="minor", linestyle=":", linewidth=0.3, alpha=0.35, zorder=0)
#     ax.set_axisbelow(True)

#     # subtle group separators
#     for xc in x_centers[:-1]:
#         ax.axvline(
#             xc + (n_methods * bar_w + group_gap) / 2,
#             color="#cccccc",
#             linewidth=0.55,
#             linestyle=":",
#             zorder=1,
#         )

#     # 删除了 ax.set_title() 这一行

# # ── Shared legend ─────────────────────────────────────────────────────────────
# handles = [
#     mpatches.Patch(
#         facecolor=colors[m],
#         edgecolor=edge_colors[m],
#         linewidth=linewidths[m],
#         alpha=alphas[m],
#         label=f"{m} (ours)" if m == "EVOLVE" else m,
#     )
#     for m in methods
# ]

# fig.legend(
#     handles=handles,
#     loc="upper center",
#     ncol=5,
#     fontsize=10,
#     frameon=True,
#     framealpha=0.92,
#     edgecolor="#dddddd",
#     handlelength=1.2,
#     handleheight=0.9,
#     columnspacing=1.5,
#     borderpad=0.55,
#     bbox_to_anchor=(0.53, 1.01),
# )

# out_png = "agentharm_results.png"
# out_pdf = "agentharm_results.pdf"
# fig.savefig(out_png, dpi=300, bbox_inches="tight")
# fig.savefig(out_pdf, dpi=300, bbox_inches="tight")
# print("Done.")

# 第二张图
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as ticker
import numpy as np

matplotlib.rcParams["font.family"] = "DejaVu Sans"
matplotlib.rcParams["axes.spines.top"] = False
matplotlib.rcParams["axes.spines.right"] = False
matplotlib.rcParams["axes.linewidth"] = 0.8
matplotlib.rcParams["xtick.major.size"] = 0
matplotlib.rcParams["ytick.major.size"] = 3

# ── Data  (EVOLVE only) ───────────────────────────────────────────────────────
backbones = ["DeepSeek", "GPT-4o", "Claude", "Gemini"]
metrics = ["Acc", "Prec", "Rec", "F1"]

data = {
    "AgentHarm": {
        "DeepSeek": [0.8800, 0.9346, 0.8171, 0.8719],
        "GPT-4o": [0.8758, 1.0000, 0.8240, 0.9035],
        "Claude": [0.8812, 0.9854, 0.8272, 0.8994],
        "Gemini": [0.8886, 0.9851, 0.8468, 0.9107],
    },
    "RJudge": {
        "DeepSeek": [0.9107, 0.8982, 0.9148, 0.9065],
        "GPT-4o": [0.9002, 0.9338, 0.8916, 0.9123],
        "Claude": [0.8841, 0.9349, 0.8385, 0.8841],
        "Gemini": [0.9255, 0.9257, 0.9222, 0.9240],
    },
}

# ── Palette — 4 muted colors for 4 metrics ───────────────────────────────────
metric_colors = {
    "Acc": "#6E9EC8",  # muted blue
    "Prec": "#6BBAA7",  # muted teal
    "Rec": "#C8A96E",  # muted amber
    "F1": "#C0544F",  # coral-red  (most important → slightly stronger)
}
metric_edge = {
    "Acc": "#3d6a8a",
    "Prec": "#3a7a69",
    "Rec": "#8a6e3a",
    "F1": "#8B2E2B",
}
metric_alpha = {"Acc": 0.80, "Prec": 0.80, "Rec": 0.80, "F1": 1.00}
metric_lw = {"Acc": 0.45, "Prec": 0.45, "Rec": 0.45, "F1": 1.10}

# ── Layout ────────────────────────────────────────────────────────────────────
n_groups = len(backbones)  # 4  →  x-axis groups
n_bars = len(metrics)  # 4  →  bars per group
bar_w = 0.15
group_gap = 0.18
x_centers = np.arange(n_groups) * (n_bars * bar_w + group_gap)

# ── Figure ────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharey=False)
fig.subplots_adjust(wspace=0.36, left=0.07, right=0.98, top=0.87, bottom=0.12)

panel_titles = ["AgentHarm  (EVOLVE)", "RJudge  (EVOLVE)"]
y_lims = [(0.75, 1.06), (0.82, 0.96)]

for ax, dataset, title, ylim in zip(
    axes, ["AgentHarm", "RJudge"], panel_titles, y_lims
):
    bdata = data[dataset]

    for mi, metric in enumerate(metrics):
        offset = (mi - (n_bars - 1) / 2) * bar_w
        xpos = x_centers + offset
        vals = [bdata[b][mi] for b in backbones]

        ax.bar(
            xpos,
            vals,
            width=bar_w,
            color=metric_colors[metric],
            alpha=metric_alpha[metric],
            edgecolor=metric_edge[metric],
            linewidth=metric_lw[metric],
            zorder=3,
        )

    ax.set_xticks(x_centers)
    ax.set_xticklabels(backbones, fontsize=11)
    ax.set_ylabel("Score", fontsize=11, labelpad=5)
    ax.set_ylim(ylim)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.05))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.025))
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
    ax.tick_params(axis="y", labelsize=9)

    ax.grid(
        axis="y", which="major", linestyle="--", linewidth=0.5, alpha=0.55, zorder=0
    )
    ax.grid(axis="y", which="minor", linestyle=":", linewidth=0.3, alpha=0.35, zorder=0)
    ax.set_axisbelow(True)

    for xc in x_centers[:-1]:
        ax.axvline(
            xc + (n_bars * bar_w + group_gap) / 2,
            color="#cccccc",
            linewidth=0.55,
            linestyle=":",
            zorder=1,
        )

    ax.set_title(title, fontsize=11.5, fontweight="normal", pad=7, color="#1a1a1a")

# ── Shared legend ─────────────────────────────────────────────────────────────
handles = [
    mpatches.Patch(
        facecolor=metric_colors[m],
        edgecolor=metric_edge[m],
        linewidth=metric_lw[m],
        alpha=metric_alpha[m],
        label=m,
    )
    for m in metrics
]

fig.legend(
    handles=handles,
    loc="upper center",
    ncol=4,
    fontsize=10.5,
    frameon=True,
    framealpha=0.92,
    edgecolor="#dddddd",
    handlelength=1.2,
    handleheight=0.9,
    columnspacing=1.6,
    borderpad=0.55,
    bbox_to_anchor=(0.53, 1.01),
)

fig.savefig("evolve_two_datasets.png", dpi=300, bbox_inches="tight")
fig.savefig("evolve_two_datasets.pdf", dpi=300, bbox_inches="tight")
print("Done.")
