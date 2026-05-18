"""
plotting of accuracy
"""

import numpy as np
import matplotlib.pyplot as plt

BLUE  = "#185FA5"
CORAL = "#D85A30"
TEAL  = "#3A9999"
GRAY  = "#888888"


def plot_accuracy_bar(before, after, technique_name):
    """
    Two bars: pretrained baseline vs after fine-tuning.
    The delta is annotated directly on the chart.
    """
    fig, ax = plt.subplots(figsize=(5, 3.2))
    bars = ax.bar(
        ["Pretrained\n(no fine-tune)", f"After\n{technique_name}"],
        [before, after],
        color=[GRAY, BLUE],
        edgecolor="#333", linewidth=0.8,
        width=0.5,
    )
    ax.set_ylabel("Test accuracy")
    ax.set_ylim(0, 1.0)
    ax.set_title("Accuracy comparison")
    ax.grid(alpha=0.25, axis="y")

    for bar, v in zip(bars, [before, after]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            v + 0.02,
            f"{v:.1%}",
            ha="center", fontweight="bold", fontsize=13,
        )

    delta = after - before
    sign = "+" if delta >= 0 else ""
    ax.annotate(
        f"{sign}{delta*100:.1f} pts",
        xy=(1, after), xytext=(1.35, (before + after) / 2),
        fontsize=11, color=BLUE if delta >= 0 else CORAL,
        fontweight="bold",
        arrowprops=dict(arrowstyle="->", color=GRAY, lw=1.2),
    )
    fig.tight_layout()
    return fig


def plot_training_curve(history, technique_name):

    fig, ax = plt.subplots(figsize=(6, 3.2))
    acc = history["train_acc"]
    ax.plot(acc, linewidth=2, color=BLUE)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Train accuracy")
    ax.set_ylim(0, 1)
    ax.set_title(f"{technique_name} - training curve")
    ax.grid(alpha=0.25)

    # Mark stage transitions for LP-FT
    if "stage" in history:
        stages = history["stage"]
        for i in range(1, len(stages)):
            if stages[i] != stages[i - 1]:
                ax.axvline(i - 0.5, color=CORAL, linestyle="--",
                           alpha=0.7, linewidth=1.2)
                ax.text(i - 0.5, 0.05, f"→ {stages[i]}",
                        color=CORAL, fontsize=8, ha="left")

    fig.tight_layout()
    return fig


def plot_comparison_bars(results):

    n = len(results)
    labels   = [r[0] for r in results]
    pre_vals = [r[1] for r in results]
    ft_vals  = [r[2] for r in results]

    colours = [BLUE, CORAL, TEAL, "#6B5B95", "#D4AC0D"][:n]

    x = np.arange(n)
    w = 0.35
    fig, ax = plt.subplots(figsize=(max(5, 2.5 * n), 3.8))

    pre_bars = ax.bar(x - w/2, pre_vals, w, color=GRAY,
                      label="Pretrained", edgecolor="#333", linewidth=0.6)
    ft_bars  = ax.bar(x + w/2, ft_vals,  w, color=colours,
                      label="Fine-tuned", edgecolor="#333", linewidth=0.6)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Test accuracy")
    ax.set_ylim(0, 1.0)
    ax.set_title("Strategy comparison - pretrained vs fine-tuned")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.25, axis="y")

    for bar, v in zip(list(pre_bars) + list(ft_bars),
                      pre_vals + ft_vals):
        ax.text(bar.get_x() + bar.get_width() / 2,
                v + 0.015, f"{v:.0%}",
                ha="center", fontsize=9, fontweight="bold")

    fig.tight_layout()
    return fig
