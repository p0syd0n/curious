"""
leaf_value_delta.py  [graphset]
Round-over-round change in mean leaf value.
Helps spot sudden jumps or stalls in learning.
"""

import matplotlib.pyplot as plt


def plot(data, save=None):
    stats = data["tree_stats"]
    values = [t["mean_leaf_value"] for t in stats]
    deltas = [values[i] - values[i - 1] for i in range(1, len(values))]
    rounds = list(range(1, len(values)))

    colors = ["tomato" if d > 0 else "steelblue" for d in deltas]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(rounds, deltas, color=colors, width=0.8)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Boosting Round")
    ax.set_ylabel("Δ Mean Leaf Value")
    ax.set_title("Leaf Value Delta (round-over-round)")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()

    if save:
        fig.savefig(save, dpi=150)
    else:
        plt.show()
    plt.close(fig)