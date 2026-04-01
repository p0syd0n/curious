"""
tree_structure.py  [graphset]
Histogram of leaves, depth, and decisions per round.
All rounds share the same structure here, but this will show variance
if your model ever uses variable-depth trees.
"""

import matplotlib.pyplot as plt
from collections import Counter


def plot(data, save=None):
    stats = data["tree_stats"]

    depths = [t["depth"] for t in stats]
    leaves = [t["num_leaves"] for t in stats]
    decisions = [t["num_decisions"] for t in stats]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    for ax, values, label, color in zip(
        axes,
        [depths, leaves, decisions],
        ["Depth", "Num Leaves", "Num Decisions"],
        ["steelblue", "darkorange", "mediumseagreen"],
    ):
        counts = Counter(values)
        ax.bar(counts.keys(), counts.values(), color=color, width=0.4)
        ax.set_xlabel(label)
        ax.set_ylabel("# Rounds")
        ax.set_title(f"Distribution of {label}")
        ax.grid(True, axis="y", alpha=0.3)
        ax.set_xticks(sorted(counts.keys()))

    fig.tight_layout()

    if save:
        fig.savefig(save, dpi=150)
    else:
        plt.show()
    plt.close(fig)