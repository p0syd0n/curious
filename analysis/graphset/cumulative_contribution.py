"""
cumulative_contribution.py  [graphset]
Per-round leaf value alongside its cumulative sum.
Flattening cumulative curve = model converging.
"""

import itertools
import matplotlib.pyplot as plt


def plot(data, save=None):
    stats = data["tree_stats"]
    rounds = list(range(len(stats)))
    values = [t["mean_leaf_value"] for t in stats]
    cumulative = list(itertools.accumulate(values))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(rounds, values, linewidth=1.5, color="steelblue")
    ax1.set_xlabel("Round")
    ax1.set_ylabel("Mean Leaf Value")
    ax1.set_title("Per-Round Leaf Value")
    ax1.grid(True, alpha=0.3)

    ax2.plot(rounds, cumulative, linewidth=1.5, color="darkorange")
    ax2.set_xlabel("Round")
    ax2.set_ylabel("Cumulative Sum")
    ax2.set_title("Cumulative Leaf Value")
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()

    if save:
        fig.savefig(save, dpi=150)
    else:
        plt.show()
    plt.close(fig)