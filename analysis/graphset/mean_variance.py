"""
mean_variance.py  [graphset]
Mean variance of leaf values per boosting round.
Higher variance = tree is making more aggressive splits.
Dropping variance over rounds = model is settling/converging.
"""

import matplotlib.pyplot as plt


def plot(data, save=None):
    stats = data["tree_stats"]
    rounds = list(range(len(stats)))
    variance = [t["mean_variance"] for t in stats]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(rounds, variance, marker="o", markersize=3, linewidth=1.5, color="mediumorchid")
    ax.fill_between(rounds, variance, alpha=0.15, color="mediumorchid")
    ax.set_xlabel("Boosting Round")
    ax.set_ylabel("Mean Variance")
    ax.set_title("Mean Leaf Variance per Round")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if save:
        fig.savefig(save, dpi=150)
    else:
        plt.show()
    plt.close(fig)