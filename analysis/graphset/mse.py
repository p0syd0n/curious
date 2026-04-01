"""
mse.py  [graphset]
Mean squared error per boosting round.
"""

import matplotlib.pyplot as plt


def plot(data, save=None):
    stats = data["tree_stats"]
    rounds = list(range(len(stats)))
    values = [t["MSE"] for t in stats]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(rounds, values, marker="o", markersize=3, linewidth=1.5, color="steelblue")
    ax.set_xlabel("Boosting Round")
    ax.set_ylabel("MSE")
    ax.set_title("MSE per Round")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if save:
        fig.savefig(save, dpi=150)
    else:
        plt.show()
    plt.close(fig)