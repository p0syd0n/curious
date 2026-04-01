"""
feature_importance.py  [graphset]
Bar chart of feature importances. Zero-importance features are grayed out.
"""

import matplotlib.pyplot as plt


def plot(data, save=None):
    importance = data["feature_data"]["feature_importance"]
    indices = list(range(len(importance)))
    colors = ["steelblue" if v > 0 else "#cccccc" for v in importance]

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.bar(indices, importance, color=colors)
    ax.set_xlabel("Feature Index")
    ax.set_ylabel("Importance")
    ax.set_title("Feature Importance  (gray = zero)")
    ax.set_xticks(indices)
    ax.set_xticklabels([str(i) for i in indices], fontsize=7)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()

    if save:
        fig.savefig(save, dpi=150)
    else:
        plt.show()
    plt.close(fig)