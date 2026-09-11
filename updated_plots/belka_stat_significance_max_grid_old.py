import numpy as np
import matplotlib.pyplot as plt

def main():
    # Model sizes (Millions of parameters)
    sizes = np.array([170, 380, 1300, 3000])
    x_labels = ["170M", "380M", "1.3B", "3B"]

    # Public AP runs (3 runs per size) - Max Grid
    public_runs = {
        170: np.array([0.33577, 0.28640, 0.30613]),
        380: np.array([0.32810, 0.30505, 0.32019]),
        1300: np.array([0.34764, 0.34255, 0.34304]),
        3000: np.array([0.30909, 0.31198, 0.32455])
    }

    # Private AP runs (3 runs per size) - Max Grid
    private_runs = {
        170: np.array([0.20161, 0.18136, 0.18855]),
        380: np.array([0.19381, 0.20220, 0.19818]),
        1300: np.array([0.22321, 0.22801, 0.21918]),
        3000: np.array([0.18508, 0.19164, 0.19036])
    }

    # Compute statistics (mean and standard deviation)
    public_means = np.array([np.mean(public_runs[s]) for s in sizes])
    public_stds = np.array([np.std(public_runs[s], ddof=1) if len(public_runs[s]) > 1 else 0.0 for s in sizes])

    private_means = np.array([np.mean(private_runs[s]) for s in sizes])
    private_stds = np.array([np.std(private_runs[s], ddof=1) if len(private_runs[s]) > 1 else 0.0 for s in sizes])

    # Setup aesthetic styles
    plt.rcParams["font.sans-serif"] = "Arial"
    plt.rcParams["font.family"] = "sans-serif"

    # Grid of subplots: 2 rows (metrics) x 1 column
    fig, axes = plt.subplots(2, 1, figsize=(7, 10))

    # Color scheme
    color_public = "#457b9d"   # Steel blue for Public AP
    color_private = "#e07a5f"  # Terracotta for Private AP

    # --- Plot 1: Public AP ---
    ax1 = axes[0]
    ax1.fill_between(sizes, public_means - public_stds, public_means + public_stds, color=color_public, alpha=0.15)
    ax1.errorbar(
        sizes, public_means, yerr=public_stds, fmt="-o", color=color_public,
        linewidth=2.5, markersize=8, capsize=5, elinewidth=1.5,
        markeredgewidth=1.5, label="Mean ± SD"
    )
    # Plot individual runs
    for s in sizes:
        ax1.scatter([s] * len(public_runs[s]), public_runs[s], color=color_public, alpha=0.6, edgecolors='none', s=40, zorder=3, label="Individual Runs" if s == 170 else "")

    ax1.set_title("Public AP", fontsize=14, fontweight="bold", pad=12, color="#2c3e50")
    ax1.set_ylabel("AP Score", fontsize=12, fontweight="bold", color="#2c3e50")
    ax1.set_xscale("log")
    ax1.minorticks_off()
    ax1.set_xticks(sizes)
    ax1.set_xticklabels(x_labels, fontsize=11, color="#2c3e50")
    ax1.set_xlim(120, 4500)
    ax1.grid(True, which="both", linestyle=":", alpha=0.5, color="#bbb")
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.spines["left"].set_color("#bdc3c7")
    ax1.spines["bottom"].set_color("#bdc3c7")
    ax1.tick_params(axis="both", colors="#2c3e50", labelsize=11)
    # Remove duplicate legend labels
    handles, labels = ax1.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax1.legend(by_label.values(), by_label.keys(), loc="upper right", frameon=True, edgecolor="#bdc3c7", facecolor="#ffffff", fontsize=10)

    # --- Plot 2: Private AP ---
    ax2 = axes[1]
    ax2.fill_between(sizes, private_means - private_stds, private_means + private_stds, color=color_private, alpha=0.15)
    ax2.errorbar(
        sizes, private_means, yerr=private_stds, fmt="-s", color=color_private,
        linewidth=2.5, markersize=8, capsize=5, elinewidth=1.5,
        markeredgewidth=1.5, label="Mean ± SD"
    )
    # Plot individual runs
    for s in sizes:
        ax2.scatter([s] * len(private_runs[s]), private_runs[s], color=color_private, alpha=0.6, edgecolors='none', s=40, zorder=3, label="Individual Runs" if s == 170 else "")

    ax2.set_title("Private AP", fontsize=14, fontweight="bold", pad=12, color="#2c3e50")
    ax2.set_ylabel("AP Score", fontsize=12, fontweight="bold", color="#2c3e50")
    ax2.set_xscale("log")
    ax2.minorticks_off()
    ax2.set_xticks(sizes)
    ax2.set_xticklabels(x_labels, fontsize=11, color="#2c3e50")
    ax2.set_xlim(120, 4500)
    ax2.grid(True, which="both", linestyle=":", alpha=0.5, color="#bbb")
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.spines["left"].set_color("#bdc3c7")
    ax2.spines["bottom"].set_color("#bdc3c7")
    ax2.tick_params(axis="both", colors="#2c3e50", labelsize=11)
    # Remove duplicate legend labels
    handles, labels = ax2.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax2.legend(by_label.values(), by_label.keys(), loc="upper right", frameon=True, edgecolor="#bdc3c7", facecolor="#ffffff", fontsize=10)

    # Title the entire plot
    fig.suptitle("BELKA max grid statistical significance", fontsize=18, fontweight="bold", y=0.98, color="#2c3e50")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # Save the output image
    output_filename = "belka_stat_significance_max_grid.png"
    plt.savefig(output_filename, dpi=300, bbox_inches="tight")
    print(f"Plot successfully saved to {output_filename}")

if __name__ == "__main__":
    main()
