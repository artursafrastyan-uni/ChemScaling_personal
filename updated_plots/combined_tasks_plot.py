import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def main():
    # Setup base directories
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    pxr_dir = os.path.join(project_root, "pxr")

    # Models and labels
    models = ["170M", "380M", "1.3B", "3B"]
    x = np.arange(len(models))  # Categorical positions for clean layout

    # =========================================================================
    # 1. Polaris ADME (R2) Data (from polaris_max_grid.py)
    # =========================================================================
    polaris_r2_runs = {
        "170M": np.array([0.496, 0.490158, 0.501787, 0.491401, 0.483011]),
        "380M": np.array([0.485776, 0.505399, 0.51302, 0.48477, 0.478909]),
        "1.3B": np.array([0.495186, 0.495137, 0.494791, 0.487685, 0.513127]),
        "3B":   np.array([0.51355, 0.485445, 0.496758, 0.507458, 0.49295])
    }
    polaris_means = np.array([np.mean(polaris_r2_runs[m]) for m in models])
    polaris_stds = np.array([np.std(polaris_r2_runs[m], ddof=1) for m in models])

    # =========================================================================
    # 2. PXR Induction Prediction (R2) Data (from scale_plot_max_grid.py)
    # =========================================================================
    pxr_files = {
        "170M": "/mnt/weka/asafrastyan/refined/pxr/multiseed/variance_metrics_170_full.csv",
        "380M": "/mnt/weka/asafrastyan/refined/pxr/multiseed/variance_metrics_380_full2.csv",
        "1.3B": "/mnt/weka/asafrastyan/refined/pxr/multiseed/variance_metrics_13_full2.csv",
        "3B":   "/mnt/weka/asafrastyan/refined/pxr/multiseed/variance_metrics_3_full2.csv"
    }

    pxr_means = []
    pxr_stds = []
    for m in models:
        path = pxr_files[m]
        if not os.path.exists(path):
            raise FileNotFoundError(f"Required PXR file not found: {path}")
        df = pd.read_csv(path).set_index("Task")
        pxr_means.append(df.loc["pEC50", "R2_mean"])
        pxr_stds.append(df.loc["pEC50", "R2_std"])
    pxr_means = np.array(pxr_means)
    pxr_stds = np.array(pxr_stds)

    # =========================================================================
    # 3. Belka Public & Private (AP) Data
    # =========================================================================
    belka_public_runs = {
        "170M": np.array([0.39095, 0.41026, 0.39090, 0.39508, 0.39416]),
        "380M": np.array([0.40033, 0.40665, 0.40376, 0.40838, 0.40450]),
        "1.3B": np.array([0.37087, 0.38897, 0.39783, 0.39170, 0.38871]),
        "3B":   np.array([0.38990, 0.39307, 0.39321, 0.38651, 0.39106])
    }
    belka_private_runs = {
        "170M": np.array([0.23542, 0.23557, 0.23597, 0.23753, 0.23986]),
        "380M": np.array([0.25402, 0.24878, 0.25398, 0.25108, 0.25305]),
        "1.3B": np.array([0.24227, 0.24482, 0.25780, 0.24456, 0.24152]),
        "3B":   np.array([0.23804, 0.23964, 0.24012, 0.23886, 0.24195])
    }

    belka_pub_means = np.array([np.mean(belka_public_runs[m]) for m in models])
    belka_pub_stds = np.array([np.std(belka_public_runs[m], ddof=1) for m in models])

    belka_priv_means = np.array([np.mean(belka_private_runs[m]) for m in models])
    belka_priv_stds = np.array([np.std(belka_private_runs[m], ddof=1) for m in models])

    # =========================================================================
    # Plotting: Two Panels (R2 and AP)
    # =========================================================================
    plt.rcParams["font.sans-serif"] = "Arial"
    plt.rcParams["font.family"] = "sans-serif"

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8), dpi=300)

    # Styling colors to match the reference image palette
    c_polaris = "#4a72b0"    # Blue
    c_pxr = "#dd8452"        # Orange/Coral
    c_belka_pub = "#55a868"  # Green
    c_belka_priv = "#c44e52" # Red

    # -------------------------------------------------------------------------
    # Panel 1: R2 Tasks (Polaris & PXR)
    # -------------------------------------------------------------------------
    # 1. Avg Polaris ADME (R2)
    ax1.fill_between(x, polaris_means - polaris_stds, polaris_means + polaris_stds, color=c_polaris, alpha=0.15)
    line_polaris = ax1.errorbar(
        x, polaris_means, yerr=polaris_stds,
        fmt="-o", color=c_polaris, linewidth=2.2, markersize=7,
        capsize=4, capthick=1.2, elinewidth=1.2,
        label=r"Avg Polaris ADME ($R^2$)"
    )

    # 2. PXR Induction Prediction (R2)
    ax1.fill_between(x, pxr_means - pxr_stds, pxr_means + pxr_stds, color=c_pxr, alpha=0.15)
    line_pxr = ax1.errorbar(
        x, pxr_means, yerr=pxr_stds,
        fmt="-s", color=c_pxr, linewidth=2.2, markersize=7,
        capsize=4, capthick=1.2, elinewidth=1.2,
        label=r"PXR Induction Prediction ($R^2$)"
    )

    ax1.set_title(r"$R^2$ Tasks", fontsize=12, fontweight="bold", pad=10)
    ax1.set_xlabel("Model Size", fontsize=11, fontweight="bold", labelpad=8)
    ax1.set_ylabel(r"$R^2$ Score", fontsize=11, fontweight="bold", labelpad=8)
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, fontsize=10.5)
    ax1.set_xlim(-0.25, len(models) - 0.75)
    ax1.grid(True, linestyle="-", alpha=0.35, color="#ccc")
    for spine in ax1.spines.values():
        spine.set_color("#cccccc")
        spine.set_linewidth(1.0)

    # -------------------------------------------------------------------------
    # Panel 2: AP Tasks (BELKA Public & Private)
    # -------------------------------------------------------------------------
    # 3. Belka Public (AP)
    ax2.fill_between(x, belka_pub_means - belka_pub_stds, belka_pub_means + belka_pub_stds, color=c_belka_pub, alpha=0.15)
    line_belka_pub = ax2.errorbar(
        x, belka_pub_means, yerr=belka_pub_stds,
        fmt="--^", color=c_belka_pub, linewidth=2.2, markersize=7,
        capsize=4, capthick=1.2, elinewidth=1.2,
        label="Belka Public (AP)"
    )

    # 4. Belka Private (AP)
    ax2.fill_between(x, belka_priv_means - belka_priv_stds, belka_priv_means + belka_priv_stds, color=c_belka_priv, alpha=0.15)
    line_belka_priv = ax2.errorbar(
        x, belka_priv_means, yerr=belka_priv_stds,
        fmt="--v", color=c_belka_priv, linewidth=2.2, markersize=7,
        capsize=4, capthick=1.2, elinewidth=1.2,
        label="Belka Private (AP)"
    )

    ax2.set_title("Average Precision (AP) Tasks", fontsize=12, fontweight="bold", pad=10)
    ax2.set_xlabel("Model Size", fontsize=11, fontweight="bold", labelpad=8)
    ax2.set_ylabel("AP Score", fontsize=11, fontweight="bold", labelpad=8)
    ax2.set_xticks(x)
    ax2.set_xticklabels(models, fontsize=10.5)
    ax2.set_xlim(-0.25, len(models) - 0.75)
    ax2.grid(True, linestyle="-", alpha=0.35, color="#ccc")
    for spine in ax2.spines.values():
        spine.set_color("#cccccc")
        spine.set_linewidth(1.0)

    # -------------------------------------------------------------------------
    # Global Super Title and Shared Legend
    # -------------------------------------------------------------------------
    fig.suptitle("Performance vs. Model Size Across Tasks", fontsize=14, fontweight="bold", y=0.98)

    all_handles = [line_polaris, line_pxr, line_belka_pub, line_belka_priv]
    all_labels = [
        r"Avg Polaris ADME ($R^2$)",
        r"PXR Induction Prediction ($R^2$)",
        "Belka Public (AP)",
        "Belka Private (AP)"
    ]

    fig.legend(
        all_handles,
        all_labels,
        title="Tasks & Metrics",
        title_fontsize=10.5,
        fontsize=9.5,
        loc="center left",
        bbox_to_anchor=(0.84, 0.5),
        borderaxespad=0,
        frameon=True,
        edgecolor="#d0d0d0",
        facecolor="#ffffff"
    )

    plt.tight_layout(rect=[0, 0, 0.83, 0.94])

    # Save two-panel plot
    output_path = os.path.join(script_dir, "combined_performance_two_panels.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Two-panel plot successfully saved to: {output_path}")

    # Also save separate individual plots just in case the user wants them individually
    # Plot A: R2 (Polaris + Pxr)
    fig_r2, ax_r2 = plt.subplots(figsize=(6.5, 4.8), dpi=300)
    ax_r2.fill_between(x, polaris_means - polaris_stds, polaris_means + polaris_stds, color=c_polaris, alpha=0.15)
    ax_r2.errorbar(x, polaris_means, yerr=polaris_stds, fmt="-o", color=c_polaris, linewidth=2.2, markersize=7, capsize=4, capthick=1.2, elinewidth=1.2, label=r"Avg Polaris ADME ($R^2$)")
    ax_r2.fill_between(x, pxr_means - pxr_stds, pxr_means + pxr_stds, color=c_pxr, alpha=0.15)
    ax_r2.errorbar(x, pxr_means, yerr=pxr_stds, fmt="-s", color=c_pxr, linewidth=2.2, markersize=7, capsize=4, capthick=1.2, elinewidth=1.2, label=r"PXR Induction Prediction ($R^2$)")
    ax_r2.set_title(r"Performance vs. Model Size ($R^2$ Tasks)", fontsize=13, fontweight="bold", pad=12)
    ax_r2.set_xlabel("Model Size", fontsize=11, fontweight="bold", labelpad=8)
    ax_r2.set_ylabel(r"$R^2$ Score", fontsize=11, fontweight="bold", labelpad=8)
    ax_r2.set_xticks(x)
    ax_r2.set_xticklabels(models, fontsize=10.5)
    ax_r2.set_xlim(-0.25, len(models) - 0.75)
    ax_r2.set_ylim(0.31, 0.525)
    ax_r2.grid(True, linestyle="-", alpha=0.35, color="#ccc")
    for spine in ax_r2.spines.values():
        spine.set_color("#cccccc")
        spine.set_linewidth(1.0)
    ax_r2.legend(title="Tasks & Metrics", title_fontsize=10, fontsize=9.5, loc="lower left", frameon=True, edgecolor="#d0d0d0", facecolor="#ffffff")
    plt.tight_layout()
    path_r2 = os.path.join(script_dir, "performance_r2_tasks.png")
    fig_r2.savefig(path_r2, dpi=300, bbox_inches="tight")
    plt.close(fig_r2)

    # Plot B: AP (Belka Public + Private)
    fig_ap, ax_ap = plt.subplots(figsize=(6.5, 4.8), dpi=300)
    ax_ap.fill_between(x, belka_pub_means - belka_pub_stds, belka_pub_means + belka_pub_stds, color=c_belka_pub, alpha=0.15)
    ax_ap.errorbar(x, belka_pub_means, yerr=belka_pub_stds, fmt="--^", color=c_belka_pub, linewidth=2.2, markersize=7, capsize=4, capthick=1.2, elinewidth=1.2, label="Belka Public (AP)")
    ax_ap.fill_between(x, belka_priv_means - belka_priv_stds, belka_priv_means + belka_priv_stds, color=c_belka_priv, alpha=0.15)
    ax_ap.errorbar(x, belka_priv_means, yerr=belka_priv_stds, fmt="--v", color=c_belka_priv, linewidth=2.2, markersize=7, capsize=4, capthick=1.2, elinewidth=1.2, label="Belka Private (AP)")
    ax_ap.set_title("Performance vs. Model Size (BELKA AP Tasks)", fontsize=13, fontweight="bold", pad=12)
    ax_ap.set_xlabel("Model Size", fontsize=11, fontweight="bold", labelpad=8)
    ax_ap.set_ylabel("AP Score", fontsize=11, fontweight="bold", labelpad=8)
    ax_ap.set_xticks(x)
    ax_ap.set_xticklabels(models, fontsize=10.5)
    ax_ap.set_xlim(-0.25, len(models) - 0.75)
    ax_ap.grid(True, linestyle="-", alpha=0.35, color="#ccc")
    for spine in ax_ap.spines.values():
        spine.set_color("#cccccc")
        spine.set_linewidth(1.0)
    ax_ap.legend(title="Tasks & Metrics", title_fontsize=10, fontsize=9.5, loc="lower left", frameon=True, edgecolor="#d0d0d0", facecolor="#ffffff")
    plt.tight_layout()
    path_ap = os.path.join(script_dir, "performance_belka_tasks.png")
    fig_ap.savefig(path_ap, dpi=300, bbox_inches="tight")
    plt.close(fig_ap)

if __name__ == "__main__":
    main()
