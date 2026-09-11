import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # File paths
    metrics_170_path = "/mnt/weka/asafrastyan/refined/pxr/multiseed/variance_metrics_170_full.csv"
    metrics_380_path = "/mnt/weka/asafrastyan/refined/pxr/multiseed/variance_metrics_380_full2.csv"
    metrics_13_path = "/mnt/weka/asafrastyan/refined/pxr/multiseed/variance_metrics_13_full2.csv"
    metrics_3_path = "/mnt/weka/asafrastyan/refined/pxr/multiseed/variance_metrics_3_full2.csv"
    
    # Verify file existence
    paths = [metrics_170_path, metrics_380_path, metrics_13_path, metrics_3_path]
    for p in paths:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Required file not found: {p}")
            
    # Read files
    df_170 = pd.read_csv(metrics_170_path).set_index("Task")
    df_380 = pd.read_csv(metrics_380_path).set_index("Task")
    df_13 = pd.read_csv(metrics_13_path).set_index("Task")
    df_3 = pd.read_csv(metrics_3_path).set_index("Task")
    
    task = "pEC50"
    metrics = ["R2", "RAE"]
    
    # Setup aesthetic style
    plt.rcParams["font.sans-serif"] = "Arial"
    plt.rcParams["font.family"] = "sans-serif"
    
    # Grid of subplots: 2 rows (metrics) x 1 column
    fig, axes = plt.subplots(2, 1, figsize=(6, 9))
    
    # Colors: Professional, modern color
    color_curve = "#5e35b1"  # Vibrant purple/indigo for Pxr
    
    # Model sizes on parameter scale (Millions of params):
    # 170M is 170, 380M is 380, 1.3B is 1300, 3B is 3000
    x_sizes = np.array([170, 380, 1300, 3000])
    x_labels = ["170M", "380M", "1.3B", "3B"]
    
    for row_idx, metric in enumerate(metrics):
        ax = axes[row_idx]
        
        # --- Extract Data ---
        # Sorted sizes: 170M, 380M, 1.3B, 3B
        val_170 = df_170.loc[task, f"{metric}_mean"]
        std_170 = df_170.loc[task, f"{metric}_std"]
        
        val_380 = df_380.loc[task, f"{metric}_mean"]
        std_380 = df_380.loc[task, f"{metric}_std"]
        
        val_13 = df_13.loc[task, f"{metric}_mean"]
        std_13 = df_13.loc[task, f"{metric}_std"]
        
        val_3 = df_3.loc[task, f"{metric}_mean"]
        std_3 = df_3.loc[task, f"{metric}_std"]
        
        y_vals = np.array([val_170, val_380, val_13, val_3])
        err_std = np.array([std_170, std_380, std_13, std_3])
        
        # Use CI if available
        ci_lower_col = f"{metric}_CI_lower"
        ci_upper_col = f"{metric}_CI_upper"
        if (ci_lower_col in df_170.columns and 
            ci_lower_col in df_380.columns and
            ci_lower_col in df_13.columns and
            ci_lower_col in df_3.columns):
            ci_lower_170 = df_170.loc[task, ci_lower_col]
            ci_upper_170 = df_170.loc[task, ci_upper_col]
            
            ci_lower_380 = df_380.loc[task, ci_lower_col]
            ci_upper_380 = df_380.loc[task, ci_upper_col]
            
            ci_lower_13 = df_13.loc[task, ci_lower_col]
            ci_upper_13 = df_13.loc[task, ci_upper_col]
            
            ci_lower_3 = df_3.loc[task, ci_lower_col]
            ci_upper_3 = df_3.loc[task, ci_upper_col]
            
            err_ci = np.array([
                [val_170 - ci_lower_170, val_380 - ci_lower_380, val_13 - ci_lower_13, val_3 - ci_lower_3],
                [ci_upper_170 - val_170, ci_upper_380 - val_380, ci_upper_13 - val_13, ci_upper_3 - val_3]
            ])
            err_ci = np.clip(err_ci, 0, None)
            lower_bound = np.array([ci_lower_170, ci_lower_380, ci_lower_13, ci_lower_3])
            upper_bound = np.array([ci_upper_170, ci_upper_380, ci_upper_13, ci_upper_3])
        else:
            err_ci = err_std
            lower_bound = y_vals - err_std
            upper_bound = y_vals + err_std
            
        # --- Plot validation curve ---
        ax.fill_between(x_sizes, lower_bound, upper_bound, color=color_curve, alpha=0.15)
        ax.errorbar(
            x_sizes, y_vals, yerr=err_ci, fmt="-o", color=color_curve,
            linewidth=2.5, markersize=8, capsize=5, elinewidth=1.5,
            markeredgewidth=1.5, label="Pxr Validation"
        )
        
        # Subplot titles and labels
        ax.set_title(f"{task} ({metric})", fontsize=14, fontweight="bold", pad=12)
        
        y_label = r"$R^2$" if metric == "R2" else "RAE"
        ax.set_ylabel(y_label, fontsize=12, fontweight="bold")
        
        # Style improvements
        ax.set_xscale("log")
        ax.minorticks_off()
        ax.set_xticks(x_sizes)
        ax.set_xticklabels(x_labels, fontsize=11)
        ax.set_xlim(120, 4500)  # Logarithmic scale bounding 170M, 380M, 1.3B, and 3B
        
        # Add horizontal line at R2=0 for reference in R2 plots
        if metric == "R2":
            ax.axhline(0, color="#cccccc", linestyle="--", linewidth=1, zorder=0)
        
        # Add soft grid
        ax.grid(True, which="both", linestyle=":", alpha=0.5, color="#bbb")
        
        # Hide top and right spines
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        
        # Adjust labels size
        ax.tick_params(axis="both", labelsize=11)
        
        # Add legend
        ax.legend(loc="lower left", frameon=True, edgecolor="#e0e0e0", facecolor="#ffffff", fontsize=10)
    
    # Title the entire plot
    fig.suptitle("Pxr max grid", fontsize=18, fontweight="bold", y=0.98)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.94])
    
    # Save the plot
    output_filename = os.path.join(script_dir, "pxr_max_grid.png")
    plt.savefig(output_filename, dpi=300, bbox_inches="tight")
    print(f"Plot successfully saved to {output_filename}")

    plots_dir = os.path.join(script_dir, "plots")
    if os.path.exists(plots_dir):
        plots_output = os.path.join(plots_dir, "pxr_max_grid.png")
        plt.savefig(plots_output, dpi=300, bbox_inches="tight")
        print(f"Plot also saved to {plots_output}")

if __name__ == "__main__":
    main()
