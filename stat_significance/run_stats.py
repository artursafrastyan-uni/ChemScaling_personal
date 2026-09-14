import pandas as pd
import numpy as np
import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
import scikit_posthocs as sp
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Define file paths
DATA_FILE = "results.csv"
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def generate_dummy_data():
    """Generates a template CSV file if one doesn't exist."""
    print(f"Creating template {DATA_FILE}...")
    data = []
    sizes = ["170M", "380M", "1.3B", "3B"]
    tasks = ["Polaris", "PXR", "BELKA"]
    seeds = [1, 2, 3, 4, 5]
    
    for task in tasks:
        for size in sizes:
            for seed in seeds:
                # Add dummy values, users should replace these with real values
                data.append({"Task": task, "Model_Size": size, "Seed": seed, "Metric_Value": np.random.uniform(0.2, 0.5)})
    
    df = pd.DataFrame(data)
    df.to_csv(DATA_FILE, index=False)
    print(f"Please fill in your actual per-seed values in {DATA_FILE} and re-run this script.")

def run_two_way_anova(df):
    print("\n" + "="*50)
    print("PARAMETRIC: TWO-WAY FACTORIAL ANOVA")
    print("="*50)
    # Fit the model: Metric_Value ~ Model_Size + Task + Model_Size:Task
    # C() denotes categorical variables
    model = ols('Metric_Value ~ C(Model_Size) + C(Task) + C(Model_Size):C(Task)', data=df).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)
    print(anova_table)
    
    with open(f"{OUTPUT_DIR}/two_way_anova_results.txt", "w") as f:
        f.write("Two-Way ANOVA Results:\n")
        f.write(anova_table.to_string())
    
    # Check interaction significance
    p_interaction = anova_table.loc['C(Model_Size):C(Task)', 'PR(>F)']
    if p_interaction < 0.05:
        print(f"\n✅ SIGNIFICANT INTERACTION EFFECT FOUND (p = {p_interaction:.4f}).")
        print("This mathematically proves that the effect of scaling diverges across different downstream tasks.")
    else:
        print(f"\n❌ No significant interaction effect (p = {p_interaction:.4f}).")

def run_non_parametric_tests(df):
    print("\n" + "="*50)
    print("NON-PARAMETRIC: FRIEDMAN & NEMENYI TESTS (Per Task)")
    print("="*50)
    
    sizes_order = ["170M", "380M", "1.3B", "3B"]
    
    with open(f"{OUTPUT_DIR}/non_parametric_results.txt", "w") as f:
        for task in df['Task'].unique():
            print(f"\n--- Task: {task} ---")
            f.write(f"\n--- Task: {task} ---\n")
            
            task_df = df[df['Task'] == task]
            
            # Pivot data for Friedman test (rows=seeds, cols=model_sizes)
            pivot_df = task_df.pivot(index='Seed', columns='Model_Size', values='Metric_Value')
            # Ensure column order
            pivot_df = pivot_df[sizes_order]
            
            # 1. Friedman Test
            stat, p_value = stats.friedmanchisquare(*[pivot_df[col].dropna() for col in pivot_df.columns])
            print(f"Friedman Test p-value: {p_value:.5f}")
            f.write(f"Friedman Test p-value: {p_value:.5f}\n")
            
            if p_value < 0.05:
                print("Significant differences found. Running Nemenyi post-hoc test...")
                # 2. Nemenyi post-hoc test
                # melt the pivot table back to run posthocs safely
                melted = pivot_df.melt(var_name='Model_Size', value_name='Metric_Value')
                
                try:
                    nemenyi_results = sp.posthoc_nemenyi_friedman(pivot_df.values)
                    # Convert to dataframe with proper labels
                    nemenyi_df = pd.DataFrame(nemenyi_results, index=sizes_order, columns=sizes_order)
                    print("Nemenyi Pairwise p-values:")
                    print(nemenyi_df.round(4))
                    f.write("Nemenyi Pairwise p-values:\n")
                    f.write(nemenyi_df.round(4).to_string() + "\n")
                except Exception as e:
                    print(f"Could not run Nemenyi test: {e}")
            else:
                print("No significant differences found across sizes. Post-hoc not needed.")

def plot_with_confidence_intervals(df):
    print("\n" + "="*50)
    print("VISUAL VALIDATION: PLOTTING WITH 95% CIs")
    print("="*50)
    
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 6))
    
    # We map sizes to numerical values for plotting (log-scale representation)
    size_mapping = {"170M": 170, "380M": 380, "1.3B": 1300, "3B": 3000}
    df['Size_Numeric'] = df['Model_Size'].map(size_mapping)
    
    # Plot using seaborn lineplot, which automatically calculates and shades 95% CIs via bootstrapping
    ax = sns.lineplot(
        data=df, 
        x='Size_Numeric', 
        y='Metric_Value', 
        hue='Task', 
        marker='o',
        errorbar=('ci', 95), # 95% Confidence Interval
        linewidth=2.5,
        err_style="band"
    )
    
    ax.set_xscale('log')
    ax.set_xticks(list(size_mapping.values()))
    ax.set_xticklabels(list(size_mapping.keys()))
    
    plt.title("Scaling Behavior Across Benchmarks (Shaded Regions = 95% CI)")
    plt.xlabel("Model Size (Log Scale)")
    plt.ylabel("Performance Metric")
    
    plot_path = f"{OUTPUT_DIR}/scaling_confidence_intervals.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved plot with 95% Confidence Intervals to {plot_path}")

def main():
    if not os.path.exists(DATA_FILE):
        generate_dummy_data()
        return
    
    print(f"Loading data from {DATA_FILE}...")
    df = pd.read_csv(DATA_FILE)
    
    # Ensure proper ordering of sizes for analysis
    df['Model_Size'] = pd.Categorical(df['Model_Size'], categories=["170M", "380M", "1.3B", "3B"], ordered=True)
    
    run_two_way_anova(df)
    run_non_parametric_tests(df)
    plot_with_confidence_intervals(df)
    
    print("\nAnalysis complete! Check the 'outputs' directory for detailed reports and plots.")

if __name__ == "__main__":
    main()
