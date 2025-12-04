import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from ML import config, data_prep

def analyze_correlations():
    # Load Data
    df = data_prep.load_and_prep_data()
    
    target = config.TARGET_COL
    
    # Calculate correlations with Target
    corrs = df.corr()[target].drop(target)
    
    # Sort by absolute correlation
    corrs_abs = corrs.abs().sort_values(ascending=False)
    
    print("\n--- Top 20 Feature Correlations with Target ---")
    print(corrs[corrs_abs.index[:20]])
    
    # Plot
    plt.figure(figsize=(10, 8))
    sns.heatmap(df[corrs_abs.index[:20].tolist() + [target]].corr(), annot=True, fmt=".2f", cmap='coolwarm')
    plt.title("Correlation Heatmap (Top 20 Features)")
    plt.tight_layout()
    plt.savefig('ML_Correlations.png')
    print("\nCorrelation heatmap saved to ML_Correlations.png")

if __name__ == "__main__":
    analyze_correlations()
