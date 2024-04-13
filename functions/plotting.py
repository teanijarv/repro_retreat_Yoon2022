import numpy as np
import pandas as pd
import pingouin as pg
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

from statannotations.Annotator import Annotator
from itertools import combinations

def covariates_pairwise_comparison(df, group_col, cont_vars=[], cat_vars=[], plot_title=None,
                                   group_order=['A-T-', 'A-T+', 'A+T+']):
    """Perform pairwise comparison of continuous and categorical covariates with plotting."""

    # Prepare plotting grid
    max_cols_per_row = 5
    all_vars = cont_vars + cat_vars
    n_vars = len(all_vars)
    n_rows = np.ceil(n_vars / max_cols_per_row).astype(int)
    n_cols = min(n_vars, max_cols_per_row)
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 6*n_rows), squeeze=False)
    axs = axs.ravel()

    # Suppress specific warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=(FutureWarning, RuntimeWarning, UserWarning))

        # Print group counts
        group_counts = df[group_col].value_counts()
        print(f"Total participants = {len(df)}")
        for group, count in group_counts.items():
            print(f"{group} participants = {count}")

        # Plotting each variable
        order = group_order
        for i, var in enumerate(all_vars):
            current_ax = axs[i]
            if df[var].name in cat_vars:
                sns.countplot(data=df, x=group_col, hue=var, ax=current_ax, order=order)
                _, _, stats = pg.chi2_independence(data=df, x=var, y=group_col, correction=True)
                chi2_stats = stats.loc[stats['test']=='log-likelihood']
                # Create the annotation text with chi-squared results
                chi2_text = f"p = {chi2_stats['pval'].values[0]:.3e}"
                
                # Annotate the plot with chi-squared results
                current_ax.annotate(chi2_text, xy=(0.03, 0.97), xycoords='axes fraction',
                                    ha='left', va='top', fontsize='small',
                                    bbox=dict(boxstyle='round,pad=0.5', fc='white', alpha=0.5))
                
                # Set variable name as title
                current_ax.set_xlabel('')
                current_ax.set_ylabel(var)

                legend = current_ax.legend()
                legend.set_title("")
            else:
                sns.violinplot(data=df, x=group_col, y=var, ax=current_ax, order=order)
                current_ax.set_xlabel('')
                current_ax.set_ylabel(var)

                # Calculate pairwise comparisons
                pairwise_comparisons = list(combinations(order, 2))
                
                # Use statannotations for adding statistical annotations
                annotator = Annotator(current_ax, pairwise_comparisons, data=df, x=group_col, y=var, 
                                      order=order, verbose=0)
                annotator.configure(test='t-test_ind', text_format='star', loc='outside',
                                    hide_non_significant=True, comparisons_correction='bonferroni',
                                    pvalue_thresholds=[[1e-4, "****"], [1e-3, "***"], [1e-2, "**"], [0.05, "*"]])
                annotator.apply_test().annotate()

        # Remove any unused axes
        for ax in axs[n_vars:]:
            ax.remove()

        plt.suptitle(plot_title, y=1.0025, fontsize='x-large')
        plt.tight_layout()
        plt.show()

def plot_histogram(data, title, all_regions):
    """Plot histogram for top tau regions."""
    data = data.reindex(all_regions).fillna(0).sort_values(ascending=True)
    
    # Create the horizontal bar plot
    plt.figure(figsize=(6, 8))
    data.plot(kind='barh')
    plt.title(title)
    plt.xlabel('Frequency of top tau PET SUVR (%)')
    plt.ylabel('Regions')
    plt.show()