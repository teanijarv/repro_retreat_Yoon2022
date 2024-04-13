import pandas as pd
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import statsmodels.api as sm
from statsmodels.formula.api import ols
from scipy.stats import kruskal, mannwhitneyu, chi2_contingency

class colours:
    """Terminal color codes"""
    CEND = '\33[0m'
    CBLUE = '\33[94m'
    CGREEN = '\33[92m'
    CRED = '\33[91m'
    CYELLOW = '\33[93m'
    UBOLD = '\33[1m'

def descriptive_stats(df, group_col, continuous_vars):
    """Calculate basic statistics for continuous variables."""
    desc_stats = df.groupby(group_col)[continuous_vars].agg(['mean', 'std', 'count']).reset_index()
    return desc_stats

def categorical_frequencies(df, group_col, categorical_vars):
    """Calculate frequencies for categorical variables."""
    freq_tables = {var: pd.crosstab(df[var], df[group_col], margins=True) for var in categorical_vars}
    return freq_tables

def anova_posthoc(df, group_col, continuous_var):
    """Perform ANOVA and pairwise Tukey HSD post-hoc tests for continuous variables."""
    model = ols(f'{continuous_var} ~ C({group_col})', data=df).fit()
    anova_result = sm.stats.anova_lm(model, typ=2)
    print('\nANOVA result for ' + colours.CGREEN + f'{continuous_var}' + colours.CEND + ':')
    display(anova_result)
    
    # If the ANOVA is significant, perform post-hoc test
    if anova_result['PR(>F)'].iloc[0] < 0.05:
        tukey = pairwise_tukeyhsd(df[continuous_var], df[group_col])
        print(tukey)

def kruskal_wallis_posthoc(df, group_col, continuous_var):
    """Perform Kruskal-Wallis and post-hoc tests for continuous variables."""
    # Prepare data for Kruskal-Wallis
    groups = df.groupby(group_col)[continuous_var].apply(list).values
    kruskal_result = kruskal(*groups)
    
    print('\nKruskal-Wallis result for ' + colours.CGREEN + f'{continuous_var}' + colours.CEND + ':')
    print(f'H-statistic: {kruskal_result.statistic:.4f}, p-value: {kruskal_result.pvalue:.4f}')
    
    # If the Kruskal-Wallis test is significant, perform post-hoc tests
    if kruskal_result.pvalue < 0.05:
        # Conduct Mann-Whitney U tests
        print('Post-hoc pairwise Mann-Whitney U tests for ' + colours.CGREEN + f'{continuous_var}' + colours.CEND + ':')
        # Get unique pairs for post-hoc comparisons
        unique_groups = df[group_col].unique()
        comparisons = [(i, j) for i in unique_groups for j in unique_groups if i < j]
        for group1, group2 in comparisons:
            data1 = df[df[group_col] == group1][continuous_var]
            data2 = df[df[group_col] == group2][continuous_var]
            u_statistic, p_value = mannwhitneyu(data1, data2, alternative='two-sided')
            # Bonferroni correction for multiple comparisons
            corrected_p_value = p_value * len(comparisons)
            print(f'{group1} vs {group2}: U-statistic={u_statistic:.4f}, p-value={corrected_p_value:.4f} (Bonferroni corrected)')

def chi_squared_test(df, group_col, categorical_var):
    """Perform Chi-Squared tests for categorical variables."""
    contingency_table = pd.crosstab(df[categorical_var], df[group_col])
    chi2, p, dof, expected = chi2_contingency(contingency_table)
    
    print('\nChi-Squared test result for ' + colours.CGREEN + f'{categorical_var}' + colours.CEND + ':')
    print(f'Chi2-statistic: {chi2:.4f}, p-value: {p:.4f}, dof: {dof}')

def ancova_posthoc(df, dependent_var, factor_var, covariates):
    """Perform ANCOVA and pairwise Bonferroni-corrected post-hoc tests for continuous variables."""
    # Define the model formula for ANCOVA
    formula = f'{dependent_var} ~ C({factor_var}) + ' + ' + '.join(covariates)
    model = ols(formula, data=df).fit()
    ancova_result = sm.stats.anova_lm(model, typ=2)
    
    print('\nANCOVA result for ' + colours.CGREEN + f'{dependent_var}' + colours.CEND + ':')
    display(ancova_result)

    # If the ANCOVA is significant, perform post-hoc comparisons using Bonferroni correction
    if ancova_result['PR(>F)'].iloc[0] < 0.05:
        print('Performing pairwise comparisons with Bonferroni correction for ' + colours.CGREEN + f'{dependent_var}' + colours.CEND + ':')
        
        # Get unique groups
        unique_groups = df[factor_var].dropna().unique()
        
        # Perform pairwise comparisons and apply Bonferroni correction
        comparisons = []
        for i in range(len(unique_groups)):
            for j in range(i+1, len(unique_groups)):
                # Construct two sub-dataframes for each group
                group1 = df[df[factor_var] == unique_groups[i]]
                group2 = df[df[factor_var] == unique_groups[j]]

                # Combine group data for pairwise comparison
                combined_data = pd.concat([group1, group2], axis=0)
                
                # Run the ANCOVA model on the combined data
                pairwise_model = ols(formula, data=combined_data).fit()
                pairwise_lm = sm.stats.anova_lm(pairwise_model, typ=2)
                
                # Extract the p-value for the group effect
                group_effect_p = pairwise_lm.loc[f'C({factor_var})', 'PR(>F)']
                
                # Add the p-value to the comparisons list
                comparisons.append((unique_groups[i], unique_groups[j], group_effect_p))
        
        # Correct for multiple comparisons
        bonferroni_correction_factor = len(comparisons)
        bonferroni_corrected_p_values = [(group1, group2, p * bonferroni_correction_factor)
                                         for group1, group2, p in comparisons]

        # Display the results
        for comparison in bonferroni_corrected_p_values:
            group1, group2, corrected_p = comparison
            print(f'{group1} vs {group2}: p={corrected_p:.6f} (Bonferroni corrected)')
