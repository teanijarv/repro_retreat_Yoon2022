import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

def determine_amyloid_status(row):
    """Set amyloid-positivity status for each row (subject) based on amyloid-PET or CSF."""
    # If there is amyloid-PET data, set to 1 if value larger than 1.033
    if pd.notna(row['fnc_ber_com_composite']):
        return int(row['fnc_ber_com_composite'] > 1.033)
    # Else if there is CSF data only, set to 1 if the value is 1
    elif pd.notna(row['Abnormal_CSF_Ab42_Ab40_Ratio']):
        return int(row['Abnormal_CSF_Ab42_Ab40_Ratio'] == 1)
    return np.nan

def compute_roi_tau(df, ebm_regions, side_substrs):
    """Compute ROI tau using tau-PET and weighing them with regional volume sizes."""
    numerator = 0
    denominator = 0

    # Loop through all EBM regions for both hemispheres
    for region in ebm_regions:
        for side in side_substrs:
            # Identify tau and volume columns based on region and side
            tau_col = next((col for col in df.columns if region in col 
                            and "tnic_sr_mr_fs_" in col and side in col), None)
            vol_col = next((col for col in df.columns if region in col 
                            and "tnic_vx_fs_" in col and side in col), None)
            
            # If both tau and volume columns are found, update the numerator and denominator
            if tau_col and vol_col:
                numerator += df[tau_col] * df[vol_col]
                denominator += df[vol_col]

    # Calculate EBM tau value
    roi_tau = numerator / denominator

    return roi_tau

def assign_cu_ci_group(row):
    """Assign to either CI or CU group."""
    if row['amyloid_positive'] == 0 and row['diagnosis_baseline_variable'] in ['Normal', 'SCD']:
        return 0  # CU group
    elif row['amyloid_positive'] == 1 and row['diagnosis_baseline_variable'] in ['MCI', 'AD']:
        return 1  # CI group
    else:
        return np.nan

def find_cutoff_and_plot_roc(y_true, y_scores, roi_name='roi', plot_roc=False):
    # Compute ROC curve and ROC area
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    # Find the optimal threshold (Youden index)
    J = tpr - fpr
    ix = np.argmax(J)
    optimal_threshold = thresholds[ix]

    # Output the optimal cutoff value and AUC
    print(f'{roi_name} cut-off calculation')
    print(f'Optimal cutoff value: {optimal_threshold}')
    print(f'AUC: {roc_auc}')

    # Plot ROC curve if requested
    if plot_roc:
        plt.figure(figsize=(6, 4), dpi=150)  # Adjust the size and increase resolution with dpi
        plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='grey', lw=1, linestyle='--')
        plt.plot(fpr[ix], tpr[ix], 'o', markersize=5, label=f'Optimal cutoff (Youden index)', fillstyle="none", color='black', mew=1)
        plt.annotate(f'({fpr[ix]:.3f}, {tpr[ix]:.3f})', xy=(fpr[ix], tpr[ix]), xytext=(fpr[ix] + 0.3, tpr[ix] - 0.1),
                     arrowprops=dict(facecolor='black', shrink=0.05), horizontalalignment='right', verticalalignment='top', fontsize=8)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('1 - Specificity (False Positive Rate)')
        plt.ylabel('Sensitivity (True Positive Rate)')
        plt.title(f'ROC Curve for {roi_name} (criterion: Youden)')
        plt.legend(loc="lower right", fontsize='small')
        plt.grid(True, linestyle='--', linewidth=0.5, color='gray', alpha=0.7)  # Adjust grid appearance
        plt.show()

    return optimal_threshold, roc_auc

    
def is_above_cutoff(series, cutoff):
    """Check if values in series are above cut-off."""
    return series > cutoff

def check_tau_positivity(row):
    """Identify tau positivity based on Braak 1 or Braak 34."""
    if row['is_braak1']: return 1
    elif row['is_braak34']: return 1
    else: return 0

def assign_at_group(row):
    """Assign AT group."""
    if row['amyloid_positive'] == 0 and row['tau_positive'] == 0:
        return 'A-T-'
    elif row['amyloid_positive'] == 0 and row['tau_positive'] == 1:
        return 'A-T+'
    elif row['amyloid_positive'] == 1 and row['tau_positive'] == 1:
        return 'A+T+'
    else:
        return np.nan