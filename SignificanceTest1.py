"""
Wilcoxon Signed-Rank Test Using RAW PREDICTIONS
Compares baseline vs all 15 filters using sample-level prediction data
Creates 16x16 p-value matrices for 3 classifiers

Much more statistically valid than using AUC summaries!
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import wilcoxon
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================
BASE = Path(__file__).resolve().parent

CLASSIFIERS = {
    'ViT-B16': {
        'dir': BASE / "ViT_b16_Main" / "ViT_b16_DIP" / "test_results",
        'prefix': 'preds'
    },
    'DenseNet': {
        'dir': BASE / "DenseNet_Main" / "DenseNet_DIP" / "test_results",
        'prefix': 'preds'
    },
    'ConvFormer': {
        'dir': BASE / "ConvFormer_Main" / "ConvFormer_DIP" / "test_results",
        'prefix': 'preds'
    }
}
# Filter names (must match your file names)
FILTER_NAMES = [
    'baseline',
    'clahe_clip1.0',
    'clahe_clip2.0',
    'clahe_clip3.0',
    'clahe_clip4.0',
    'clahe_clip5.0',
    'gaussian_k3',
    'gaussian_k5',
    'gaussian_k7',
    'gaussian_k9',
    'bilateral_d5',
    'bilateral_d7',
    'bilateral_d9',
    'bilateral_d11',
    'hist_eq'
]

# Output directory
OUTPUT_DIR = BASE / "Significance_Test_Results"
OUTPUT_DIR.mkdir(exist_ok=True)


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def load_predictions(pred_dir, prefix, filter_name):
    """Load prediction file for a specific filter"""
    filepath = pred_dir / f'{prefix}_{filter_name}.npy'
    try:
        preds = np.load(filepath)
        return preds.astype(np.float32)  # Convert from float16 to float32
    except FileNotFoundError:
        print(f"  ⚠️ File not found: {filepath}")
        return None


def compute_wilcoxon_from_predictions(predictions_dict):
    """
    Compute 16x16 p-value matrix using Wilcoxon signed-rank test on RAW PREDICTIONS
    
    For each pair of filters, we compare their predictions across all samples and diseases.
    This is more statistically valid than comparing AUC summaries.
    
    Args:
        predictions_dict: dict with filter_name -> predictions array (samples × diseases)
    
    Returns:
        p_matrix: 16x16 matrix of p-values
        filter_names: ordered list of filter names
    """
    
    # Sort filters
    filter_names = sorted(predictions_dict.keys())
    if 'baseline' in filter_names:
        filter_names.remove('baseline')
        filter_names = ['baseline'] + filter_names
    
    n_filters = len(filter_names)
    p_matrix = np.ones((n_filters, n_filters))
    
    print("  Computing pairwise Wilcoxon tests...")
    
    # Compute pairwise Wilcoxon tests
    for i, filter_i in enumerate(filter_names):
        for j, filter_j in enumerate(filter_names):
            if i == j:
                p_matrix[i, j] = 1.0  # Self comparison
            elif i < j:
                preds_i = predictions_dict[filter_i]  # (samples, 14)
                preds_j = predictions_dict[filter_j]  # (samples, 14)
                
                # Flatten both to compare all predictions
                # This treats each prediction as an independent observation
                flat_i = preds_i.flatten()  # 25596 * 14 = 358,344 values
                flat_j = preds_j.flatten()
                
                try:
                    # Wilcoxon signed-rank test (two-sided)
                    # H0: distributions are identical
                    # H1: distributions differ
                    stat, pval = wilcoxon(flat_i, flat_j, alternative='two-sided')
                    p_matrix[i, j] = pval
                    p_matrix[j, i] = pval  # Symmetric
                except Exception as e:
                    print(f"    Warning: Could not compare {filter_i} vs {filter_j}: {e}")
                    p_matrix[i, j] = 1.0
                    p_matrix[j, i] = 1.0
    
    return p_matrix, filter_names


def create_heatmap(p_matrix, filter_names, classifier_name, ax):
    """
    Create heatmap with significance markers
    
    Args:
        p_matrix: 16x16 p-value matrix
        filter_names: list of filter names
        classifier_name: name of classifier (for title)
        ax: matplotlib axis
    """
    
    # Create masked array for upper triangle (symmetric, so only show lower)
    mask = np.triu(np.ones_like(p_matrix, dtype=bool), k=1)
    
    # Log scale for better visualization (-log10(p))
    log_p = -np.log10(np.maximum(p_matrix, 1e-10))
    np.fill_diagonal(log_p, 0)
    
    # Plot heatmap
    sns.heatmap(
        log_p,
        mask=mask,
        annot=False,
        fmt='.2f',
        cmap='coolwarm',
        cbar_kws={'label': '-log₁₀(p-value)', 'shrink': 0.8},
        xticklabels=filter_names,
        yticklabels=filter_names,
        ax=ax,
        vmin=0,
        vmax=4,  # Corresponds to p=0.0001
        square=True,
        linewidths=0.5,
        linecolor='gray',
    )
    
    # Add significance markers
    for i in range(len(filter_names)):
        for j in range(len(filter_names)):
            if i >= j: 
                pval = p_matrix[i, j]
                
                if pval < 0.001:
                    marker = '***'
                elif pval < 0.01:
                    marker = '**'
                elif pval < 0.05:
                    marker = '*'
                else:
                    marker = ''
                
                if marker:
                    ax.text(j + 0.5, i + 0.7, marker, 
                           ha='center', va='center', 
                           color='white', fontsize=12, fontweight='bold')
    
    ax.set_title(classifier_name, fontsize=14, fontweight='bold', pad=10)
    ax.set_xlabel('')
    ax.set_ylabel('')
    
    # Rotate labels for readability
    ax.set_xticklabels(filter_names, rotation=45, ha='right', fontsize=9)
    ax.set_yticklabels(filter_names, rotation=0, fontsize=9)


def create_performance_barchart(predictions_dict, classifier_name, ax):
    """
    Create bar chart showing mean prediction difference compared to baseline
    Green = filter predictions higher than baseline, Red = lower
    
    Args:
        predictions_dict: dict with filter_name -> predictions array
        classifier_name: name of classifier
        ax: matplotlib axis
    """
    
    # Calculate mean prediction for each filter across all samples/diseases
    baseline_mean = np.mean(predictions_dict['baseline'])
    filter_names = sorted(predictions_dict.keys())
    filter_names.remove('baseline')
    
    # Calculate differences from baseline
    pred_diffs = []
    for filter_name in filter_names:
        filter_mean = np.mean(predictions_dict[filter_name])
        diff = filter_mean - baseline_mean
        pred_diffs.append(diff)
    
    # Color code: green if higher, red if lower
    colors = ['green' if diff > 0 else 'red' for diff in pred_diffs]
    
    # Create bar chart
    bars = ax.bar(range(len(filter_names)), pred_diffs, color=colors, alpha=0.7, 
                  edgecolor='black', linewidth=1.5)
    
    # Add zero line (baseline)
    ax.axhline(y=0, color='black', linestyle='--', linewidth=2, label='Baseline')
    
    # Labels and formatting
    ax.set_xticks(range(len(filter_names)))
    ax.set_xticklabels(filter_names, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('Mean Prediction Difference vs Baseline', fontsize=11, fontweight='bold')
    ax.set_title(f'{classifier_name}: Prediction Shift', fontsize=12, fontweight='bold')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add value labels on bars
    for i, (bar, diff) in enumerate(zip(bars, pred_diffs)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{diff:+.4f}',
               ha='center', va='bottom' if diff > 0 else 'top',
               fontsize=8, fontweight='bold')


def print_significant_pairs(p_matrix, filter_names, classifier_name, alpha=0.05):
    """Print baseline vs. filter comparisons that are significant"""
    
    print(f"\n{'='*70}")
    print(f"Significant Comparisons for {classifier_name} (α={alpha})")
    print(f"{'='*70}")
    
    # Baseline index
    baseline_idx = filter_names.index('baseline')
    baseline_pvals = p_matrix[baseline_idx, :]
    
    # Get significant filters
    sig_filters = []
    for i, (filter_name, pval) in enumerate(zip(filter_names, baseline_pvals)):
        if filter_name != 'baseline' and pval < alpha:
            sig_filters.append((filter_name, pval))
    
    if sig_filters:
        sig_filters.sort(key=lambda x: x[1])  # Sort by p-value
        print(f"\nFilters with SIGNIFICANTLY DIFFERENT predictions from baseline:")
        print(f"{'Filter':<25} {'p-value':<15} {'Significance':<15}")
        print(f"{'-'*55}")
        for filter_name, pval in sig_filters:
            if pval < 0.001:
                sig = '***'
            elif pval < 0.01:
                sig = '**'
            else:
                sig = '*'
            print(f"{filter_name:<25} {pval:<15.4e} {sig:<15}")
    else:
        print(f"No filters with significantly different predictions (α={alpha})")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main pipeline"""
    
    print("="*70)
    print("WILCOXON TEST ON RAW MODEL PREDICTIONS")
    print("="*70)
    print(f"Number of Classifiers: {len(CLASSIFIERS)}")
    print(f"Classifiers: {list(CLASSIFIERS.keys())}")
    print(f"Filters per classifier: {len(FILTER_NAMES)}")
    print("="*70)
    
    # Load all predictions
    all_predictions = {}
    
    for classifier_name, classifier_config in CLASSIFIERS.items():
        pred_dir = Path(classifier_config['dir'])
        prefix = classifier_config['prefix']
        
        print(f"\n📂 Loading predictions for {classifier_name}")
        print(f"   Directory: {pred_dir}")
        print(f"   Prefix: {prefix}")
        
        predictions_dict = {}
        for filter_name in FILTER_NAMES:
            preds = load_predictions(pred_dir, prefix, filter_name)
            if preds is not None:
                predictions_dict[filter_name] = preds
                print(f"  ✅ {filter_name}: shape {preds.shape}")
        
        if len(predictions_dict) == len(FILTER_NAMES):
            all_predictions[classifier_name] = predictions_dict
            print(f"  ✓ All {len(predictions_dict)} filters loaded for {classifier_name}")
        else:
            print(f"  ⚠️ Only {len(predictions_dict)}/{len(FILTER_NAMES)} filters found")
    
    if not all_predictions:
        print("❌ No predictions loaded. Check paths and file names.")
        return
    
    # Compute p-value matrices
    print("\n" + "="*70)
    print("COMPUTING WILCOXON P-VALUES")
    print("="*70)
    
    all_pmatrices = {}
    all_filternames = {}
    
    for classifier_name, predictions_dict in all_predictions.items():
        print(f"\n🔬 Computing p-values for {classifier_name}...")
        p_matrix, filter_names = compute_wilcoxon_from_predictions(predictions_dict)
        all_pmatrices[classifier_name] = p_matrix
        all_filternames[classifier_name] = filter_names
    
    # ========== FIGURE 1: Heatmaps (p-values) ==========
    print("\n" + "="*70)
    print("CREATING VISUALIZATIONS")
    print("="*70)
    
    fig1, axes1 = plt.subplots(1, 3, figsize=(24, 8))
    fig1.suptitle('Wilcoxon Signed-Rank Test: Raw Prediction Comparison (p-values)',
                  fontsize=16, fontweight='bold', y=1.02)
    
    for (classifier_name, p_matrix), (_, filter_names), ax in zip(
        all_pmatrices.items(), 
        all_filternames.items(), 
        axes1
    ):
        create_heatmap(p_matrix, filter_names, classifier_name, ax)
        print_significant_pairs(p_matrix, filter_names, classifier_name, alpha=0.05)
    
    fig1.tight_layout()
    output_path1 = OUTPUT_DIR / 'wilcoxon_raw_predictions_heatmap.png'
    fig1.savefig(output_path1, dpi=300, bbox_inches='tight')
    print(f"\n✅ Heatmap saved to: {output_path1}")
    
    # ========== FIGURE 2: Bar Charts (Performance) ==========
    fig2, axes2 = plt.subplots(1, 3, figsize=(24, 8))
    fig2.suptitle('Mean Prediction Shift vs Baseline',
                  fontsize=16, fontweight='bold', y=1.02)
    
    for (classifier_name, predictions_dict), ax in zip(all_predictions.items(), axes2):
        create_performance_barchart(predictions_dict, classifier_name, ax)
    
    fig2.tight_layout()
    output_path2 = OUTPUT_DIR / 'prediction_shift_barchart.png'
    fig2.savefig(output_path2, dpi=300, bbox_inches='tight')
    print(f"✅ Performance chart saved to: {output_path2}")
    
    # Save numeric p-values to CSV
    print("\n" + "="*70)
    print("SAVING RESULTS")
    print("="*70)
    
    for classifier_name, p_matrix in all_pmatrices.items():
        filter_names = all_filternames[classifier_name]
        csv_path = OUTPUT_DIR / f'pvalues_{classifier_name.replace("/", "_").replace(" ", "_")}.csv'
        
        with open(csv_path, 'w') as f:
            f.write(',' + ','.join(filter_names) + '\n')
            for i, name_i in enumerate(filter_names):
                f.write(name_i)
                for j, name_j in enumerate(filter_names):
                    f.write(f',{p_matrix[i, j]:.4e}')
                f.write('\n')
        
        print(f"✅ P-value table saved to: {csv_path}")
    
    print(f"\n{'='*70}")
    print("✅ ANALYSIS COMPLETE!")
    print(f"{'='*70}\n")
    plt.show()


if __name__ == '__main__':
    main()