"""
Combine all ablation test results into a single JSON file
Reads test_results/results_*.csv files and creates combined_results.json
"""

import os
import json
import pandas as pd
import glob

# ==================== CONFIGURATION ====================
RESULTS_DIR = "test_results"
OUTPUT_JSON = "combined_results.json"

# Mapping of CSV filename to filter configuration
# (extracted from the naming convention in your test script)
def extract_config_from_filename(filename):
    """Extract model name, filter type, and parameters from filename"""
    name = filename.replace('results_', '').replace('.csv', '')
    
    # Define mapping based on your naming convention
    if name == 'baseline':
        return {'name': name, 'filter_type': 'none', 'parameters': {}}
    elif name == 'hist_eq':
        return {'name': name, 'filter_type': 'hist_eq', 'parameters': {}}
    elif name.startswith('clahe_clip'):
        clip = float(name.replace('clahe_clip', ''))
        return {'name': name, 'filter_type': 'clahe', 'parameters': {'clip_limit': clip}}
    elif name.startswith('gaussian_k'):
        kernel = int(name.replace('gaussian_k', ''))
        return {'name': name, 'filter_type': 'gaussian', 'parameters': {'kernel': kernel}}
    elif name.startswith('bilateral_d'):
        d = int(name.replace('bilateral_d', ''))
        return {'name': name, 'filter_type': 'bilateral', 'parameters': {'d': d}}
    else:
        return {'name': name, 'filter_type': 'unknown', 'parameters': {}}

# ==================== MAIN ====================
def main():
    print("="*60)
    print("COMBINING ABLATION TEST RESULTS INTO JSON")
    print("="*60)
    
    # Find all result CSV files
    csv_files = glob.glob(os.path.join(RESULTS_DIR, "results_*.csv"))
    
    if not csv_files:
        print(f"❌ No result files found in {RESULTS_DIR}/")
        print("   Make sure you have run the test script first!")
        return
    
    print(f"\n📁 Found {len(csv_files)} result files")
    
    all_results = []
    skipped = []
    
    for csv_path in sorted(csv_files):
        filename = os.path.basename(csv_path)
        
        # Extract configuration
        config = extract_config_from_filename(filename)
        model_name = config['name']
        
        print(f"\n📊 Processing: {model_name}")
        
        # Read CSV file
        df = pd.read_csv(csv_path)
        
        # Verify required columns exist
        if 'AUC' not in df.columns:
            print(f"   ⚠️ Skipping: 'AUC' column not found")
            skipped.append(model_name)
            continue
        
        # Extract per-class AUC (should be 14 values)
        per_class_auc = df['AUC'].tolist()
        
        if len(per_class_auc) != 14:
            print(f"   ⚠️ Warning: Expected 14 diseases, got {len(per_class_auc)}")
        
        # Build result entry
        result_entry = {
            "name": model_name,
            "filter_type": config['filter_type'],
            "parameters": str(config['parameters']),
            "mean_test_auc": df['AUC'].mean(),
            "per_class_auc": per_class_auc,
            "diseases": df['Disease'].tolist() if 'Disease' in df.columns else None
        }
        
        all_results.append(result_entry)
        print(f"   ✅ Mean AUC: {result_entry['mean_test_auc']:.4f}")
    
    # Sort by mean AUC (descending)
    all_results.sort(key=lambda x: x['mean_test_auc'], reverse=True)
    
    # Save to JSON
    with open(OUTPUT_JSON, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print("\n" + "="*60)
    print("✅ RESULTS COMBINED SUCCESSFULLY!")
    print("="*60)
    print(f"\n📁 Output file: {OUTPUT_JSON}")
    print(f"📊 Total models processed: {len(all_results)}")
    
    if skipped:
        print(f"⚠️ Skipped files: {skipped}")
    
    # Print summary
    print("\n📈 TOP 5 MODELS BY TEST AUC:")
    print("-" * 40)
    for i, r in enumerate(all_results[:5]):
        print(f"   {i+1}. {r['name']:20s}: {r['mean_test_auc']:.4f}")

if __name__ == '__main__':
    main()