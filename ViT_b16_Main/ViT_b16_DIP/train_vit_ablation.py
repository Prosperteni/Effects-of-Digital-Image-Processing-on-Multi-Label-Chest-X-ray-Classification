"""
DenseNet-121 Ablation Study - Testing on Official NIH Test Set
Tests all 15 trained models on 25,596 test images
"""

import os
import cv2
import json
import pandas as pd
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# ==================== OPTIMIZATION FLAGS ====================
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# ==================== CONFIGURATION ====================
DATA_DIR = os.path.join("..", "dataset", "images-224", "images-224")
CSV_PATH = os.path.join("..", "dataset", "Data_Entry_2017.csv")
TEST_FILE = os.path.join("..", "dataset", "test_list_NIH.txt")
BATCH_SIZE = 128
IMG_SIZE = 224
NUM_CLASSES = 14
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PATHOLOGIES = [
    'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 
    'Mass', 'Nodule', 'Pneumonia', 'Pneumothorax', 
    'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 
    'Pleural_Thickening', 'Hernia'
]

# ==================== FILTER FUNCTIONS ====================
def apply_clahe(image, clip_limit=2.0, tile_grid=8):
    if isinstance(image, Image.Image):
        image = np.array(image)
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid, tile_grid))
    l_eq = clahe.apply(l)
    lab_eq = cv2.merge((l_eq, a, b))
    enhanced = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2RGB)
    return Image.fromarray(enhanced)

def apply_gaussian(image, kernel=5, sigma=1.0):
    if isinstance(image, Image.Image):
        image = np.array(image)
    filtered = cv2.GaussianBlur(image, (kernel, kernel), sigma)
    return Image.fromarray(filtered)

def apply_bilateral(image, d=9, sigma_color=75, sigma_space=75):
    if isinstance(image, Image.Image):
        image = np.array(image)
    filtered = cv2.bilateralFilter(image, d, sigma_color, sigma_space)
    return Image.fromarray(filtered)

def apply_histogram_equalization(image):
    if isinstance(image, Image.Image):
        image = np.array(image)
    yuv = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
    yuv[:,:,0] = cv2.equalizeHist(yuv[:,:,0])
    enhanced = cv2.cvtColor(yuv, cv2.COLOR_YUV2RGB)
    return Image.fromarray(enhanced)

def apply_filter(image, filter_type, params):
    if filter_type == "clahe":
        return apply_clahe(image, params.get('clip_limit', 2.0), params.get('tile_grid', 8))
    elif filter_type == "gaussian":
        return apply_gaussian(image, params.get('kernel', 5), params.get('sigma', 1.0))
    elif filter_type == "bilateral":
        return apply_bilateral(image, params.get('d', 9), params.get('sigma_color', 75), params.get('sigma_space', 75))
    elif filter_type == "hist_eq":
        return apply_histogram_equalization(image)
    else:
        return image

# ==================== DATASET CLASS ====================
class NIHChestXrayTestDataset(Dataset):
    def __init__(self, img_list, csv_data, img_dir, transform=None, filter_type=None, filter_params=None):
        self.img_list = img_list
        self.img_dir = img_dir
        self.transform = transform
        self.filter_type = filter_type
        self.filter_params = filter_params if filter_params else {}
        self.labels_dict = {}
        for _, row in csv_data.iterrows():
            self.labels_dict[row['Image Index']] = row
        
    def __len__(self):
        return len(self.img_list)
    
    def __getitem__(self, idx):
        img_name = self.img_list[idx]
        img_path = os.path.join(self.img_dir, img_name).replace('\\', '/')
        
        image = Image.open(img_path).convert('RGB')
        
        if self.filter_type and self.filter_type != "none":
            image = apply_filter(image, self.filter_type, self.filter_params)
        
        row = self.labels_dict.get(img_name)
        if row is None:
            labels = np.zeros(NUM_CLASSES, dtype=np.float32)
        else:
            finding_labels = row['Finding Labels'].split('|')
            labels = np.array([1.0 if p in finding_labels else 0.0 for p in PATHOLOGIES], dtype=np.float32)
        
        if self.transform:
            image = self.transform(image)
        
        return image, torch.from_numpy(labels), img_name

# ==================== MODEL CREATION ====================
def create_densenet_model(num_classes=14):
    model = models.densenet121(pretrained=False)
    num_features = model.classifier.in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(num_features, num_classes)
    )
    return model

# ==================== EVALUATION FUNCTION ====================
def evaluate_test(model, loader, criterion):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    all_img_names = []
    
    with torch.no_grad():
        for images, labels, img_names in tqdm(loader, desc="Testing"):
            images = images.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)
            
            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            
            all_preds.append(torch.sigmoid(outputs).cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            all_img_names.extend(img_names)
    
    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    
    results = []
    per_class_auc = []
    
    for i, pathology in enumerate(PATHOLOGIES):
        try:
            auc = roc_auc_score(all_labels[:, i], all_preds[:, i])
            per_class_auc.append(auc)
        except:
            auc = 0.0
            per_class_auc.append(0.0)
        
        pred_binary = (all_preds[:, i] > 0.5).astype(int)
        f1 = f1_score(all_labels[:, i], pred_binary, zero_division=0)
        
        tn, fp, fn, tp = confusion_matrix(all_labels[:, i], pred_binary, labels=[0,1]).ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
        
        results.append({
            'Disease': pathology,
            'AUC': auc,
            'F1_Score': f1,
            'Sensitivity': sensitivity,
            'Specificity': specificity,
            'Accuracy': accuracy,
            'Prevalence': all_labels[:, i].mean()
        })
    
    results_df = pd.DataFrame(results)
    mean_auc = np.mean(per_class_auc)
    os.makedirs('test_results/predictions', exist_ok=True)
    np.save(f'test_results/predictions/{model_config["name"]}_labels.npy', all_labels)
    np.save(f'test_results/predictions/{model_config["name"]}_preds.npy', all_preds)
    print(f"    💾 Saved predictions for {model_config['name']}")
    return results_df, mean_auc, all_labels, all_preds, all_img_names


# ==================== ALL MODELS TO TEST ====================
# Based on the ablation study training results (sorted by AUC)
MODELS_TO_TEST = [
    {'name': 'baseline', 'path': 'models/best_baseline.pth', 'filter_type': 'none', 'params': {}},
    {'name': 'clahe_clip2.0', 'path': 'models/best_clahe_clip2.0.pth', 'filter_type': 'clahe', 'params': {'clip_limit': 2.0}},
    {'name': 'clahe_clip1.0', 'path': 'models/best_clahe_clip1.0.pth', 'filter_type': 'clahe', 'params': {'clip_limit': 1.0}},
    {'name': 'clahe_clip4.0', 'path': 'models/best_clahe_clip4.0.pth', 'filter_type': 'clahe', 'params': {'clip_limit': 4.0}},
    {'name': 'bilateral_d5', 'path': 'models/best_bilateral_d5.pth', 'filter_type': 'bilateral', 'params': {'d': 5}},
    {'name': 'hist_eq', 'path': 'models/best_hist_eq.pth', 'filter_type': 'hist_eq', 'params': {}},
    {'name': 'clahe_clip5.0', 'path': 'models/best_clahe_clip5.0.pth', 'filter_type': 'clahe', 'params': {'clip_limit': 5.0}},
    {'name': 'gaussian_k9', 'path': 'models/best_gaussian_k9.pth', 'filter_type': 'gaussian', 'params': {'kernel': 9}},
    {'name': 'clahe_clip3.0', 'path': 'models/best_clahe_clip3.0.pth', 'filter_type': 'clahe', 'params': {'clip_limit': 3.0}},
    {'name': 'gaussian_k7', 'path': 'models/best_gaussian_k7.pth', 'filter_type': 'gaussian', 'params': {'kernel': 7}},
    {'name': 'gaussian_k3', 'path': 'models/best_gaussian_k3.pth', 'filter_type': 'gaussian', 'params': {'kernel': 3}},
    {'name': 'gaussian_k5', 'path': 'models/best_gaussian_k5.pth', 'filter_type': 'gaussian', 'params': {'kernel': 5}},
    {'name': 'bilateral_d7', 'path': 'models/best_bilateral_d7.pth', 'filter_type': 'bilateral', 'params': {'d': 7}},
    {'name': 'bilateral_d9', 'path': 'models/best_bilateral_d9.pth', 'filter_type': 'bilateral', 'params': {'d': 9}},
    {'name': 'bilateral_d11', 'path': 'models/best_bilateral_d11.pth', 'filter_type': 'bilateral', 'params': {'d': 11}},
]

# ==================== PLOTTING FUNCTIONS ====================
def plot_comparison_bar(results_df, baseline_auc, save_path='test_results/densenet_comparison.png'):
    plt.figure(figsize=(14, 8))
    
    names = results_df['name'].values
    aucs = results_df['mean_test_auc'].values
    
    colors = []
    for name in names:
        if name == 'baseline':
            colors.append('gray')
        elif 'clahe' in name:
            colors.append('green')
        elif 'gaussian' in name:
            colors.append('blue')
        elif 'bilateral' in name:
            colors.append('orange')
        elif 'hist_eq' in name:
            colors.append('gold')
        else:
            colors.append('red')
    
    bars = plt.bar(names, aucs, color=colors, alpha=0.7)
    plt.axhline(y=baseline_auc, color='gray', linestyle='--', linewidth=2, label=f'Baseline: {baseline_auc:.4f}')
    
    plt.ylabel('Mean Test AUC', fontsize=12)
    plt.xlabel('Experiment', fontsize=12)
    plt.title('DenseNet-121 Ablation Study: Test AUC for Different Filters', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0.75, 0.85)
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"✅ Comparison chart saved to {save_path}")

def plot_improvement_chart(results_df, baseline_auc, save_path='test_results/densenet_improvement.png'):
    plt.figure(figsize=(14, 6))
    
    results_df = results_df.copy()
    results_df['improvement'] = results_df['mean_test_auc'] - baseline_auc
    
    colors = ['green' if x > 0 else 'red' for x in results_df['improvement']]
    bars = plt.bar(results_df['name'], results_df['improvement'] * 100, color=colors, alpha=0.7)
    
    plt.axhline(y=0, color='black', linestyle='-', linewidth=1)
    plt.ylabel('Improvement over Baseline (%)', fontsize=12)
    plt.xlabel('Experiment', fontsize=12)
    plt.title('DenseNet-121: Improvement vs Baseline', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    
    for bar, imp in zip(bars, results_df['improvement']):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + (0.02 if imp >= 0 else -0.05), 
                f'{imp*100:.2f}%', ha='center', va='bottom' if imp >= 0 else 'top', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"✅ Improvement chart saved to {save_path}")

# ==================== MAIN ====================
if __name__ == '__main__':
    print("="*60)
    print("DENSENET-121 ABLATION STUDY - TEST SET EVALUATION")
    print("="*60)
    
    # Load test data
    print("\n[1/4] Loading test data...")
    if not os.path.exists(TEST_FILE):
        print(f"⚠️ Test file not found. Creating test split...")
        csv_data_temp = pd.read_csv(CSV_PATH)
        all_images = csv_data_temp['Image Index'].unique().tolist()
        split_idx = int(0.8 * len(all_images))
        test_files = all_images[split_idx:]
        print(f"Created test split with {len(test_files)} images")
    else:
        with open(TEST_FILE, 'r') as f:
            test_files = [line.strip() for line in f.readlines()]
        print(f"✅ Loaded {len(test_files)} test images")
    
    csv_data = pd.read_csv(CSV_PATH)
    
    # Transform
    test_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    os.makedirs('test_results', exist_ok=True)
    all_results = []
    
    print("\n[2/4] Testing models on test set...")
    
    for i, model_config in enumerate(MODELS_TO_TEST):
        print(f"\n  [{i+1}/{len(MODELS_TO_TEST)}] Testing: {model_config['name']}")
        
        if not os.path.exists(model_config['path']):
            print(f"      ⚠️ Model not found: {model_config['path']}")
            continue
        
        test_dataset = NIHChestXrayTestDataset(
            test_files, csv_data, DATA_DIR, test_transform,
            filter_type=model_config['filter_type'],
            filter_params=model_config['params']
        )
        
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
        
        model = create_densenet_model(num_classes=NUM_CLASSES)
        checkpoint = torch.load(model_config['path'], map_location=DEVICE)
        
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model = model.to(DEVICE)
        criterion = nn.BCEWithLogitsLoss()
        
        results_df, mean_auc, _, _, _ = evaluate_test(model, test_loader, criterion)
        results_df.to_csv(f'test_results/results_{model_config["name"]}.csv', index=False)
        
        all_results.append({
            'name': model_config['name'],
            'filter_type': model_config['filter_type'],
            'mean_test_auc': mean_auc
        })
        
        print(f"      ✅ Mean Test AUC: {mean_auc:.4f}")
        torch.cuda.empty_cache()
    
    # Create comparison table
    print("\n[3/4] Saving results...")
    comparison_df = pd.DataFrame(all_results)
    comparison_df = comparison_df.sort_values('mean_test_auc', ascending=False)
    comparison_df.to_csv('test_results/densenet_test_comparison.csv', index=False)
    
    # Print results
    print("\n[4/4] Final Results")
    print("="*80)
    print("DENSENET-121 ABLATION STUDY - TEST SET RESULTS (25,596 images)")
    print("="*80)
    print(comparison_df.to_string(index=False))
    
    # Find baseline
    baseline_row = comparison_df[comparison_df['name'] == 'baseline']
    if len(baseline_row) > 0:
        baseline_auc = baseline_row['mean_test_auc'].values[0]
        print(f"\n📊 Baseline (No Filter): {baseline_auc:.4f}")
        
        print("\n📈 Improvements vs Baseline:")
        for _, row in comparison_df.iterrows():
            if row['name'] != 'baseline':
                imp = (row['mean_test_auc'] - baseline_auc) * 100
                sign = "+" if imp > 0 else ""
                print(f"    {row['name']:20s}: {row['mean_test_auc']:.4f} ({sign}{imp:.2f}%)")
        # Create visualizations
        print("\n🎨 Generating visualizations...")
        plot_comparison_bar(comparison_df, baseline_auc)
        plot_improvement_chart(comparison_df, baseline_auc)
    
    print("\n" + "="*60)
    print("✅ DENSENET-121 ABLATION TESTING COMPLETE!")
    print("="*60)
    print("\n📁 Output files:")
    print("   - test_results/densenet_test_comparison.csv")
    print("   - test_results/densenet_comparison.png")
    print("   - test_results/densenet_improvement.png")
    print("   - test_results/results_*.csv (for each model)") 