"""
Complete DIP Study Testing - Test all models on official NIH test set
Auto-creates test split if test_list_NIH.txt not found
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
from torchvision import transforms
import timm
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
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(BASE_DIR, "dataset", "images-224")
CSV_PATH = os.path.join(BASE_DIR, "dataset", "Data_Entry_2017.csv")
TRAIN_VAL_FILE = os.path.join(BASE_DIR, "dataset", "train_val_list_NIH.txt")
TEST_FILE = os.path.join(BASE_DIR, "dataset", "test_list_NIH.txt")

BATCH_SIZE = 64
IMG_SIZE = 224
NUM_CLASSES = 14
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_NAME = "convformer_s18"

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
def create_model(model_name, num_classes=14):
    model = timm.create_model(model_name, pretrained=False, num_classes=num_classes)
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
    
    return results_df, mean_auc, all_labels, all_preds, all_img_names

# ==================== ALL 15 MODELS TO TEST (FIXED) ====================
MODELS_TO_TEST = [
    # Baseline
    {'name': 'baseline', 'path': 'models/best_baseline.pth', 'filter_type': 'none', 'params': {}},
    
    # CLAHE variants (5)
    {'name': 'clahe_clip1.0', 'path': 'models/best_clahe_clip1.0.pth', 'filter_type': 'clahe', 'params': {'clip_limit': 1.0}},
    {'name': 'clahe_clip2.0', 'path': 'models/best_clahe_clip2.0.pth', 'filter_type': 'clahe', 'params': {'clip_limit': 2.0}},
    {'name': 'clahe_clip3.0', 'path': 'models/best_clahe_clip3.0.pth', 'filter_type': 'clahe', 'params': {'clip_limit': 3.0}},
    {'name': 'clahe_clip4.0', 'path': 'models/best_clahe_clip4.0.pth', 'filter_type': 'clahe', 'params': {'clip_limit': 4.0}},
    {'name': 'clahe_clip5.0', 'path': 'models/best_clahe_clip5.0.pth', 'filter_type': 'clahe', 'params': {'clip_limit': 5.0}},
    
    # Gaussian variants (4)
    {'name': 'gaussian_k3', 'path': 'models/best_gaussian_k3.pth', 'filter_type': 'gaussian', 'params': {'kernel': 3}},
    {'name': 'gaussian_k5', 'path': 'models/best_gaussian_k5.pth', 'filter_type': 'gaussian', 'params': {'kernel': 5}},
    {'name': 'gaussian_k7', 'path': 'models/best_gaussian_k7.pth', 'filter_type': 'gaussian', 'params': {'kernel': 7}},
    {'name': 'gaussian_k9', 'path': 'models/best_gaussian_k9.pth', 'filter_type': 'gaussian', 'params': {'kernel': 9}},
    
    # Bilateral variants (4)
    {'name': 'bilateral_d5', 'path': 'models/best_bilateral_d5.pth', 'filter_type': 'bilateral', 'params': {'d': 5}},
    {'name': 'bilateral_d7', 'path': 'models/best_bilateral_d7.pth', 'filter_type': 'bilateral', 'params': {'d': 7}},
    {'name': 'bilateral_d9', 'path': 'models/best_bilateral_d9.pth', 'filter_type': 'bilateral', 'params': {'d': 9}},
    {'name': 'bilateral_d11', 'path': 'models/best_bilateral_d11.pth', 'filter_type': 'bilateral', 'params': {'d': 11}},
    
    # Histogram Equalization (1)
    {'name': 'hist_eq', 'path': 'models/best_hist_eq.pth', 'filter_type': 'hist_eq', 'params': {}},
]

# ==================== MAIN ====================
if __name__ == '__main__':
    print("="*60)
    print("COMPLETE DIP STUDY: TESTING ALL 15 FILTER CONFIGURATIONS")
    print("="*60)
    
    # Load test data - AUTO CREATE IF NOT EXISTS
    print("\n[1/4] Loading test data...")
    
    if not os.path.exists(TEST_FILE):
        print(f"⚠️ Test file {TEST_FILE} not found!")
        print("Creating test split from CSV...")
        csv_data_temp = pd.read_csv(CSV_PATH)
        all_images = csv_data_temp['Image Index'].unique().tolist()
        split_idx = int(0.8 * len(all_images))
        test_files = all_images[split_idx:]
        print(f"✅ Created test split with {len(test_files)} images (20% of dataset)")
    else:
        with open(TEST_FILE, 'r') as f:
            test_files = [line.strip() for line in f.readlines()]
        print(f"✅ Loaded {len(test_files)} test images from {TEST_FILE}")
    
    csv_data = pd.read_csv(CSV_PATH)
    
    # Transform
    test_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    os.makedirs('test_results', exist_ok=True)
    all_test_results = []
    
    print("\n[2/4] Testing models on test set...")
    print(f"Total models to test: {len(MODELS_TO_TEST)}")
    
    for i, model_config in enumerate(MODELS_TO_TEST):
        print(f"\n  [{i+1}/{len(MODELS_TO_TEST)}] Testing: {model_config['name']}")
        
        if not os.path.exists(model_config['path']):
            print(f"      ⚠️ Model file not found: {model_config['path']}")
            continue
        
        test_dataset = NIHChestXrayTestDataset(
            test_files, csv_data, DATA_DIR, test_transform,
            filter_type=model_config['filter_type'],
            filter_params=model_config['params']
        )
        
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
        
        model = create_model(MODEL_NAME, num_classes=NUM_CLASSES)
        checkpoint = torch.load(model_config['path'], map_location=DEVICE)
        
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model = model.to(DEVICE)
        criterion = nn.BCEWithLogitsLoss()
        
        results_df, mean_auc, _, _, _ = evaluate_test(model, test_loader, criterion)
        results_df.to_csv(f'test_results/results_{model_config["name"]}.csv', index=False)
        
        all_test_results.append({
            'name': model_config['name'],
            'filter_type': model_config['filter_type'],
            'mean_test_auc': mean_auc
        })
        
        print(f"      ✅ Mean Test AUC: {mean_auc:.4f}")
        torch.cuda.empty_cache()
    
    # Save results
    print("\n[3/4] Saving results...")
    comparison_df = pd.DataFrame(all_test_results)
    comparison_df = comparison_df.sort_values('mean_test_auc', ascending=False)
    comparison_df.to_csv('test_results/DIP_test_comparison.csv', index=False)
    
    print("\n[4/4] Final Results")
    print("="*80)
    print("COMPLETE DIP STUDY - TEST SET RESULTS (15 CONFIGURATIONS)")
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
    
    print("\n✅ Testing complete!")
    print("📁 Results saved to: test_results/DIP_test_comparison.csv")