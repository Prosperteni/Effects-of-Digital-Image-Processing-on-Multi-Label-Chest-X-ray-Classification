"""
ResNet-50 Ablation Study - Testing Multiple Filter Parameters
Same filter configurations as ConvFormer and DenseNet for fair comparison
"""

import os
import cv2
import json
import pandas as pd
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.metrics import roc_auc_score
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

BATCH_SIZE = 256                    # ResNet-50 can use large batches
EPOCHS = 100
LEARNING_RATE = 1e-4
IMG_SIZE = 224
NUM_CLASSES = 14
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PATHOLOGIES = [
    'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 
    'Mass', 'Nodule', 'Pneumonia', 'Pneumothorax', 
    'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 
    'Pleural_Thickening', 'Hernia'
]

# ==================== ABLATION CONFIGURATIONS ====================
ABLATION_CONFIGS = [
    # Baseline (no filter)
    {'filter_type': 'none', 'params': {}, 'name': 'baseline'},
    
    # CLAHE with different clip limits
    {'filter_type': 'clahe', 'params': {'clip_limit': 1.0, 'tile_grid': 8}, 'name': 'clahe_clip1.0'},
    {'filter_type': 'clahe', 'params': {'clip_limit': 2.0, 'tile_grid': 8}, 'name': 'clahe_clip2.0'},
    {'filter_type': 'clahe', 'params': {'clip_limit': 3.0, 'tile_grid': 8}, 'name': 'clahe_clip3.0'},
    {'filter_type': 'clahe', 'params': {'clip_limit': 4.0, 'tile_grid': 8}, 'name': 'clahe_clip4.0'},
    {'filter_type': 'clahe', 'params': {'clip_limit': 5.0, 'tile_grid': 8}, 'name': 'clahe_clip5.0'},
    
    # Gaussian with different kernel sizes
    {'filter_type': 'gaussian', 'params': {'kernel': 3, 'sigma': 1.0}, 'name': 'gaussian_k3'},
    {'filter_type': 'gaussian', 'params': {'kernel': 5, 'sigma': 1.0}, 'name': 'gaussian_k5'},
    {'filter_type': 'gaussian', 'params': {'kernel': 7, 'sigma': 1.0}, 'name': 'gaussian_k7'},
    {'filter_type': 'gaussian', 'params': {'kernel': 9, 'sigma': 1.0}, 'name': 'gaussian_k9'},
    
    # Bilateral with different diameters
    {'filter_type': 'bilateral', 'params': {'d': 5, 'sigma_color': 75, 'sigma_space': 75}, 'name': 'bilateral_d5'},
    {'filter_type': 'bilateral', 'params': {'d': 7, 'sigma_color': 75, 'sigma_space': 75}, 'name': 'bilateral_d7'},
    {'filter_type': 'bilateral', 'params': {'d': 9, 'sigma_color': 75, 'sigma_space': 75}, 'name': 'bilateral_d9'},
    {'filter_type': 'bilateral', 'params': {'d': 11, 'sigma_color': 75, 'sigma_space': 75}, 'name': 'bilateral_d11'},
    
    # Histogram Equalization
    {'filter_type': 'hist_eq', 'params': {}, 'name': 'hist_eq'},
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

# ==================== EARLY STOPPING ====================
class EarlyStopping:
    def __init__(self, patience=10, delta=0, metric="loss", mode="min"):
        self.patience = patience
        self.delta = delta
        self.metric = metric
        self.mode = mode
        if self.mode == "min":
            self.best_score = float("inf")
            self.best_model_wts = None
            self.counter = 0
        elif self.mode == "max":
            self.best_score = -float("inf")
            self.best_model_wts = None
            self.counter = 0
        
    def step(self, metric_value, model):
        if self.metric == "loss":
            score = metric_value
        elif self.metric == "auc":
            score = metric_value
        else:
            raise ValueError("Metric should be either 'loss' or 'auc'.")
        
        if (self.mode == "min" and score < self.best_score - self.delta) or \
           (self.mode == "max" and score > self.best_score + self.delta):
            self.best_score = score
            self.best_model_wts = model.state_dict().copy()
            self.counter = 0
        else:
            self.counter += 1
            print(f"  EarlyStopping counter: {self.counter}/{self.patience}")
        
        if self.counter >= self.patience:
            return True
        return False

# ==================== DATASET CLASS ====================
class NIHChestXrayFilteredDataset(Dataset):
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
        return image, torch.from_numpy(labels)

# ==================== RESNET-50 MODEL ====================
def create_resnet50_model(num_classes=14):
    model = models.resnet50(pretrained=True)
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(num_features, num_classes)
    )
    return model

# ==================== TRAINING FUNCTIONS ====================
def train_epoch(model, loader, optimizer, criterion, scaler):
    model.train()
    total_loss = 0
    for images, labels in tqdm(loader, desc="Training"):
        images = images.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)
        with torch.cuda.amp.autocast():
            outputs = model(images)
            loss = criterion(outputs, labels)
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()
    return total_loss / len(loader)

def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Evaluating"):
            images = images.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)
            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
            total_loss += loss.item()
            all_preds.append(torch.sigmoid(outputs).cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    
    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    auc_scores = []
    for i in range(NUM_CLASSES):
        try:
            auc = roc_auc_score(all_labels[:, i], all_preds[:, i])
            auc_scores.append(auc)
        except:
            auc_scores.append(0.0)
    return total_loss / len(loader), np.mean(auc_scores), auc_scores

# ==================== RUN EXPERIMENT ====================
def run_experiment(config):
    print("\n" + "="*60)
    print(f"EXPERIMENT: {config['name'].upper()}")
    print(f"Filter: {config['filter_type']}")
    print(f"Parameters: {config['params']}")
    print("="*60)
    
    # Load data
    csv_data = pd.read_csv(CSV_PATH)
    with open(TRAIN_VAL_FILE, 'r') as f:
        train_val_files = [line.strip() for line in f.readlines()]
    split_idx = int(0.9 * len(train_val_files))
    train_files = train_val_files[:split_idx]
    val_files = train_val_files[split_idx:]
    
    print(f"Train: {len(train_files)} images, Val: {len(val_files)} images")
    
    # Transforms
    train_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    val_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Datasets with filter
    train_dataset = NIHChestXrayFilteredDataset(
        train_files, csv_data, DATA_DIR, train_transform,
        filter_type=config['filter_type'], filter_params=config['params']
    )
    val_dataset = NIHChestXrayFilteredDataset(
        val_files, csv_data, DATA_DIR, val_transform,
        filter_type=config['filter_type'], filter_params=config['params']
    )
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    
    # Model
    model = create_resnet50_model(num_classes=NUM_CLASSES)
    model = model.to(DEVICE)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    
    # Training components
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)
    scaler = torch.cuda.amp.GradScaler()
    early_stopping = EarlyStopping(patience=10, metric="loss", mode="min")
    
    best_val_auc = 0.0
    best_epoch = 0
    best_auc_scores = None
    
    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        train_loss = train_epoch(model, train_loader, optimizer, criterion, scaler)
        val_loss, val_auc, val_auc_scores = evaluate(model, val_loader, criterion)
        print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val AUC: {val_auc:.4f}")
        
        if early_stopping.step(val_loss, model):
            print(f"Early stopping triggered at epoch {epoch+1}")
            break
        
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_epoch = epoch
            best_auc_scores = val_auc_scores.copy()
            torch.save(model.state_dict(), f'models/best_{config["name"]}.pth')
            print(f"✓ New best model! AUC: {val_auc:.4f}")
        
        scheduler.step(val_loss)
    
    # Save results
    result = {
        'experiment_name': config['name'],
        'filter_type': config['filter_type'],
        'parameters': config['params'],
        'best_val_auc': float(best_val_auc),
        'best_epoch': best_epoch,
        'per_class_auc': [float(x) for x in best_auc_scores] if best_auc_scores else [],
        'diseases': PATHOLOGIES
    }
    
    os.makedirs('results', exist_ok=True)
    with open(f'results/result_{config["name"]}.json', 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"\n✅ Experiment {config['name']} complete! Best AUC: {best_val_auc:.4f}")
    return result

# ==================== MAIN ====================
if __name__ == '__main__':
    print("="*60)
    print("RESNET-50 ABLATION STUDY")
    print(f"Total experiments to run: {len(ABLATION_CONFIGS)}")
    print("="*60)
    
    os.makedirs('models', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    all_results = []
    
    for i, config in enumerate(ABLATION_CONFIGS):
        print(f"\n{'='*60}")
        print(f"Running experiment {i+1}/{len(ABLATION_CONFIGS)}: {config['name']}")
        print(f"{'='*60}")
        
        result = run_experiment(config)
        all_results.append(result)
    
    # Save summary
    summary_df = pd.DataFrame([{
        'experiment': r['experiment_name'],
        'filter': r['filter_type'],
        'best_val_auc': r['best_val_auc'],
        'best_epoch': r['best_epoch']
    } for r in all_results])
    
    summary_df = summary_df.sort_values('best_val_auc', ascending=False)
    summary_df.to_csv('results/resnet50_ablation_summary.csv', index=False)
    
    print("\n" + "="*60)
    print("RESNET-50 ABLATION STUDY COMPLETE!")
    print("="*60)
    print("\nResults Summary (sorted by AUC):")
    print(summary_df.to_string(index=False))
    print("\n✅ All results saved to 'results/' folder")