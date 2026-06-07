"""
ConvFormer (Hybrid CNN-Transformer) Training for NIH Chest X-ray
Based on the 2025 benchmark study: Achieved 0.841 mean AUROC on NIH ChestX-ray14
Highest performer among 14 architectures including Swin, DenseNet, and VMamba
"""

import os
# Set mirror for China (if downloading pretrained weights)
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import pandas as pd
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import timm
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

# ConvFormer Hyperparameters
BATCH_SIZE = 64                     # Good balance for ConvFormer (85M params)
EPOCHS = 100
LEARNING_RATE = 1e-4                # Standard fine-tuning LR
IMG_SIZE = 224
NUM_CLASSES = 14
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ConvFormer model variants (from timm)
# - 'convformer_s18': 85M params, 83.6% ImageNet
# - 'convformer_s36': 124M params, 84.1% ImageNet
# - 'convformer_b36': 148M params, 84.6% ImageNet (best but slower)
MODEL_NAME = "convformer_s18"  # Recommended balance of speed/accuracy

PATHOLOGIES = [
    'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 
    'Mass', 'Nodule', 'Pneumonia', 'Pneumothorax', 
    'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 
    'Pleural_Thickening', 'Hernia'
]

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
class NIHChestXrayDataset(Dataset):
    def __init__(self, img_list, csv_data, img_dir, transform=None):
        self.img_list = img_list
        self.img_dir = img_dir
        self.transform = transform
        self.labels_dict = {}
        for _, row in csv_data.iterrows():
            self.labels_dict[row['Image Index']] = row
        
    def __len__(self):
        return len(self.img_list)
    
    def __getitem__(self, idx):
        img_name = self.img_list[idx]
        img_path = os.path.join(self.img_dir, img_name).replace('\\', '/')
        
        image = Image.open(img_path).convert('RGB')
        
        row = self.labels_dict.get(img_name)
        if row is None:
            labels = np.zeros(NUM_CLASSES, dtype=np.float32)
        else:
            finding_labels = row['Finding Labels'].split('|')
            labels = np.array([1.0 if p in finding_labels else 0.0 for p in PATHOLOGIES], dtype=np.float32)
        
        if self.transform:
            image = self.transform(image)
        
        return image, torch.from_numpy(labels)

# ==================== CONVFORMER MODEL ====================
def create_convformer_model(model_name, num_classes=14):
    """Create ConvFormer model using timm"""
    print(f"Loading ConvFormer model: {model_name}")
    
    model = timm.create_model(
        model_name,
        pretrained=True,
        num_classes=num_classes
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

# ==================== MAIN TRAINING ====================
if __name__ == '__main__':
    print("="*60)
    print("CONVFORMER (Hybrid CNN-Transformer) TRAINING ON NIH CHEST X-RAY")
    print("="*60)
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Model: {MODEL_NAME}")
    print(f"Device: {DEVICE}")
    print("="*60)
    
    print("\n[1/5] Loading data...")
    csv_data = pd.read_csv(CSV_PATH)

    with open(TRAIN_VAL_FILE, 'r') as f:
        train_val_files = [line.strip() for line in f.readlines()]

    split_idx = int(0.9 * len(train_val_files))
    train_files = train_val_files[:split_idx]
    val_files = train_val_files[split_idx:]

    print(f"  Train: {len(train_files)} images")
    print(f"  Validation: {len(val_files)} images")

    print("\n[2/5] Setting up transforms...")
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

    train_dataset = NIHChestXrayDataset(train_files, csv_data, DATA_DIR, train_transform)
    val_dataset = NIHChestXrayDataset(val_files, csv_data, DATA_DIR, val_transform)

    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )

    print("\n[3/5] Building ConvFormer model...")
    model = create_convformer_model(MODEL_NAME, num_classes=NUM_CLASSES)
    model = model.to(DEVICE)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {total_params/1e6:.2f}M")

    print("\n[4/5] Setting up training...")
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2, verbose=True
    )
    scaler = torch.cuda.amp.GradScaler()
    early_stopping = EarlyStopping(patience=10, metric="loss", mode="min")

    print(f"\n[5/5] Starting training...")
    print("="*60)
    
    best_val_auc = 0.0

    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        print("-" * 40)
        
        train_loss = train_epoch(model, train_loader, optimizer, criterion, scaler)
        val_loss, val_auc, val_auc_scores = evaluate(model, val_loader, criterion)
        
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f}")
        print(f"Val Mean AUC: {val_auc:.4f}")
        
        if (epoch + 1) % 10 == 0:
            print("\nPer-class Validation AUC:")
            for i, pathology in enumerate(PATHOLOGIES):
                print(f"  {pathology:20s}: {val_auc_scores[i]:.4f}")
        
        if early_stopping.step(val_loss, model):
            print("\n⚠️ Early stopping triggered.")
            model.load_state_dict(early_stopping.best_model_wts)
            break
        
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_auc': val_auc,
                'val_auc_scores': val_auc_scores
            }, 'best_convformer_nih.pth')
            print(f"✓ Saved best model with AUC: {val_auc:.4f}")
        
        scheduler.step(val_loss)

    print("\n" + "="*60)
    print(f"✅ Training complete! Best AUC: {best_val_auc:.4f}")
    print(f"Model saved to: best_convformer_nih.pth")
    print("="*60)