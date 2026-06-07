"""
ResNet-101 Testing on NIH Chest X-ray Test Set
"""

import os
import pandas as pd
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
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
BATCH_SIZE = 128                     # Match training batch size
IMG_SIZE = 224
NUM_CLASSES = 14
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "best_resnet101_nih.pth"  # Path to trained model

PATHOLOGIES = [
    'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 
    'Mass', 'Nodule', 'Pneumonia', 'Pneumothorax', 
    'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 
    'Pleural_Thickening', 'Hernia'
]

# ==================== DATASET CLASS ====================
class NIHChestXrayTestDataset(Dataset):
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
        
        return image, torch.from_numpy(labels), img_name

# ==================== MODEL CREATION ====================
def create_resnet101_model(num_classes=14):
    """Create ResNet-101 model (same architecture as training)"""
    model = models.resnet101(pretrained=False)  # False because we load trained weights
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
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
    
    auc_scores = []
    for i in range(NUM_CLASSES):
        try:
            auc = roc_auc_score(all_labels[:, i], all_preds[:, i])
            auc_scores.append(auc)
        except:
            auc_scores.append(0.0)
    
    return total_loss / len(loader), np.mean(auc_scores), auc_scores, all_preds, all_labels, all_img_names

# ==================== MAIN TESTING ====================
if __name__ == '__main__':
    print("="*60)
    print("RESNET-101 TESTING ON NIH CHEST X-RAY")
    print("="*60)
    
    # Load test data
    print("\n[1/4] Loading test data...")
    if not os.path.exists(TEST_FILE):
        print(f"⚠️ Test file {TEST_FILE} not found!")
        print("Creating test split from CSV...")
        csv_data_temp = pd.read_csv(CSV_PATH)
        all_images = csv_data_temp['Image Index'].unique().tolist()
        split_idx = int(0.8 * len(all_images))
        test_files = all_images[split_idx:]
        print(f"Created test split with {len(test_files)} images (20% of dataset)")
    else:
        with open(TEST_FILE, 'r') as f:
            test_files = [line.strip() for line in f.readlines()]
        print(f"✅ Loaded {len(test_files)} test images from test_list_NIH.txt")
    
    # Load CSV
    csv_data = pd.read_csv(CSV_PATH)
    
    # Transforms
    test_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Dataset & DataLoader
    test_dataset = NIHChestXrayTestDataset(test_files, csv_data, DATA_DIR, test_transform)
    test_loader = DataLoader(
        test_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )
    
    print(f"\n[2/4] Test setup:")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Number of batches: {len(test_loader)}")
    print(f"  Device: {DEVICE}")
    
    # Load model
    print("\n[3/4] Loading ResNet-101 model...")
    model = create_resnet101_model(num_classes=NUM_CLASSES)
    
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    
    # Handle checkpoint format
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        epoch_info = checkpoint.get('epoch', 'unknown')
        val_auc_info = checkpoint.get('val_auc', 'unknown')
        print(f"  ✓ Loaded model from epoch {epoch_info}")
        if isinstance(val_auc_info, (int, float)):
            print(f"    Training validation AUC was: {val_auc_info:.4f}")
    else:
        model.load_state_dict(checkpoint)
        print("  ✓ Loaded model weights")
    
    model = model.to(DEVICE)
    criterion = nn.BCEWithLogitsLoss()
    
    # Print GPU info
    if torch.cuda.is_available():
        print(f"\n  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Evaluate
    print("\n[4/4] Running evaluation on test set...")
    test_loss, test_auc, test_auc_scores, all_preds, all_labels, img_names = evaluate_test(
        model, test_loader, criterion
    )
    
    # Print results
    print("\n" + "="*80)
    print("TEST RESULTS PER DISEASE")
    print("="*80)
    
    # Create results dataframe
    results = []
    for i, pathology in enumerate(PATHOLOGIES):
        results.append({
            'Disease': pathology,
            'AUC': test_auc_scores[i],
            'Prevalence': all_labels[:, i].mean()
        })
    
    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False))
    
    print("\n" + "="*80)
    print(f"📈 MEAN AUC ACROSS ALL DISEASES: {test_auc:.4f}")
    print("="*80)
    
    # Save results
    results_df.to_csv('test_auc_results_resnet101.csv', index=False)
    print("\n✅ Results saved to: test_auc_results_resnet101.csv")
    
    # Save detailed predictions
    detailed_df = pd.DataFrame({
        'Image': img_names
    })
    
    for i, pathology in enumerate(PATHOLOGIES):
        detailed_df[f'{pathology}_true'] = all_labels[:, i]
        detailed_df[f'{pathology}_pred'] = all_preds[:, i]
    
    detailed_df.to_csv('test_predictions_resnet101.csv', index=False)
    print("✅ Detailed predictions saved to: test_predictions_resnet101.csv")
    
    # Print summary
    print("\n" + "="*60)
    print("PERFORMANCE SUMMARY")
    print("="*60)
    
    best_idx = results_df['AUC'].idxmax()
    best_disease = results_df.loc[best_idx, 'Disease']
    best_auc = results_df.loc[best_idx, 'AUC']
    worst_idx = results_df['AUC'].idxmin()
    worst_disease = results_df.loc[worst_idx, 'Disease']
    worst_auc = results_df.loc[worst_idx, 'AUC']
    
    print(f"🏆 Best performing disease: {best_disease} (AUC = {best_auc:.4f})")
    print(f"⚠️ Worst performing disease: {worst_disease} (AUC = {worst_auc:.4f})")
    
    # Diseases above 0.7 AUC
    above_07 = results_df[results_df['AUC'] > 0.7]['Disease'].tolist()
    if above_07:
        print(f"\n✅ Diseases with AUC > 0.7: {', '.join(above_07)}")
    
    print("\n✅ Testing complete!")
    print("📁 Output files:")
    print("   - test_auc_results_resnet101.csv")
    print("   - test_predictions_resnet101.csv")