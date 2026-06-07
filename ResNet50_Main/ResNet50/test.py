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
BATCH_SIZE = 256
IMG_SIZE = 224
NUM_CLASSES = 14
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "best_resnet50_nih.pth"

PATHOLOGIES = [
    'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 
    'Mass', 'Nodule', 'Pneumonia', 'Pneumothorax', 
    'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 
    'Pleural_Thickening', 'Hernia'
]

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

def create_resnet50_model():
    model = models.resnet50(pretrained=False)
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(num_features, NUM_CLASSES)
    )
    return model

def evaluate_test(model, loader, criterion):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    all_names = []
    
    with torch.no_grad():
        for images, labels, names in tqdm(loader, desc="Testing"):
            images = images.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)
            
            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            
            all_preds.append(torch.sigmoid(outputs).cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            all_names.extend(names)
    
    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    
    auc_scores = []
    for i in range(NUM_CLASSES):
        try:
            auc = roc_auc_score(all_labels[:, i], all_preds[:, i])
            auc_scores.append(auc)
        except:
            auc_scores.append(0.0)
    
    return total_loss / len(loader), np.mean(auc_scores), auc_scores, all_preds, all_labels, all_names

if __name__ == '__main__':
    print("="*60)
    print("RESNET-50 TESTING")
    print("="*60)
    
    print("\nLoading test data...")
    csv_data = pd.read_csv(CSV_PATH)
    
    if not os.path.exists(TEST_FILE):
        print(f"⚠️ Test file not found. Creating split...")
        all_images = csv_data['Image Index'].unique().tolist()
        split_idx = int(0.8 * len(all_images))
        test_files = all_images[split_idx:]
    else:
        with open(TEST_FILE, 'r') as f:
            test_files = [line.strip() for line in f.readlines()]
    
    print(f"Test samples: {len(test_files)}")
    
    test_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    test_dataset = NIHChestXrayTestDataset(test_files, csv_data, DATA_DIR, test_transform)
    test_loader = DataLoader(
        test_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=4,
        pin_memory=True
    )
    
    print("\nLoading model...")
    model = create_resnet50_model()
    
    # ========== FIXED LOADING CODE ==========
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    
    # Check if it's a full checkpoint dictionary
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✓ Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
        print(f"  Validation AUC was: {checkpoint.get('val_auc', 'unknown'):.4f}")
    else:
        # If it's just the state_dict
        model.load_state_dict(checkpoint)
        print("✓ Loaded model weights")
    # =======================================
    
    model = model.to(DEVICE)
    criterion = nn.BCEWithLogitsLoss()
    
    print(f"\nTesting on {DEVICE}")
    test_loss, test_auc, test_auc_scores, all_preds, all_labels, all_names = evaluate_test(model, test_loader, criterion)
    
    print("\n" + "="*50)
    print("TEST RESULTS")
    print("="*50)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Mean AUC: {test_auc:.4f}")
    
    print("\nPer-class AUCs:")
    for pathology, auc in zip(PATHOLOGIES, test_auc_scores):
        print(f"  {pathology:20s}: {auc:.4f}")
    
    # Save predictions
    detailed_df = pd.DataFrame({
        'Image': all_names
    })
    
    for i, pathology in enumerate(PATHOLOGIES):
        detailed_df[f'{pathology}_true'] = all_labels[:, i]
        detailed_df[f'{pathology}_pred'] = all_preds[:, i]
    
    detailed_df.to_csv('test_predictions_resnet50.csv', index=False)
    print("\n✓ Predictions saved to test_predictions_resnet50.csv")
    
    # Also save per-class AUC summary
    results_df = pd.DataFrame({
        'Disease': PATHOLOGIES,
        'AUC': test_auc_scores
    })
    results_df.to_csv('test_auc_results_resnet50.csv', index=False)
    print("✓ AUC results saved to test_auc_results_resnet50.csv")
    print("\nTesting complete!")