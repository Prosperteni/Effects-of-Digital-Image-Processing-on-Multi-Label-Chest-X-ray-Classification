"""
ConvFormer (Hybrid CNN-Transformer) Testing for NIH Chest X-ray
Test script for trained ConvFormer model
"""

import os
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
TEST_FILE = os.path.join(BASE_DIR, "dataset", "test_list_NIH.txt")
BATCH_SIZE = 64                      # Match training batch size
IMG_SIZE = 224
NUM_CLASSES = 14
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "best_convformer_nih.pth"  # Path to trained ConvFormer model
MODEL_NAME = "convformer_s18"            # ConvFormer model

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
def create_convformer_model(model_name, num_classes=14):
    """Create ConvFormer model (same architecture as training)"""
    model = timm.create_model(
        model_name,
        pretrained=False,  # False because we're loading our trained weights
        num_classes=num_classes
    )
    return model

# ==================== EVALUATION FUNCTIONS ====================
def evaluate_test(model, loader, criterion):
    """Evaluate model on test set with optimized inference"""
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
        # AUC
        try:
            auc = roc_auc_score(all_labels[:, i], all_preds[:, i])
            per_class_auc.append(auc)
        except:
            auc = 0.0
            per_class_auc.append(0.0)
        
        # F1 Score (threshold 0.5)
        pred_binary = (all_preds[:, i] > 0.5).astype(int)
        f1 = f1_score(all_labels[:, i], pred_binary, zero_division=0)
        
        # Confusion matrix metrics
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

def plot_roc_curves(all_labels, all_preds, save_path='test_results/roc_curves_convformer.png'):
    """Plot ROC curves for all diseases"""
    from sklearn.metrics import roc_curve
    
    plt.figure(figsize=(14, 10))
    colors = plt.cm.tab20(np.linspace(0, 1, NUM_CLASSES))
    
    for i, (disease, color) in enumerate(zip(PATHOLOGIES, colors)):
        fpr, tpr, _ = roc_curve(all_labels[:, i], all_preds[:, i])
        auc = roc_auc_score(all_labels[:, i], all_preds[:, i])
        plt.plot(fpr, tpr, color=color, lw=2, label=f'{disease} (AUC = {auc:.3f})')
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random (AUC = 0.5)')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curves - ConvFormer on NIH Chest X-ray', fontsize=14)
    plt.legend(loc='lower right', fontsize=8, ncol=2)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"✅ ROC curves saved to {save_path}")

def plot_confusion_matrix_heatmap(all_labels, all_preds, save_path='test_results/confusion_matrix_convformer.png'):
    """Plot confusion matrix heatmap"""
    pred_binary = (all_preds > 0.5).astype(int)
    
    cm_data = []
    for i in range(NUM_CLASSES):
        tn, fp, fn, tp = confusion_matrix(all_labels[:, i], pred_binary[:, i], labels=[0,1]).ravel()
        cm_data.append([tn, fp, fn, tp])
    
    cm_df = pd.DataFrame(cm_data, index=PATHOLOGIES, 
                         columns=['True Neg', 'False Pos', 'False Neg', 'True Pos'])
    cm_normalized = cm_df.div(cm_df.sum(axis=1), axis=0)
    
    plt.figure(figsize=(14, 10))
    sns.heatmap(cm_normalized[['True Pos', 'False Neg', 'False Pos', 'True Neg']], 
                annot=True, fmt='.2f', cmap='RdYlGn', cbar_kws={'label': 'Proportion'})
    plt.title('Normalized Confusion Matrix - Per Disease (ConvFormer)', fontsize=14)
    plt.xlabel('Predicted Class', fontsize=12)
    plt.ylabel('True Disease', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"✅ Confusion matrix saved to {save_path}")

def plot_performance_bars(results_df, save_path='test_results/performance_bars_convformer.png'):
    """Plot bar chart of AUC per disease"""
    plt.figure(figsize=(14, 6))
    
    diseases = results_df['Disease'].values
    auc_scores = results_df['AUC'].values
    
    colors = ['green' if x > 0.7 else 'orange' if x > 0.6 else 'red' for x in auc_scores]
    plt.bar(diseases, auc_scores, color=colors, alpha=0.7)
    
    plt.axhline(y=0.7, color='green', linestyle='--', linewidth=1, label='Good (AUC > 0.7)')
    plt.axhline(y=0.6, color='orange', linestyle='--', linewidth=1, label='Moderate (AUC > 0.6)')
    
    plt.xlabel('Disease', fontsize=12)
    plt.ylabel('AUC Score', fontsize=12)
    plt.title('Per-class AUC - ConvFormer on NIH Test Set', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.ylim([0, 1])
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"✅ Performance chart saved to {save_path}")

def plot_predictions_grid(all_labels, all_preds, all_img_names, num_samples=20, save_path='test_results/predictions_grid_convformer.png'):
    """Plot grid of sample predictions with confidence scores"""
    np.random.seed(42)
    sample_idx = np.random.choice(len(all_img_names), min(num_samples, len(all_img_names)), replace=False)
    
    fig, axes = plt.subplots(4, 5, figsize=(20, 16))
    axes = axes.ravel()
    
    for idx, ax in enumerate(axes):
        if idx < len(sample_idx):
            sample_id = sample_idx[idx]
            img_name = all_img_names[sample_id]
            
            img_path = os.path.join(DATA_DIR, img_name).replace('\\', '/')
            img = Image.open(img_path).convert('RGB')
            ax.imshow(img)
            
            true_labels = all_labels[sample_id]
            pred_probs = all_preds[sample_id]
            
            top3_idx = np.argsort(pred_probs)[-3:][::-1]
            top3_diseases = [PATHOLOGIES[i] for i in top3_idx if pred_probs[i] > 0.3]
            top3_probs = [pred_probs[i] for i in top3_idx if pred_probs[i] > 0.3]
            
            true_diseases = [PATHOLOGIES[i] for i in range(NUM_CLASSES) if true_labels[i] == 1]
            
            title = f"True: {', '.join(true_diseases[:2])}\n"
            if top3_diseases:
                title += f"Pred: {', '.join([f'{d}({p:.2f})' for d, p in zip(top3_diseases, top3_probs)])}"
            else:
                title += "Pred: No findings"
            
            ax.set_title(title, fontsize=9, wrap=True)
            ax.axis('off')
        else:
            ax.axis('off')
    
    plt.suptitle('Sample Predictions - ConvFormer', fontsize=16)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"✅ Predictions grid saved to {save_path}")

# ==================== MAIN TESTING ====================
if __name__ == '__main__':
    print("="*60)
    print("CONVFORMER TESTING")
    print("="*60)
    
    # Check if test file exists
    if not os.path.exists(TEST_FILE):
        print(f"\n⚠️ Test file {TEST_FILE} not found!")
        print("Creating test split from CSV...")
        csv_data_temp = pd.read_csv(CSV_PATH)
        all_images = csv_data_temp['Image Index'].unique().tolist()
        split_idx = int(0.8 * len(all_images))
        test_files = all_images[split_idx:]
        print(f"Created test split with {len(test_files)} images (20% of dataset)")
    else:
        print("\nLoading test data from test_list_NIH.txt...")
        with open(TEST_FILE, 'r') as f:
            test_files = [line.strip() for line in f.readlines()]
    
    # Load CSV
    csv_data = pd.read_csv(CSV_PATH)
    print(f"Test samples: {len(test_files)}")
    
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
    
    print(f"\nBatch size: {BATCH_SIZE}")
    print(f"Number of batches: {len(test_loader)}")
    print(f"Device: {DEVICE}")
    
    # Load model
    print(f"\nLoading ConvFormer model: {MODEL_NAME}")
    model = create_convformer_model(MODEL_NAME, num_classes=NUM_CLASSES)
    
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        epoch_info = checkpoint.get('epoch', 'unknown')
        val_auc_info = checkpoint.get('val_auc', 'unknown')
        print(f"✓ Loaded model from epoch {epoch_info}")
        if isinstance(val_auc_info, (int, float)):
            print(f"  Training validation AUC was: {val_auc_info:.4f}")
    else:
        model.load_state_dict(checkpoint)
        print("✓ Loaded model weights")
    
    model = model.to(DEVICE)
    criterion = nn.BCEWithLogitsLoss()
    
    if torch.cuda.is_available():
        print(f"\nGPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Evaluate
    print("\nRunning evaluation on test set...")
    results_df, mean_auc, all_labels, all_preds, img_names = evaluate_test(model, test_loader, criterion)
    
    # Print results
    print("\n" + "="*80)
    print("TEST RESULTS PER DISEASE")
    print("="*80)
    print(results_df.to_string(index=False))
    
    print("\n" + "="*80)
    print(f"📈 MEAN AUC ACROSS ALL DISEASES: {mean_auc:.4f}")
    print("="*80)
    
    # Save results
    results_df.to_csv('test_results_convformer.csv', index=False)
    print("\n✅ Results saved to: test_results_convformer.csv")
    
    # Save detailed predictions
    detailed_df = pd.DataFrame({'Image': img_names})
    for i, pathology in enumerate(PATHOLOGIES):
        detailed_df[f'{pathology}_true'] = all_labels[:, i]
        detailed_df[f'{pathology}_pred'] = all_preds[:, i]
    
    detailed_df.to_csv('test_predictions_convformer.csv', index=False)
    print("✅ Detailed predictions saved to: test_predictions_convformer.csv")
    
    # Create visualizations
    print("\nGenerating visualizations...")
    os.makedirs('test_results', exist_ok=True)
    
    plot_roc_curves(all_labels, all_preds)
    plot_confusion_matrix_heatmap(all_labels, all_preds)
    plot_performance_bars(results_df)
    plot_predictions_grid(all_labels, all_preds, img_names, num_samples=20)
    
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
    
    above_07 = results_df[results_df['AUC'] > 0.7]['Disease'].tolist()
    if above_07:
        print(f"\n✅ Diseases with AUC > 0.7: {', '.join(above_07)}")
    
    print("\n✅ Testing complete!")
    print("📁 Output files:")
    print("   - test_results_convformer.csv")
    print("   - test_predictions_convformer.csv")
    print("   - test_results/roc_curves_convformer.png")
    print("   - test_results/confusion_matrix_convformer.png")
    print("   - test_results/performance_bars_convformer.png")
    print("   - test_results/predictions_grid_convformer.png")