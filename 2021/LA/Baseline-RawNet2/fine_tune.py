import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import librosa
import yaml
from model import RawNet
from tqdm import tqdm
import multiprocessing
import numpy as np
from sklearn.model_selection import train_test_split
from audiomentations import Compose, AddGaussianNoise, PitchShift, TimeStretch
from sklearn.metrics import roc_curve
import sys
import logging

# Set up logging
logging.basicConfig(filename='finetune.log', level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Print CUDA information
print("\n=== CUDA Information ===")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    print(f"CUDA version: {torch.version.cuda}")
    print(f"Current device: {torch.cuda.current_device()}")
    print(f"Device count: {torch.cuda.device_count()}")
print("=======================\n")

# Set multiprocessing start method
if __name__ == '__main__':
    multiprocessing.set_start_method('spawn', force=True)

# Load config
with open("model_config_RawNet.yaml", "r") as file:
    config = yaml.safe_load(file)
d_args = config["model"]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Data augmentation
augment = Compose([
    AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5),
    PitchShift(min_semitones=-2, max_semitones=2, p=0.5),
    TimeStretch(min_rate=0.8, max_rate=1.2, p=0.5)
])

# Dataset class
class InTheWildDataset(Dataset):
    def __init__(self, metadata_file, audio_dir, sample_rate=16000, target_length=64600, is_train=True):
        self.df = pd.read_csv(metadata_file)
        self.audio_dir = audio_dir
        self.sample_rate = sample_rate
        self.target_length = target_length
        self.label_map = {"spoof": 0, "bona-fide": 1}
        self.is_train = is_train
        self.debug_count = 0  # Counter for debug prints

        # Filter out missing files
        existing_files = []
        for i, row in self.df.iterrows():
            full_path = os.path.normpath(os.path.join(audio_dir, row["file"]))
            if os.path.exists(full_path):
                existing_files.append(row)
            else:
                print(f"[SKIP] Missing file: {row['file']}")
        self.df = pd.DataFrame(existing_files)
        
        # Print dataset statistics
        print(f"\nDataset Statistics:")
        print(f"Total files: {len(self.df)}")
        print("Label distribution:")
        label_counts = self.df['label'].value_counts()
        print(label_counts)

    def _get_avg_duration(self):
        durations = []
        for file in self.df['file']:
            try:
                audio_path = os.path.join(self.audio_dir, file)
                duration = librosa.get_duration(path=audio_path)
                durations.append(duration)
            except:
                continue
        return np.mean(durations) if durations else 0

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        audio_path = os.path.join(self.audio_dir, row["file"])
        try:
            waveform, _ = librosa.load(audio_path, sr=self.sample_rate)
            waveform = np.array(waveform, dtype=np.float32)
            
            # Apply augmentation only during training
            if self.is_train:
                waveform = augment(samples=waveform, sample_rate=self.sample_rate)
                waveform = waveform.astype(np.float32)  # Ensure float32 after augmentation
            
            waveform = torch.tensor(waveform).float()  # Enforce float32
            
            # Pad/trim
            if waveform.shape[0] > self.target_length:
                waveform = waveform[:self.target_length]
            else:
                pad_len = self.target_length - waveform.shape[0]
                waveform = F.pad(waveform, (0, pad_len))

            # Normalize waveform
            waveform = (waveform - waveform.mean()) / (waveform.std() + 1e-8)

            label = self.label_map[row["label"]]
            label_tensor = torch.tensor(label, dtype=torch.long)

            # Debug prints for first 10 samples
            if self.debug_count < 10:
                print(f"Sample {self.debug_count + 1}:")
                print(f"Waveform dtype after librosa.load: {waveform.dtype}")
                print(f"Waveform dtype after np.array: {np.array(waveform.cpu()).dtype}")
                print(f"Waveform dtype after augmentation: {waveform.dtype}")
                print(f"Waveform tensor dtype: {waveform.dtype}")
                print(f"Waveform dtype after normalization: {waveform.dtype}")
                print(f"Label tensor dtype: {label_tensor.dtype}")
                self.debug_count += 1

            return waveform, label_tensor, audio_path
        except Exception as e:
            print(f"Error loading {audio_path}: {str(e)}")
            logging.error(f"Error loading {audio_path}: {str(e)}")
            return torch.zeros(self.target_length, dtype=torch.float32), torch.tensor(0, dtype=torch.long), audio_path

# Compute EER
def compute_eer(labels, scores):
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    fnr = 1 - tpr
    eer_threshold = thresholds[np.argmin(np.abs(fnr - fpr))]
    eer = fpr[np.argmin(np.abs(fnr - fpr))]
    return eer

# Load model
def load_model(model_path):
    print(f"\nLoading model from: {model_path}")
    model = RawNet(d_args, device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    # Freeze early layers
    for name, param in model.named_parameters():
        if "Sinc_conv" in name or "block" in name:
            param.requires_grad = False
    print(f"Model device: {next(model.parameters()).device}")
    return model

# Train loop
def train(model, train_loader, val_loader, class_weights, epochs=10, lr=1e-5):
    model.train()
    class_weights = class_weights.float()  # Ensure float32
    print(f"Class weights dtype: {class_weights.dtype}")
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    
    print(f"\nStarting training on device: {next(model.parameters()).device}")
    print(f"Batch size: {train_loader.batch_size}")
    print(f"Number of batches: {len(train_loader)}")
    
    best_val_eer = float('inf')
    patience = 3
    epochs_no_improve = 0
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]", 
                          dynamic_ncols=True, leave=False)
        
        for x, y, _ in progress_bar:
            try:
                x, y = x.to(device), y.to(device)
                outputs = model(x)
                loss = criterion(outputs, y)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += y.size(0)
                correct += predicted.eq(y).sum().item()
                
                progress_bar.set_postfix({
                    'loss': f'{total_loss/total:.4f}',
                    'acc': f'{100.*correct/total:.2f}%'
                })
            except Exception as e:
                print(f"\nError in training batch: {str(e)}")
                logging.error(f"Error in training batch: {str(e)}")
                continue

        train_loss = total_loss / len(train_loader)
        train_acc = 100. * correct / total

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        val_labels = []
        val_scores = []
        misclassified_files = []
        
        with torch.no_grad():
            val_progress = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]",
                              dynamic_ncols=True, leave=False)
            
            for x, y, paths in val_progress:
                try:
                    x, y = x.to(device), y.to(device)
                    outputs = model(x)
                    loss = criterion(outputs, y)
                    
                    val_loss += loss.item()
                    _, predicted = outputs.max(1)
                    val_total += y.size(0)
                    val_correct += predicted.eq(y).sum().item()
                    
                    # Collect scores and labels for EER
                    scores = F.softmax(outputs, dim=1)[:, 1].cpu().numpy()
                    val_labels.extend(y.cpu().numpy())
                    val_scores.extend(scores)
                    
                    # Log misclassified samples
                    for i, (pred, true, path) in enumerate(zip(predicted, y, paths)):
                        if pred != true:
                            misclassified_files.append(f"{path}: Predicted {pred.item()}, True {true.item()}")
                    
                    val_progress.set_postfix({
                        'loss': f'{val_loss/val_total:.4f}',
                        'acc': f'{100.*val_correct/val_total:.2f}%'
                    })
                except Exception as e:
                    print(f"\nError in validation batch: {str(e)}")
                    logging.error(f"Error in validation batch: {str(e)}")
                    continue
        
        val_loss = val_loss / len(val_loader)
        val_acc = 100. * val_correct / val_total
        val_eer = compute_eer(val_labels, val_scores)
        
        # Log misclassified files
        if misclassified_files:
            logging.info(f"Epoch {epoch+1} Misclassified Files:\n" + "\n".join(misclassified_files))
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Save best model based on EER
        if val_eer < best_val_eer:
            best_val_eer = val_eer
            torch.save(model.state_dict(), "best_model_in_the_wild.pth")
            print(f"\nSaved best model with validation EER: {val_eer:.4f}")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
        
        print(f"\nEpoch {epoch+1}/{epochs}")
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% | Val EER: {val_eer:.4f}")
        
        # Early stopping
        if epochs_no_improve >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

# Run everything
if __name__ == "__main__":
    metadata_file = "C:\\Users\\Sez\\Desktop\\tryv\\meta.csv"
    audio_dir = "C:/Users/Sez/Desktop/tryv/release_in_the_wild/release_in_the_wild"
    model_path = "C:/Users/Sez/Desktop/tryv/2021/LA/Baseline-RawNet2/pretrained/pre_trained_DF_RawNet2/pre_trained_DF_RawNet2.pth"

    # Create datasets
    full_dataset = InTheWildDataset(metadata_file, audio_dir, is_train=True)
    val_dataset = InTheWildDataset(metadata_file, audio_dir, is_train=False)
    
    # Calculate class weights
    label_counts = full_dataset.df['label'].value_counts()
    class_weights = torch.tensor([1.0, len(full_dataset.df) / label_counts['bona-fide']], dtype=torch.float32).to(device)
    print(f"Class weights: {class_weights}")
    
    # Split into train and validation
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, _ = torch.utils.data.random_split(full_dataset, [train_size, val_size])
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=16,
        shuffle=True, 
        num_workers=0,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=16,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    model = load_model(model_path)
    train(model, train_loader, val_loader, class_weights, epochs=10)