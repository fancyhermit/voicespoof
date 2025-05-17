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
import sys

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

# Dataset class
class InTheWildDataset(Dataset):
    def __init__(self, metadata_file, audio_dir, sample_rate=16000, target_length=64600, is_train=True):
        self.df = pd.read_csv(metadata_file)
        self.audio_dir = audio_dir
        self.sample_rate = sample_rate
        self.target_length = target_length
        self.label_map = {"spoof": 0, "bona-fide": 1}
        self.is_train = is_train

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
        print(self.df['label'].value_counts())
        print(f"Average file duration: {self._get_avg_duration():.2f} seconds")

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
            waveform = torch.tensor(waveform, dtype=torch.float32)
            
            # Pad/trim
            if waveform.shape[0] > self.target_length:
                waveform = waveform[:self.target_length]
            else:
                pad_len = self.target_length - waveform.shape[0]
                waveform = F.pad(waveform, (0, pad_len))

            # Normalize waveform
            waveform = (waveform - waveform.mean()) / (waveform.std() + 1e-8)

            label = self.label_map[row["label"]]
            return waveform, torch.tensor(label, dtype=torch.long)
        except Exception as e:
            print(f"Error loading {audio_path}: {str(e)}")
            # Return a zero waveform as fallback
            return torch.zeros(self.target_length, dtype=torch.float32), torch.tensor(0, dtype=torch.long)


# Load model
def load_model(model_path):
    print(f"\nLoading model from: {model_path}")
    model = RawNet(d_args, device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    print(f"Model device: {next(model.parameters()).device}")
    return model
    

# Train loop
def train(model, train_loader, val_loader, epochs=5, lr=1e-4):
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)
    
    print(f"\nStarting training on device: {next(model.parameters()).device}")
    print(f"Batch size: {train_loader.batch_size}")
    print(f"Number of batches: {len(train_loader)}")
    
    best_val_acc = 0
    for epoch in range(epochs):
        # Training phase
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        # Use tqdm with dynamic_ncols=True and leave=True
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]", 
                          dynamic_ncols=True, leave=True, file=sys.stdout)
        
        for x, y in progress_bar:
            try:
                x = x.to(device)
                y = y.to(device)

                outputs = model(x)
                # Ensure outputs and labels have correct shapes
                if outputs.shape[1] != 2:  # If model outputs single value
                    outputs = outputs.unsqueeze(1)
                    outputs = torch.cat([outputs, 1-outputs], dim=1)
                loss = criterion(outputs, y)

                optimizer.zero_grad()
                loss.backward()
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                total_loss += loss.item()
                
                # Calculate accuracy
                _, predicted = outputs.max(1)
                total += y.size(0)
                correct += predicted.eq(y).sum().item()
                
                # Update progress bar
                progress_bar.set_postfix({
                    'loss': f'{total_loss/total:.4f}',
                    'acc': f'{100.*correct/total:.2f}%'
                })
            except Exception as e:
                print(f"\nError in training batch: {str(e)}")
                continue

        train_loss = total_loss / len(train_loader)
        train_acc = 100. * correct / total

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        try:
            with torch.no_grad():
                # Use tqdm with dynamic_ncols=True and leave=True
                val_progress = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]",
                                  dynamic_ncols=True, leave=True, file=sys.stdout)
                
                for x, y in val_progress:
                    try:
                        x = x.to(device)
                        y = y.to(device)
                        
                        outputs = model(x)
                        if outputs.shape[1] != 2:
                            outputs = outputs.unsqueeze(1)
                            outputs = torch.cat([outputs, 1-outputs], dim=1)
                        loss = criterion(outputs, y)
                        
                        val_loss += loss.item()
                        _, predicted = outputs.max(1)
                        val_total += y.size(0)
                        val_correct += predicted.eq(y).sum().item()
                        
                        val_progress.set_postfix({
                            'loss': f'{val_loss/val_total:.4f}',
                            'acc': f'{100.*val_correct/val_total:.2f}%'
                        })
                    except Exception as e:
                        print(f"\nError in validation batch: {str(e)}")
                        continue
        except Exception as e:
            print(f"\nError during validation: {str(e)}")
            val_loss = float('inf')
            val_acc = 0.0
        else:
            val_loss = val_loss / len(val_loader)
            val_acc = 100. * val_correct / val_total
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_model_in_the_wild.pth")
            print(f"\nSaved best model with validation accuracy: {val_acc:.2f}%")
        
        print(f"\nEpoch {epoch+1}/{epochs}")
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")

# Run everything
if __name__ == "__main__":
    metadata_file = "C:\\Users\\Sez\\Desktop\\tryv\\meta.csv"
    audio_dir = "C:/Users/Sez/Desktop/tryv/release_in_the_wild/release_in_the_wild"
    model_path = "C:/Users/Sez/Desktop/tryv/2021/LA/Baseline-RawNet2/pretrained/pre_trained_DF_RawNet2/pre_trained_DF_RawNet2.pth"

    # Create datasets
    full_dataset = InTheWildDataset(metadata_file, audio_dir)
    
    # Split into train and validation
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
    
    # Create dataloaders with modified settings
    train_loader = DataLoader(
        train_dataset, 
        batch_size=16,
        shuffle=True, 
        num_workers=0,  # Disabled multiprocessing
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=16,
        shuffle=False,
        num_workers=0,  # Disabled multiprocessing
        pin_memory=True
    )

    model = load_model(model_path)
    train(model, train_loader, val_loader, epochs=20)