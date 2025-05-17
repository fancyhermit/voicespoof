import torch
import torch.nn.functional as F
from model import RawNet
import os
import sys
import librosa
import yaml
import logging
import argparse
import glob
from pathlib import Path

# Set up logging
logging.basicConfig(
    filename='infer_audio.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Load configuration
def load_config(config_path):
    try:
        with open(config_path, "r") as file:
            config = yaml.safe_load(file)
        return config["model"]
    except Exception as e:
        logging.error(f"Error loading config {config_path}: {str(e)}")
        print(f"Error: Could not load config file: {str(e)}")
        sys.exit(1)

# Load audio file
def load_audio(file_path, sample_rate=16000):
    try:
        wav, sr = librosa.load(file_path, sr=sample_rate, mono=True)
        if sr != sample_rate:
            logging.warning(f"Sample rate mismatch in {file_path}: expected {sample_rate}, got {sr}")
            print(f"Warning: {file_path} sample rate is {sr}Hz, expected 16000Hz")
        waveform = torch.tensor(wav, dtype=torch.float32).unsqueeze(0)  # shape: [1, length]
        return waveform
    except Exception as e:
        logging.error(f"Error loading audio {file_path}: {str(e)}")
        print(f"Error: Failed to load {file_path}: {str(e)}")
        return None

# Preprocess waveform
def preprocess_waveform(waveform, target_length=64600):
    try:
        waveform = waveform.squeeze(0)  # shape: [length]
        # Normalize (match training)
        waveform = (waveform - waveform.mean()) / (waveform.std() + 1e-8)
        # Pad/trim
        if waveform.shape[0] > target_length:
            waveform = waveform[:target_length]
        else:
            padding = target_length - waveform.shape[0]
            waveform = F.pad(waveform, (0, padding))
        return waveform
    except Exception as e:
        logging.error(f"Error preprocessing waveform: {str(e)}")
        print(f"Error: Failed to preprocess audio: {str(e)}")
        return None

# Load the model
def load_model(model_path, d_args, device):
    try:
        model = RawNet(d_args, device)
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint)
        model.to(device)
        model.eval()
        logging.info(f"Loaded model from {model_path} on {device}")
        print(f"✓ Model loaded successfully: {Path(model_path).name} on {device}")
        return model
    except Exception as e:
        logging.error(f"Error loading model {model_path}: {str(e)}")
        print(f"Error: Failed to load model: {str(e)}")
        sys.exit(1)

# Classify audio as REAL or FAKE
def classify_audio(model, waveform, device, threshold=0.7):
    try:
        waveform = preprocess_waveform(waveform, target_length=64600)
        if waveform is None:
            return None, None, None
        waveform = waveform.to(device)
        with torch.no_grad():
            output = model(waveform.unsqueeze(0))  # Shape: [1, 2]
            probs = F.softmax(output, dim=1)
            # Swap probabilities to fix inversion
            real_score = probs[0][0].item()  # Treat class 0 as REAL (bona-fide)
            fake_score = probs[0][1].item()  # Treat class 1 as FAKE (spoof)
            predicted_class = 1 if real_score >= threshold else 0
            label = "REAL" if predicted_class == 1 else "FAKE"
            confidence = real_score if predicted_class == 1 else fake_score
        return real_score, fake_score, label
    except Exception as e:
        logging.error(f"Error classifying audio: {str(e)}")
        print(f"Error: Failed to classify audio: {str(e)}")
        return None, None, None

# Process multiple audio files
def process_audio_files(model, input_path, device, threshold=0.7):
    results = []
    input_path = Path(input_path)
    
    # Handle directory or single file
    if input_path.is_dir():
        audio_files = glob.glob(str(input_path / "*.wav"))
    else:
        audio_files = [str(input_path)]
    
    if not audio_files:
        print(f"No WAV files found in {input_path}")
        logging.warning(f"No WAV files found in {input_path}")
        return results

    print("\n=== Audio Deepfake Detection Demo ===")
    for file_path in audio_files:
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            logging.warning(f"File not found: {file_path}")
            continue
        
        print(f"\nAnalyzing: {Path(file_path).name}")
        waveform = load_audio(file_path, sample_rate=16000)
        if waveform is None:
            print(f"✗ Failed to load audio")
            continue
        
        real_score, fake_score, label = classify_audio(model, waveform, device, threshold)
        if real_score is None:
            print(f"✗ Failed to classify audio")
            continue
        
        result = {
            "file": file_path,
            "real_score": real_score,
            "fake_score": fake_score,
            "label": label
        }
        results.append(result)
        
        # Presentation-friendly output
        confidence = real_score if label == "REAL" else fake_score
        output = (
            f"Result: This audio is {label}\n"
            f"Confidence: {confidence*100:.1f}%\n"
            f"(Real Score: {real_score:.4f}, Fake Score: {fake_score:.4f})"
        )
        print(output)
        logging.info(f"File: {file_path}\n{output}\nThreshold: {threshold}")
    
    return results

def main():
    parser = argparse.ArgumentParser(description="Audio Deepfake Detection Demo with RawNet")
    parser.add_argument("input_path", help="Path to WAV file or directory of WAV files")
    #parser.add_argument("--model_path", default="C:/Users/Sez/Desktop/tryv/2021/LA/Baseline-RawNet2/pretrained/pre_trained_DF_RawNet2/pre_trained_DF_RawNet2.pth", 
    #                   help="Path to model checkpoint")
    parser.add_argument("--model_path", default="C:/Users/Sez/Desktop/tryv/2021/LA/Baseline-RawNet2/best_model_in_the_wild.pth", 
                       help="Path to model checkpoint")
    parser.add_argument("--config_path", default="model_config_RawNet.yaml", 
                       help="Path to model config YAML")
    parser.add_argument("--threshold", type=float, default=0.7, 
                       help="Confidence threshold for REAL classification (0.0-1.0)")
    
    args = parser.parse_args()
    
    # Initialize device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load config and model
    d_args = load_config(args.config_path)
    model = load_model(args.model_path, d_args, device)
    
    # Process audio
    results = process_audio_files(model, input_path=args.input_path, device=device, threshold=args.threshold)
    
    # Presentation summary
    if results:
        print("\n=== Demo Summary ===")
        real_count = sum(1 for r in results if r["label"] == "REAL")
        fake_count = sum(1 for r in results if r["label"] == "FAKE")
        print(f"Total files analyzed: {len(results)}")
        print(f"Detected as REAL: {real_count}")
        print(f"Detected as FAKE: {fake_count}")
        print(f"Confidence threshold: {args.threshold}")
        logging.info(f"Summary: Total={len(results)}, REAL={real_count}, FAKE={fake_count}, Threshold={args.threshold}")

if __name__ == "__main__":
    main()