import torch
import torch.nn.functional as F
from model import RawNet
import os
import sys
import librosa
import yaml

# Load configuration
with open("model_config_RawNet.yaml", "r") as file:
    config = yaml.safe_load(file)
d_args = config["model"]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load audio file
def load_audio(file_path, sample_rate=16000):
    wav, sr = librosa.load(file_path, sr=sample_rate)  # Always resampled to sample_rate
    waveform = torch.tensor(wav, dtype=torch.float32).unsqueeze(0)  # shape: [1, length]
    return waveform
def preprocess_waveform(waveform, target_length=64600):
    waveform = waveform.squeeze(0)  # shape: [length]
    if waveform.shape[0] > target_length:
        waveform = waveform[:target_length]
    else:
        padding = target_length - waveform.shape[0]
        waveform = F.pad(waveform, (0, padding))
    return waveform


# Load the model
def load_model(model_path, device):
    model = RawNet(d_args, device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    return model

# Classify audio as REAL or FAKE
def classify_audio(model, waveform, device):
    waveform = preprocess_waveform(waveform).to(device)  # Ensure fixed length
    with torch.no_grad():
        output = model(waveform.unsqueeze(0))  # Shape: [1, 64600]
        probs = F.softmax(output, dim=1)
        predicted_class = torch.argmax(probs, dim=1).item()
        score = probs[0][predicted_class].item()
        label = "REAL" if predicted_class == 1 else "FAKE"
    return score, label


if __name__ == "__main__":
    wav_path = sys.argv[1]  # Pass the path to your WAV file
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model_path = "C:/Users/Sez/Desktop/tryv/2021/LA/Baseline-RawNet2/pretrained/pre_trained_DF_RawNet2/pre_trained_DF_RawNet2.pth"
    model = load_model(model_path, device)
    
    waveform = load_audio(wav_path)
    score, label = classify_audio(model, waveform, device)

    print(f"Prediction Score: {score:.4f} | Label: {label}")
