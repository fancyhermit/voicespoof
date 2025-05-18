import streamlit as st
import torch
import torch.nn.functional as F
from model import RawNet
import librosa
import yaml
import tempfile
import os
from pathlib import Path
#hello
# Set page config
st.set_page_config(
    page_title="Audio Deepfake Detector",
    page_icon="🎵",
    layout="centered"
)

# Load configuration
def load_config(config_path):
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config["model"]

# Load audio file
def load_audio(file_path, sample_rate=16000):
    wav, sr = librosa.load(file_path, sr=sample_rate, mono=True)
    waveform = torch.tensor(wav, dtype=torch.float32).unsqueeze(0)
    return waveform

# Preprocess waveform
def preprocess_waveform(waveform, target_length=64600):
    waveform = waveform.squeeze(0)
    waveform = (waveform - waveform.mean()) / (waveform.std() + 1e-8)
    if waveform.shape[0] > target_length:
        waveform = waveform[:target_length]
    else:
        padding = target_length - waveform.shape[0]
        waveform = F.pad(waveform, (0, padding))
    return waveform

# Load the model
@st.cache(allow_output_mutation=True)
def load_model(model_path, d_args, device):
    model = RawNet(d_args, device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    return model

# Classify audio
def classify_audio(model, waveform, device, threshold=0.7):
    waveform = preprocess_waveform(waveform, target_length=64600)
    waveform = waveform.to(device)
    with torch.no_grad():
        output = model(waveform.unsqueeze(0))
        probs = F.softmax(output, dim=1)
        real_score = probs[0][0].item()
        fake_score = probs[0][1].item()
        predicted_class = 1 if real_score >= threshold else 0
        label = "REAL" if predicted_class == 1 else "FAKE"
        confidence = real_score if predicted_class == 1 else fake_score
    return real_score, fake_score, label, confidence

def main():
    st.title("🎵 Audio Deepfake Detector")
    st.write("Upload an audio file to check if it's real or fake.")

    # Initialize device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load config and model
    config_path = "model_config_RawNet.yaml"
    model_path = "best_model_in_the_wild.pth"
    
    try:
        d_args = load_config(config_path)
        model = load_model(model_path, d_args, device)
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return

    # File uploader
    uploaded_file = st.file_uploader("Choose an audio file", type=['wav'])
    
    if uploaded_file is not None:
        # Create a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name

        try:
            # Load and process audio
            waveform = load_audio(tmp_path)
            
            # Add a button to trigger analysis
            if st.button("Analyze Audio"):
                with st.spinner("Analyzing audio..."):
                    real_score, fake_score, label, confidence = classify_audio(
                        model, waveform, device, threshold=0.7
                    )
                    
                    # Display results
                    st.write("---")
                    st.subheader("Results")
                    
                    # Create columns for better layout
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric("Classification", label)
                        st.metric("Confidence", f"{confidence*100:.1f}%")
                    
                    with col2:
                        st.write("Detailed Scores:")
                        st.write(f"Real Score: {real_score:.4f}")
                        st.write(f"Fake Score: {fake_score:.4f}")
                    
                    # Add a visual indicator
                    if label == "REAL":
                        st.success("✅ This audio appears to be genuine")
                    else:
                        st.error("❌ This audio appears to be fake")
                    
                    # Add audio player
                    st.audio(uploaded_file, format='audio/wav')
        
        except Exception as e:
            st.error(f"Error processing audio: {str(e)}")
        
        finally:
            # Clean up temporary file
            os.unlink(tmp_path)

if __name__ == "__main__":
    main() 