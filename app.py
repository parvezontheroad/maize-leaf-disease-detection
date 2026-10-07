import os
import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# -----------------------------------------------------------------------------
# 1. PAGE CONFIGURATION & HEADER
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Maize Disease Detector",
    page_icon="🌽",
    layout="centered"
)

st.title("🌽 Maize Leaf Disease Classifier")
st.write(
    "Upload a clear photograph of a maize leaf to detect fungal and bacterial "
    "diseases in real time."
)

# -----------------------------------------------------------------------------
# 2. CONFIGURATION & CLASS LABELS
# Update these names to match your exact training dataset folder names
# -----------------------------------------------------------------------------
CLASS_NAMES = ['Blight', 'Common Rust', 'Gray Leaf Spot', 'Healthy']
MODEL_WEIGHTS_PATH = "best_maize_model.pth"

# -----------------------------------------------------------------------------
# 3. CACHED MODEL LOADING
# -----------------------------------------------------------------------------
@st.cache_resource
def load_model():
    # Reconstruct the model architecture used during training
    model = models.efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, len(CLASS_NAMES))
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if os.path.exists(MODEL_WEIGHTS_PATH):
        model.load_state_dict(torch.load(MODEL_WEIGHTS_PATH, map_location=device))
        model.eval()
        return model, None
    else:
        return None, f"Weights file '{MODEL_WEIGHTS_PATH}' not found in repository root!"

model, error_msg = load_model()

if error_msg:
    st.error(error_msg)
    st.info("Make sure your trained model file (`best_maize_model.pth`) is committed to your GitHub repository.")
    st.stop()

# -----------------------------------------------------------------------------
# 4. IMAGE PREPROCESSING PIPELINE
# -----------------------------------------------------------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# -----------------------------------------------------------------------------
# 5. USER INTERFACE & INFERENCE
# -----------------------------------------------------------------------------
uploaded_file = st.file_uploader("Choose a maize leaf image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Render uploaded image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Leaf Image", use_container_width=True)

    if st.button("🔍 Analyze Leaf"):
        with st.spinner("Processing image through neural network..."):
            # Prepare tensor batch
            input_tensor = transform(image).unsqueeze(0)

            # Perform forward pass
            with torch.no_grad():
                outputs = model(input_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
                confidence, pred_idx = torch.max(probabilities, dim=0)

            pred_class = CLASS_NAMES[pred_idx.item()]
            confidence_pct = confidence.item() * 100

            st.divider()

            # Output results
            if pred_class == "Healthy":
                st.success(f"### Diagnosis: {pred_class}\n**Confidence Score:** {confidence_pct:.2f}%")
            else:
                st.warning(f"### Diagnosis: {pred_class}\n**Confidence Score:** {confidence_pct:.2f}%")

            st.progress(confidence.item())

            # Detailed probability distribution breakdown
            with st.expander("📊 View Detailed Class Probabilities"):
                for idx, class_name in enumerate(CLASS_NAMES):
                    prob = probabilities[idx].item() * 100
                    st.write(f"**{class_name}:** {prob:.2f}%")
