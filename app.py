import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import cv2
import os
import plotly.graph_objects as go

from src.gradcam import GradCAM, overlay_cam_on_image
from src.dataset import CLASS_NAMES  # ["NORMAL", "PNEUMONIA", "TUBERCULOSIS", "UNKNOWN"]

# =========================
# SETTINGS
# =========================
MODEL_PATH = "models/resnet18_best.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# =========================
# UTILS
# =========================
def load_model(model_path, device):
    model = models.resnet18(pretrained=False)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, len(CLASS_NAMES))
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model

def preprocess_image(image: Image.Image):
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485,0.456,0.406],
            std=[0.229,0.224,0.225]
        )
    ])
    image_tensor = transform(image).unsqueeze(0)
    image_np = np.array(image)
    return image_np, image_tensor

def run_inference(image: Image.Image, model, device, alpha=0.3):
    image_np, input_tensor = preprocess_image(image)
    input_tensor = input_tensor.to(device)

    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.softmax(outputs, dim=1).cpu().numpy().squeeze()
        pred_idx = np.argmax(probs)
        pred_class = CLASS_NAMES[pred_idx]
        confidence = probs[pred_idx]

    # Grad-CAM
    gradcam = GradCAM(model, model.layer4[-1])
    cam = gradcam.generate(input_tensor, class_idx=pred_idx)
    cam_resized = cv2.resize(cam, (image_np.shape[1], image_np.shape[0]))
    _, overlay = overlay_cam_on_image(image_np, cam_resized, alpha=alpha)

    return pred_class, confidence, probs, overlay, image_np

# =========================
# STREAMLIT APP
# =========================
st.set_page_config(page_title="Lung Disease Grad-CAM", layout="wide")
st.title("🫁 Lung Disease Classification")
st.markdown("Upload X-ray image, lihat prediksi dan Grad-CAM overlay, lalu cek probabilitas setiap kelas.")

uploaded_file = st.file_uploader("Upload X-ray Image", type=["jpg","jpeg","png"])
alpha = st.slider("Grad-CAM overlay transparency", 0.0, 1.0, 0.3, 0.05)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    model = load_model(MODEL_PATH, DEVICE)
    pred_class, confidence, probs, overlay, orig_image = run_inference(image, model, DEVICE, alpha=alpha)

    # =========================
    # Prediksi di atas gambar
    # =========================
    st.subheader(f"Prediksi: **{pred_class} ({confidence*100:.2f}%)**")

    # =========================
    # Layout gambar
    # =========================
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original X-ray")
        st.image(orig_image, use_container_width=True)

    with col2:
        st.subheader("Grad-CAM Overlay")
        st.image(overlay, use_container_width=True)

    # =========================
    # Probabilities bar chart horizontal menggunakan Plotly
    # =========================
    colors = ["#4CAF50", "#FF9800", "#2196F3", "#9C27B0"]  # bisa diganti
    fig = go.Figure()
    for i, cls in enumerate(CLASS_NAMES):
        fig.add_trace(go.Bar(
            y=[cls],
            x=[probs[i]*100],
            name=cls,
            orientation='h',
            marker=dict(color=colors[i], line=dict(color='black', width=1)),
            text=f"{probs[i]*100:.1f}%",
            textposition='inside'
        ))
    fig.update_layout(
        barmode='stack',
        xaxis=dict(title="Probability (%)", range=[0,100]),
        yaxis=dict(autorange="reversed"),  # supaya NORMAL di atas
        height=300,
        showlegend=False,
        margin=dict(l=50, r=50, t=10, b=10)
    )
    st.plotly_chart(fig, use_container_width=True)

    # Save overlay
    base_name = os.path.splitext(os.path.basename(uploaded_file.name))[0]
    overlay_path = os.path.join(RESULTS_DIR, f"{base_name}_overlay.jpg")
    cv2.imwrite(overlay_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    st.success(f"Overlay saved to {RESULTS_DIR}")
