import torch
import torch.nn as nn
import numpy as np
import cv2
from torchvision import models, transforms
import os
from PIL import Image
import random

from src.dataset import CLASS_NAMES
from src.gradcam import GradCAM, overlay_cam_on_image

# =========================
# SETTINGS
# =========================
MODEL_PATH = "models/resnet18_best.pth"
TEST_DIR = "data/test"      # folder test utama
SAVE_DIR = "results"        # folder hasil
os.makedirs(SAVE_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# PREPROCESS IMAGE
# =========================
def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485,0.456,0.406],
            std=[0.229,0.224,0.225]
        )
    ])
    image_pil = Image.open(image_path).convert("RGB")
    image_tensor = transform(image_pil).unsqueeze(0)
    image_np = np.array(image_pil)
    return image_np, image_tensor

# =========================
# LOAD MODEL
# =========================
def load_model(model_path, device):
    model = models.resnet18(pretrained=False)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, len(CLASS_NAMES))
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model

# =========================
# RUN INFERENCE
# =========================
def run_inference_on_image(image_path, model, device):
    image_np, input_tensor = preprocess_image(image_path)
    input_tensor = input_tensor.to(device)

    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.softmax(outputs, dim=1)
        conf, pred_idx = torch.max(probs, dim=1)

    pred_class = CLASS_NAMES[pred_idx.item()]
    confidence = conf.item()

    # Grad-CAM
    gradcam = GradCAM(model, model.layer4[-1])
    cam = gradcam.generate(input_tensor, class_idx=pred_idx.item())
    cam = cv2.resize(cam, (image_np.shape[1], image_np.shape[0]))
    heatmap, overlay = overlay_cam_on_image(image_np, cam)

    # Save results
    base_name = os.path.basename(image_path).split(".")[0]
    overlay_path = os.path.join(SAVE_DIR, f"{base_name}_overlay.jpg")
    heatmap_path = os.path.join(SAVE_DIR, f"{base_name}_heatmap.jpg")

    cv2.imwrite(overlay_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    cv2.imwrite(heatmap_path, cv2.cvtColor(heatmap, cv2.COLOR_RGB2BGR))

    print(f"[INFO] {base_name} -> {pred_class} ({confidence:.4f})")
    print(f"[INFO] Results saved to {SAVE_DIR}")

# =========================
# MAIN
# =========================
if __name__ == "__main__":
    # Ambil random class folder
    class_folders = [d for d in os.listdir(TEST_DIR) if os.path.isdir(os.path.join(TEST_DIR, d))]
    if not class_folders:
        raise FileNotFoundError(f"No class folders found in {TEST_DIR}")

    selected_class = random.choice(class_folders)
    folder_path = os.path.join(TEST_DIR, selected_class)

    # Ambil random image dari folder itu
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith((".jpg",".jpeg",".png"))]
    if not image_files:
        raise FileNotFoundError(f"No image found in folder: {folder_path}")

    selected_image = random.choice(image_files)
    image_path = os.path.join(folder_path, selected_image)

    print(f"[INFO] Selected class: {selected_class}, image: {selected_image}")

    # Load model & run inference
    model = load_model(MODEL_PATH, DEVICE)
    run_inference_on_image(image_path, model, DEVICE)
