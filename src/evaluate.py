import os
import argparse
import torch
import torch.nn as nn
import numpy as np
from torchvision import models
from torchvision.models import ResNet18_Weights
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

from src.dataset import get_dataloader, CLASS_NAMES

# =========================
# LOAD MODEL
# =========================
def load_model(model_path, device):
    model = models.resnet18(weights=None)  # gunakan weights=None, karena kita pakai model sendiri
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, len(CLASS_NAMES))

    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)

    model = model.to(device)
    model.eval()
    return model

# =========================
# EVALUATION
# =========================
def evaluate(model, dataloader, device):
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return np.array(all_labels), np.array(all_preds)

# =========================
# SAVE METRICS
# =========================
def save_metrics(y_true, y_pred):
    os.makedirs("outputs", exist_ok=True)

    # Accuracy
    acc = accuracy_score(y_true, y_pred)
    with open("outputs/accuracy.txt", "w") as f:
        f.write(f"Accuracy: {acc:.4f}\n")

    # Classification report
    report = classification_report(
        y_true, y_pred, target_names=CLASS_NAMES, digits=4
    )
    with open("outputs/classification_report.txt", "w") as f:
        f.write(report)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.savefig("outputs/confusion_matrix_test.png")
    plt.close()

    print(f"\n[INFO] Metrics saved to 'outputs/'")
    print(f"Accuracy: {acc:.4f}")

# =========================
# MAIN
# =========================
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    # Load test data
    test_loader, _ = get_dataloader(
        args.data_dir,
        phase="test",
        batch_size=args.batch_size,
        shuffle=False
    )

    # Load model
    model = load_model(args.model, device)

    # Run evaluation
    y_true, y_pred = evaluate(model, test_loader, device)

    # Save metrics
    save_metrics(y_true, y_pred)

# =========================
# CLI
# =========================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--model", type=str, default="models/resnet18_best.pth")
    parser.add_argument("--batch_size", type=int, default=16)

    args = parser.parse_args()
    main(args)
