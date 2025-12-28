import torch
import torch.nn.functional as F
import cv2
import numpy as np
import os

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        # Forward hook untuk menyimpan aktivasi
        def forward_hook(module, input, output):
            self.activations = output.detach()

        # Full backward hook untuk menyimpan gradien
        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate(self, input_tensor, class_idx=None):
        """
        input_tensor: [1,3,H,W]
        class_idx: target class (optional)
        """
        self.model.eval()
        output = self.model(input_tensor)

        if class_idx is None:
            class_idx = output.argmax(dim=1).item()

        self.model.zero_grad()
        score = output[:, class_idx]
        score.backward(retain_graph=True)

        gradients = self.gradients        # [B,C,H,W]
        activations = self.activations    # [B,C,H,W]

        weights = gradients.mean(dim=(2,3), keepdim=True)
        cam = (weights * activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = cam.squeeze().cpu().numpy()

        # normalize 0-1
        cam -= cam.min()
        if cam.max() != 0:
            cam /= cam.max()

        return cam

# =========================
# VISUALIZATION UTILS
# =========================
def overlay_cam_on_image(image_np, cam, alpha=0.5):
    """
    image_np: numpy array [H,W,3] RGB uint8
    cam: numpy array [H,W] 0-1
    """
    # Resize CAM ke ukuran image
    cam_resized = cv2.resize(cam, (image_np.shape[1], image_np.shape[0]))

    # Heatmap RGB
    heatmap = cv2.applyColorMap(np.uint8(255*cam_resized), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    # Pastikan image uint8
    image_uint8 = np.uint8(image_np)

    # Overlay
    overlay = cv2.addWeighted(image_uint8, 1-alpha, heatmap, alpha, 0)

    return heatmap, overlay

# =========================
# SAVE UTILS
# =========================
def save_cam_outputs(image_name, heatmap, overlay, save_dir="results"):
    os.makedirs(save_dir, exist_ok=True)

    heatmap_path = os.path.join(save_dir, f"{image_name}_heatmap.jpg")
    overlay_path = os.path.join(save_dir, f"{image_name}_overlay.jpg")

    cv2.imwrite(heatmap_path, cv2.cvtColor(heatmap, cv2.COLOR_RGB2BGR))
    cv2.imwrite(overlay_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

    print(f"[INFO] Grad-CAM saved: {heatmap_path}, {overlay_path}")
