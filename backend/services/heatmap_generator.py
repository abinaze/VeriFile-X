"""
Patch-based Grad-CAM heatmap generator for manipulation localization.

Uses the trained EfficientNet-B0 model to produce a spatial activation
map highlighting regions most indicative of AI generation.

Method:
1. Register a forward hook on the last convolutional block of EfficientNet-B0
2. Run a forward pass with gradients enabled
3. Backpropagate the AI-probability score
4. Average gradient-weighted activations across channels (Grad-CAM)
5. Resize to original image dimensions
6. Apply COLORMAP_JET and alpha-blend with original image
"""
from backend.core.logger import setup_logger
import numpy as np
from pathlib import Path
from typing import Dict, Any
from io import BytesIO

logger = setup_logger(__name__)

_MODEL_PATH = Path(__file__).parent.parent.parent / "data" / "reference" / "own_embedding_model.pt"



def generate_heatmap(image_bytes: bytes, filename: str = "unknown") -> Dict[str, Any]:
    """
    Generate Grad-CAM localization heatmap.

    Args:
        image_bytes: Raw image bytes
        filename:    Image filename for logging

    Returns:
        Dict with keys:
          heatmap_b64   - base64-encoded PNG of the blended heatmap
          width, height - original image dimensions
          method        - 'gradcam' or 'neutral_fallback'
    """
    import base64
    import cv2
    from PIL import Image

    original_pil = Image.open(BytesIO(image_bytes)).convert("RGB")
    orig_w, orig_h = original_pil.size
    original_np = np.array(original_pil)

    if not _MODEL_PATH.exists():
        logger.warning("EfficientNet model absent — returning neutral heatmap")
        grey = np.full((orig_h, orig_w), 128, dtype=np.uint8)
        heatmap_color = cv2.applyColorMap(grey, cv2.COLORMAP_JET)
        blended = cv2.addWeighted(
            cv2.cvtColor(original_np, cv2.COLOR_RGB2BGR), 0.6,
            heatmap_color, 0.4, 0
        )
        _, buf = cv2.imencode(".png", blended)
        return {
            "heatmap_b64": base64.b64encode(buf).decode(),
            "width": orig_w,
            "height": orig_h,
            "method": "neutral_fallback",
        }

    try:
        import torch
        from torchvision import transforms
        from backend.services.own_detector.model import OwnEmbeddingModel

        device = "cuda" if torch.cuda.is_available() else "cpu"

        # Load model with gradients enabled (no torch.no_grad here)
        model = OwnEmbeddingModel()
        state = torch.load(_MODEL_PATH, map_location=device, weights_only=True)
        model.load_state_dict(state)
        model = model.to(device)
        model.eval()

        # Register Grad-CAM hooks on last conv block of EfficientNet features
        activations: list = []
        gradients: list = []

        def fwd_hook(module, inp, out):
            activations.append(out.detach())

        def bwd_hook(module, grad_in, grad_out):
            gradients.append(grad_out[0].detach())

        # EfficientNet-B0 features[-1] is the last MBConv block
        target_layer = model.features[-1]
        fwd_handle = target_layer.register_forward_hook(fwd_hook)
        bwd_handle = target_layer.register_full_backward_hook(bwd_hook)

        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])

        tensor = transform(original_pil).unsqueeze(0).to(device)

        # Forward pass — keep graph for backward
        _, prob = model(tensor)
        ai_score = prob.squeeze()

        # Backward pass on the AI probability
        model.zero_grad()
        ai_score.backward()

        fwd_handle.remove()
        bwd_handle.remove()

        if not activations or not gradients:
            raise RuntimeError("Hooks did not fire — layer not found")

        act = activations[0].squeeze(0)    # (C, H, W)
        grad = gradients[0].squeeze(0)     # (C, H, W)

        # Global average pool gradients over spatial dims
        weights = grad.mean(dim=(1, 2), keepdim=True)  # (C, 1, 1)

        # Weighted combination of activations
        cam = (weights * act).sum(dim=0)   # (H, W)
        cam = torch.relu(cam)
        cam = cam.cpu().numpy()

        # Normalize to [0, 255]
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        cam_uint8 = (cam * 255).astype(np.uint8)

        # Resize to original image dimensions
        cam_resized = cv2.resize(
            cam_uint8, (orig_w, orig_h),
            interpolation=cv2.INTER_CUBIC
        )

        # Apply colormap and blend
        heatmap_color = cv2.applyColorMap(cam_resized, cv2.COLORMAP_JET)
        original_bgr  = cv2.cvtColor(original_np, cv2.COLOR_RGB2BGR)
        blended = cv2.addWeighted(original_bgr, 0.6, heatmap_color, 0.4, 0)

        _, buf = cv2.imencode(".png", blended)
        heatmap_b64 = base64.b64encode(buf).decode()

        logger.info(
            f"Grad-CAM heatmap generated for {filename}: "
            f"ai_score={float(ai_score):.3f}, size={orig_w}x{orig_h}"
        )

        return {
            "heatmap_b64": heatmap_b64,
            "width": orig_w,
            "height": orig_h,
            "method": "gradcam",
        }

    except Exception as e:
        logger.warning(f"Grad-CAM failed for {filename}: {e} — returning neutral fallback")
        grey = np.full((orig_h, orig_w), 128, dtype=np.uint8)
        heatmap_color = cv2.applyColorMap(grey, cv2.COLORMAP_JET)
        original_bgr  = cv2.cvtColor(original_np, cv2.COLOR_RGB2BGR)
        blended = cv2.addWeighted(original_bgr, 0.6, heatmap_color, 0.4, 0)
        _, buf = cv2.imencode(".png", blended)
        return {
            "heatmap_b64": base64.b64encode(buf).decode(),
            "width": orig_w,
            "height": orig_h,
            "method": "neutral_fallback",
        }
