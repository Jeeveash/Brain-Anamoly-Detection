"""
Explainability and interpretability utilities for model predictions
Implements Grad-CAM for classification and mask overlays for segmentation
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Union, List
import cv2
import os
from pathlib import Path
from PIL import Image


class GradCAM:
    """
    Grad-CAM implementation using PyTorch hooks for CNN classification layers
    """
    
    def __init__(self, model: nn.Module, target_layer: Optional[nn.Module] = None):
        """
        Initialize Grad-CAM
        
        Args:
            model: PyTorch model
            target_layer: Target convolutional layer for Grad-CAM (auto-detected if None)
        """
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hook_handles = []
        
        # Auto-detect target layer if not provided
        if target_layer is None:
            self.target_layer = self._find_target_layer()
        
        # Register hooks
        self._register_hooks()
    
    def _find_target_layer(self) -> nn.Module:
        """
        Automatically find the last convolutional layer in the model
        
        Returns:
            Target convolutional layer
        """
        # Common patterns for medical imaging models (DenseNet, ResNet, etc.)
        for module in reversed(list(self.model.modules())):
            if isinstance(module, (nn.Conv2d, nn.Conv3d)):
                return module
        
        # Fallback: try to find any convolutional layer
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Conv3d)):
                return module
        
        raise ValueError("Could not find a convolutional layer for Grad-CAM")
    
    def _register_hooks(self):
        """Register forward and backward hooks"""
        def forward_hook(module, input, output):
            self.activations = output
        
        def backward_hook(module, grad_input, grad_output):
            # grad_output is a tuple, get the first element
            if grad_output is not None and len(grad_output) > 0:
                self.gradients = grad_output[0]
        
        handle_forward = self.target_layer.register_forward_hook(forward_hook)
        
        # Try full_backward_hook first (PyTorch 1.9+), fallback to backward_hook
        try:
            handle_backward = self.target_layer.register_full_backward_hook(backward_hook)
        except AttributeError:
            # Fallback for older PyTorch versions
            handle_backward = self.target_layer.register_backward_hook(backward_hook)
        
        self.hook_handles = [handle_forward, handle_backward]
    
    def _remove_hooks(self):
        """Remove registered hooks"""
        for handle in self.hook_handles:
            handle.remove()
        self.hook_handles = []
    
    def generate_cam(self, input_tensor: torch.Tensor, target_class: Optional[int] = None) -> np.ndarray:
        """
        Generate Grad-CAM heatmap
        
        Args:
            input_tensor: Input tensor (batch, channels, height, width)
            target_class: Target class index (None for predicted class)
            
        Returns:
            Grad-CAM heatmap as numpy array
        """
        self.model.eval()
        input_tensor.requires_grad = True
        
        # Forward pass
        output = self.model(input_tensor)
        
        # Get target class
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        
        # Backward pass
        self.model.zero_grad()
        target_score = output[:, target_class].sum()
        target_score.backward()
        
        # Get gradients and activations
        gradients = self.gradients[0].cpu().data.numpy()
        activations = self.activations[0].cpu().data.numpy()
        
        # Handle different dimensions (2D or 3D)
        if len(gradients.shape) == 4:  # 2D: (channels, height, width)
            # Compute weights (global average pooling of gradients)
            weights = np.mean(gradients, axis=(1, 2), keepdims=True)
            
            # Generate CAM
            cam = np.zeros(activations.shape[1:], dtype=np.float32)
            for i, w in enumerate(weights):
                cam += w[0, 0] * activations[i, :, :]
        elif len(gradients.shape) == 5:  # 3D: (channels, depth, height, width)
            # For 3D, use middle slice or average across depth
            mid_depth = gradients.shape[1] // 2
            gradients_2d = gradients[:, mid_depth, :, :]
            activations_2d = activations[:, mid_depth, :, :]
            
            weights = np.mean(gradients_2d, axis=(1, 2), keepdims=True)
            cam = np.zeros(activations_2d.shape[1:], dtype=np.float32)
            for i, w in enumerate(weights):
                cam += w[0, 0] * activations_2d[i, :, :]
        else:
            raise ValueError(f"Unsupported gradient shape: {gradients.shape}")
        
        # Apply ReLU
        cam = np.maximum(cam, 0)
        
        # Normalize
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()
        
        # Resize to input size if needed (for 2D only)
        if len(cam.shape) == 2 and cam.shape != input_tensor.shape[2:]:
            cam = cv2.resize(cam, tuple(reversed(input_tensor.shape[2:])), interpolation=cv2.INTER_LINEAR)
        
        return cam


def generate_gradcam_explanation(
    model: nn.Module, 
    image: Union[np.ndarray, torch.Tensor],
    target_layer: Optional[nn.Module] = None,
    target_class: Optional[int] = None,
    device: Optional[torch.device] = None
) -> np.ndarray:
    """
    Generate Grad-CAM heatmap for model prediction explanation
    
    Args:
        model: Trained PyTorch model
        image: Input image (numpy array or tensor)
        target_layer: Target convolutional layer (auto-detected if None)
        target_class: Target class index (None for predicted class)
        device: Device to run inference on
        
    Returns:
        Grad-CAM heatmap as numpy array
    """
    # Convert to tensor if needed
    if isinstance(image, np.ndarray):
        image_tensor = torch.from_numpy(image).float()
    else:
        image_tensor = image
    
    # Add batch dimension if needed
    if image_tensor.dim() == 3:
        image_tensor = image_tensor.unsqueeze(0)
    
    # Move to device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_tensor = image_tensor.to(device)
    model = model.to(device)
    
    # Generate CAM
    gradcam = GradCAM(model, target_layer)
    try:
        cam = gradcam.generate_cam(image_tensor, target_class)
    finally:
        gradcam._remove_hooks()
    
    return cam


def overlay_segmentation_mask(
    image_slice: np.ndarray,
    mask: np.ndarray,
    alpha: float = 0.5,
    colormap: int = cv2.COLORMAP_JET
) -> np.ndarray:
    """
    Overlay segmentation mask on MRI slice
    
    Args:
        image_slice: Original MRI slice (2D or 3D)
        mask: Segmentation mask (same spatial dimensions as image_slice)
        alpha: Transparency factor for overlay (0-1)
        colormap: OpenCV colormap to use for mask visualization
        
    Returns:
        Overlaid image
    """
    # Normalize image slice to 0-255
    if image_slice.dtype != np.uint8:
        img_normalized = ((image_slice - image_slice.min()) / 
                         (image_slice.max() - image_slice.min() + 1e-8) * 255).astype(np.uint8)
    else:
        img_normalized = image_slice.copy()
    
    # Convert grayscale to RGB if needed
    if len(img_normalized.shape) == 2:
        img_rgb = cv2.cvtColor(img_normalized, cv2.COLOR_GRAY2RGB)
    elif len(img_normalized.shape) == 3 and img_normalized.shape[2] == 1:
        img_rgb = cv2.cvtColor(img_normalized.squeeze(), cv2.COLOR_GRAY2RGB)
    else:
        img_rgb = img_normalized
    
    # Normalize mask to 0-255
    if mask.dtype != np.uint8:
        mask_normalized = ((mask - mask.min()) / 
                          (mask.max() - mask.min() + 1e-8) * 255).astype(np.uint8)
    else:
        mask_normalized = mask.copy()
    
    # Apply colormap to mask
    if len(mask_normalized.shape) == 2:
        mask_colored = cv2.applyColorMap(mask_normalized, colormap)
    else:
        mask_colored = mask_normalized
    
    # Resize mask if needed to match image
    if mask_colored.shape[:2] != img_rgb.shape[:2]:
        mask_colored = cv2.resize(mask_colored, 
                                (img_rgb.shape[1], img_rgb.shape[0]),
                                interpolation=cv2.INTER_NEAREST)
    
    # Blend images
    overlaid = cv2.addWeighted(img_rgb, 1 - alpha, mask_colored, alpha, 0)
    
    return overlaid


def overlay_heatmap(
    image: np.ndarray, 
    heatmap: np.ndarray, 
    alpha: float = 0.4,
    colormap: int = cv2.COLORMAP_JET
) -> np.ndarray:
    """
    Overlay heatmap on original image
    
    Args:
        image: Original image (can be C,H,W or H,W,C or H,W)
        heatmap: Heatmap to overlay
        alpha: Transparency factor
        colormap: OpenCV colormap to use
        
    Returns:
        Overlaid image in (H,W,C) format
    """
    # Convert from channels-first (C,H,W) to channels-last (H,W,C) if needed
    if len(image.shape) == 3:
        if image.shape[0] in [1, 3] and image.shape[0] < image.shape[1]:
            # Channels first: (C, H, W) -> (H, W, C)
            image = np.transpose(image, (1, 2, 0))
    
    # If single channel at end, squeeze to (H, W)
    if len(image.shape) == 3 and image.shape[2] == 1:
        image = image.squeeze(axis=2)
    
    # Normalize image to 0-255
    if image.dtype != np.uint8:
        img_normalized = ((image - image.min()) / 
                         (image.max() - image.min() + 1e-8) * 255).astype(np.uint8)
    else:
        img_normalized = image.copy()
    
    # Normalize heatmap to 0-255
    heatmap_normalized = ((heatmap - heatmap.min()) / 
                          (heatmap.max() - heatmap.min() + 1e-8) * 255).astype(np.uint8)
    
    # Apply colormap
    heatmap_colored = cv2.applyColorMap(heatmap_normalized, colormap)
    
    # Convert image to RGB if grayscale
    if len(img_normalized.shape) == 2:
        img_rgb = cv2.cvtColor(img_normalized, cv2.COLOR_GRAY2RGB)
    elif len(img_normalized.shape) == 3 and img_normalized.shape[2] == 1:
        img_rgb = cv2.cvtColor(img_normalized.squeeze(), cv2.COLOR_GRAY2RGB)
    elif len(img_normalized.shape) == 3 and img_normalized.shape[2] == 3:
        img_rgb = img_normalized
    else:
        img_rgb = img_normalized
    
    # Resize heatmap if needed
    if heatmap_colored.shape[:2] != img_rgb.shape[:2]:
        heatmap_colored = cv2.resize(heatmap_colored, 
                                    (img_rgb.shape[1], img_rgb.shape[0]),
                                    interpolation=cv2.INTER_LINEAR)
    
    # Blend images
    overlaid = cv2.addWeighted(img_rgb, 1 - alpha, heatmap_colored, alpha, 0)
    return overlaid


def save_explanation_overlay(
    overlay_image: np.ndarray,
    output_path: str,
    filename: Optional[str] = None
) -> str:
    """
    Save overlay image as PNG file
    
    Args:
        overlay_image: Overlaid image (numpy array, can be (C,H,W) or (H,W,C) or (H,W))
        output_path: Directory or full path to save file
        filename: Filename (optional, auto-generated if None)
        
    Returns:
        Full path to saved file
    """
    # Create output directory if needed
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    if filename:
        full_path = output_dir / filename
    else:
        if output_dir.is_dir():
            # Generate filename with timestamp
            import datetime
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"overlay_{timestamp}.png"
            full_path = output_dir / filename
        else:
            full_path = output_dir
    
    # Convert from channels-first (C,H,W) to channels-last (H,W,C) if needed
    if len(overlay_image.shape) == 3:
        # Check if channels-first format (C,H,W) - typical for PyTorch/MONAI
        if overlay_image.shape[0] in [1, 3] and overlay_image.shape[0] < overlay_image.shape[1]:
            # Channels first: (C, H, W) -> (H, W, C)
            overlay_image = np.transpose(overlay_image, (1, 2, 0))
            print(f"[DEBUG] Converted from (C,H,W) to (H,W,C), new shape: {overlay_image.shape}")
    
    # If single channel, squeeze to (H, W)
    if len(overlay_image.shape) == 3 and overlay_image.shape[2] == 1:
        overlay_image = overlay_image.squeeze(axis=2)
        print(f"[DEBUG] Squeezed single channel, new shape: {overlay_image.shape}")
    
    # For 3-channel grayscale (all channels identical), convert to single channel
    if len(overlay_image.shape) == 3 and overlay_image.shape[2] == 3:
        # Check if all channels are identical (grayscale stored as RGB)
        if np.allclose(overlay_image[:,:,0], overlay_image[:,:,1]) and np.allclose(overlay_image[:,:,1], overlay_image[:,:,2]):
            overlay_image = overlay_image[:,:,0]
            print(f"[DEBUG] Converted 3-channel grayscale to single channel, shape: {overlay_image.shape}")
    
    # Ensure image is uint8 and in correct format
    if overlay_image.dtype != np.uint8:
        overlay_image = ((overlay_image - overlay_image.min()) / 
                        (overlay_image.max() - overlay_image.min() + 1e-8) * 255).astype(np.uint8)
    
    # Convert BGR to RGB if needed (OpenCV uses BGR) - only for 3-channel images
    if len(overlay_image.shape) == 3 and overlay_image.shape[2] == 3:
        # Note: We assume our images are already RGB from MONAI, so skip this conversion
        # overlay_image = cv2.cvtColor(overlay_image, cv2.COLOR_BGR2RGB)
        pass
    
    print(f"[DEBUG] Final image shape before save: {overlay_image.shape}, dtype: {overlay_image.dtype}")
    
    # Save as PNG
    pil_image = Image.fromarray(overlay_image)
    pil_image.save(str(full_path), "PNG")
    
    return str(full_path)


def generate_segmentation_overlay(
    image_volume: np.ndarray,
    segmentation_mask: np.ndarray,
    slice_index: Optional[int] = None,
    output_dir: str = "data/outputs",
    prefix: str = "segmentation_overlay",
    alpha: float = 0.5
) -> List[str]:
    """
    Generate and save segmentation mask overlays on MRI slices
    
    Args:
        image_volume: 3D MRI volume (can be multi-channel)
        segmentation_mask: 3D segmentation mask
        slice_index: Specific slice index to overlay (None for all slices)
        output_dir: Output directory for saved images
        prefix: Prefix for output filenames
        alpha: Transparency factor
        
    Returns:
        List of paths to saved overlay images
    """
    saved_paths = []
    
    # Extract 2D slice from volume
    if len(image_volume.shape) == 4:
        # Multi-channel volume: use first channel or average
        image_2d = image_volume[0] if image_volume.shape[0] == 1 else np.mean(image_volume, axis=0)
    elif len(image_volume.shape) == 3:
        image_2d = image_volume
    else:
        raise ValueError(f"Unsupported image volume shape: {image_volume.shape}")
    
    # Get slice indices to process
    if slice_index is not None:
        slice_indices = [slice_index]
    else:
        # Process middle slices or all slices (limit to 10 for performance)
        num_slices = image_2d.shape[0] if len(image_2d.shape) == 3 else 1
        if num_slices > 10:
            # Sample evenly spaced slices
            slice_indices = np.linspace(0, num_slices - 1, 10, dtype=int).tolist()
        else:
            slice_indices = list(range(num_slices))
    
    # Process each slice
    for idx in slice_indices:
        if len(image_2d.shape) == 3:
            slice_img = image_2d[idx]
            slice_mask = segmentation_mask[idx] if len(segmentation_mask.shape) == 3 else segmentation_mask
        else:
            slice_img = image_2d
            slice_mask = segmentation_mask
        
        # Generate overlay
        overlay = overlay_segmentation_mask(slice_img, slice_mask, alpha=alpha)
        
        # Save overlay
        filename = f"{prefix}_slice_{idx:03d}.png"
        saved_path = save_explanation_overlay(overlay, output_dir, filename)
        saved_paths.append(saved_path)
    
    return saved_paths


def generate_classification_gradcam(
    model: nn.Module,
    image: Union[np.ndarray, torch.Tensor],
    target_layer: Optional[nn.Module] = None,
    target_class: Optional[int] = None,
    output_dir: str = "data/outputs",
    filename: Optional[str] = None,
    device: Optional[torch.device] = None
) -> str:
    """
    Generate Grad-CAM heatmap for classification model and save as PNG
    
    Args:
        model: PyTorch classification model
        image: Input image
        target_layer: Target layer for Grad-CAM
        target_class: Target class (None for predicted class)
        output_dir: Output directory
        filename: Output filename (auto-generated if None)
        device: Device to run on
        
    Returns:
        Path to saved Grad-CAM overlay image
    """
    # Generate Grad-CAM heatmap
    cam = generate_gradcam_explanation(model, image, target_layer, target_class, device)
    
    # Prepare image for overlay
    if isinstance(image, torch.Tensor):
        image_np = image.squeeze().cpu().numpy()
    else:
        image_np = image.copy()
    
    # Remove batch dimension if present
    if len(image_np.shape) == 4:
        image_np = image_np[0]
    if len(image_np.shape) == 3 and image_np.shape[0] == 1:
        image_np = image_np[0]
    
    # Generate overlay
    overlay = overlay_heatmap(image_np, cam, alpha=0.4)
    
    # Save overlay
    if filename is None:
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"gradcam_{timestamp}.png"
    
    saved_path = save_explanation_overlay(overlay, output_dir, filename)
    return saved_path


def explain_prediction(
    prediction: Dict[str, Any],
    explanation_type: str = 'gradcam'
) -> Dict[str, Any]:
    """
    Generate explanation for model prediction

    Args:
        prediction: Model prediction dictionary
        explanation_type: Type of explanation ('gradcam', 'saliency', etc.)

    Returns:
        Explanation dictionary with visualization
    """
    explanation = {
        'explanation_type': explanation_type,
        'highlighted_regions': [],
        'feature_importance': {},
        'visualization': None
    }

    # Extract prediction details
    probability = prediction.get('probability', 0.0)
    mask = prediction.get('mask', None)
    uncertainty = prediction.get('uncertainty', 0.0)

    if explanation_type == 'gradcam':
        # For Grad-CAM, highlight regions with high activation
        if mask is not None and isinstance(mask, np.ndarray):
            # Use mask as explanation for segmentation models
            explanation['highlighted_regions'] = np.where(mask > 0)
            explanation['feature_importance'] = {
                'mask_coverage': float(np.sum(mask > 0) / mask.size),
                'max_intensity': float(np.max(mask)) if mask.size > 0 else 0.0
            }
        else:
            # For classification, use probability as importance
            explanation['feature_importance'] = {
                'prediction_probability': probability,
                'uncertainty': uncertainty
            }

    elif explanation_type == 'saliency':
        # Simple saliency based on prediction confidence
        explanation['feature_importance'] = {
            'confidence_score': probability * (1 - uncertainty),
            'prediction_strength': probability
        }

    # Generate visualization path if available
    overlay_path = prediction.get('overlay_image_path', None)
    if overlay_path:
        explanation['visualization'] = overlay_path

    return explanation


def generate_prediction_report(
    prediction: Dict[str, Any],
    explanation: Optional[Dict[str, Any]] = None
) -> str:
    """
    Generate human-readable report of prediction and explanation
    
    Args:
        prediction: Model prediction dictionary
        explanation: Optional explanation dictionary
        
    Returns:
        Formatted report string
    """
    report_lines = [
        "=== Prediction Report ===",
        f"Anomaly Detected: {prediction.get('anomaly_detected', False)}",
        f"Confidence: {prediction.get('confidence', 0.0):.2%}",
    ]
    
    if explanation:
        report_lines.append("\n=== Explanation ===")
        report_lines.append(f"Explanation Type: {explanation.get('explanation_type', 'N/A')}")
    
    return "\n".join(report_lines)
