"""
Preprocessing utilities for brain scan images
"""

import numpy as np
from typing import Tuple, Optional

def normalize_intensity(image: np.ndarray, 
                       min_percentile: float = 1.0,
                       max_percentile: float = 99.0) -> np.ndarray:
    """
    Normalize image intensity values
    
    Args:
        image: Input image array
        min_percentile: Minimum percentile for clipping
        max_percentile: Maximum percentile for clipping
        
    Returns:
        Normalized image
    """
    p_min = np.percentile(image, min_percentile)
    p_max = np.percentile(image, max_percentile)
    image_clipped = np.clip(image, p_min, p_max)
    image_normalized = (image_clipped - p_min) / (p_max - p_min + 1e-8)
    return image_normalized

def resize_image(image: np.ndarray,
                target_size: Tuple[int, int, int],
                interpolation: str = 'linear') -> np.ndarray:
    """
    Resize image to target dimensions

    Args:
        image: Input image array
        target_size: Target dimensions (depth, height, width)
        interpolation: Interpolation method

    Returns:
        Resized image
    """
    try:
        from scipy.ndimage import zoom
    except ImportError:
        raise ImportError("scipy is required for image resizing. Install with: pip install scipy")

    # Calculate zoom factors
    current_shape = image.shape[-3:]  # Last 3 dimensions (depth, height, width)
    zoom_factors = [target / current for target, current in zip(target_size, current_shape)]

    # Choose interpolation order
    if interpolation == 'nearest':
        order = 0
    elif interpolation == 'linear':
        order = 1
    elif interpolation == 'cubic':
        order = 3
    else:
        order = 1  # Default to linear

    # Apply zoom
    resized = zoom(image, zoom_factors, order=order, mode='nearest')

    return resized

def apply_window_level(image: np.ndarray,
                      window_center: float,
                      window_width: float) -> np.ndarray:
    """
    Apply window/level transformation for CT/MRI visualization
    
    Args:
        image: Input image array
        window_center: Window center value
        window_width: Window width value
        
    Returns:
        Windowed image
    """
    window_min = window_center - window_width / 2
    window_max = window_center + window_width / 2
    windowed = np.clip(image, window_min, window_max)
    windowed = (windowed - window_min) / (window_width + 1e-8)
    return windowed

def skull_stripping(image: np.ndarray) -> np.ndarray:
    """
    Perform skull stripping to isolate brain tissue

    Args:
        image: Input brain scan image

    Returns:
        Skull-stripped image
    """
    try:
        from scipy.ndimage import morphology
    except ImportError:
        raise ImportError("scipy is required for skull stripping. Install with: pip install scipy")

    # Simple skull stripping using morphological operations
    # This is a basic implementation - in production, use dedicated tools like FSL BET or ROBEX

    # Assume image is 3D (depth, height, width)
    if len(image.shape) != 3:
        raise ValueError("Skull stripping expects 3D image")

    # Normalize image to 0-1 range
    img_norm = normalize_intensity(image)

    # Create brain mask using simple thresholding
    # This is a very basic approach - real skull stripping uses more sophisticated methods
    brain_mask = (img_norm > 0.1).astype(np.uint8)  # Threshold to separate brain from background

    # Apply morphological operations to clean up the mask
    # Remove small holes and noise
    brain_mask = morphology.binary_fill_holes(brain_mask)
    brain_mask = morphology.binary_opening(brain_mask, iterations=2)
    brain_mask = morphology.binary_closing(brain_mask, iterations=2)

    # Apply mask to original image
    skull_stripped = image * brain_mask

    return skull_stripped

def augment_image(image: np.ndarray,
                 rotation_range: Optional[float] = None,
                 flip_axes: Optional[Tuple[int, ...]] = None) -> np.ndarray:
    """
    Apply data augmentation transformations

    Args:
        image: Input image
        rotation_range: Maximum rotation angle in degrees
        flip_axes: Axes along which to flip

    Returns:
        Augmented image
    """
    try:
        from scipy.ndimage import rotate
    except ImportError:
        raise ImportError("scipy is required for image augmentation. Install with: pip install scipy")

    augmented = image.copy()

    # Apply rotation if specified
    if rotation_range is not None and rotation_range > 0:
        # Random rotation angle
        angle = np.random.uniform(-rotation_range, rotation_range)
        augmented = rotate(augmented, angle, axes=(1, 2), reshape=False, mode='nearest')

    # Apply flipping if specified
    if flip_axes is not None:
        for axis in flip_axes:
            if np.random.random() > 0.5:  # 50% chance to flip
                augmented = np.flip(augmented, axis=axis)

    return augmented

