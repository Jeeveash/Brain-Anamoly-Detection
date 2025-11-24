"""
Postprocessing utilities for model predictions
"""

import numpy as np
from typing import Dict, Any, List, Tuple

def apply_threshold(prediction: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """
    Apply confidence threshold to prediction probabilities
    
    Args:
        prediction: Prediction probability array
        threshold: Confidence threshold
        
    Returns:
        Binary prediction array
    """
    return (prediction >= threshold).astype(np.uint8)

def extract_bounding_boxes(mask: np.ndarray) -> List[Dict[str, int]]:
    """
    Extract bounding boxes from segmentation mask

    Args:
        mask: Binary segmentation mask

    Returns:
        List of bounding boxes with 'x', 'y', 'z', 'width', 'height', 'depth' keys
    """
    try:
        from scipy.ndimage import label, find_objects
    except ImportError:
        raise ImportError("scipy is required for bounding box extraction. Install with: pip install scipy")

    # Label connected components
    labeled_mask, num_features = label(mask)

    boxes = []
    # Find bounding boxes for each connected component
    for i in range(1, num_features + 1):
        slices = find_objects(labeled_mask == i)[0]

        # Extract coordinates
        z_start, y_start, x_start = slices[0].start, slices[1].start, slices[2].start
        z_stop, y_stop, x_stop = slices[0].stop, slices[1].stop, slices[2].stop

        # Calculate dimensions
        depth = z_stop - z_start
        height = y_stop - y_start
        width = x_stop - x_start

        box = {
            'z': z_start,
            'y': y_start,
            'x': x_start,
            'depth': depth,
            'height': height,
            'width': width
        }
        boxes.append(box)

    return boxes

def calculate_volume(mask: np.ndarray, voxel_size: Tuple[float, float, float]) -> float:
    """
    Calculate volume of detected anomaly
    
    Args:
        mask: Binary segmentation mask
        voxel_size: Size of each voxel in mm (depth, height, width)
        
    Returns:
        Volume in cubic millimeters
    """
    voxel_count = np.sum(mask > 0)
    volume = voxel_count * voxel_size[0] * voxel_size[1] * voxel_size[2]
    return volume

def filter_small_regions(mask: np.ndarray, min_size: int = 10) -> np.ndarray:
    """
    Remove small connected components from mask

    Args:
        mask: Binary segmentation mask
        min_size: Minimum number of voxels to keep

    Returns:
        Filtered mask
    """
    try:
        from scipy.ndimage import label
    except ImportError:
        raise ImportError("scipy is required for connected components filtering. Install with: pip install scipy")

    # Label connected components
    labeled_mask, num_features = label(mask)

    # Count voxels in each component
    component_sizes = np.bincount(labeled_mask.flat)

    # Create mask to keep only large enough components
    keep_mask = component_sizes >= min_size
    filtered_mask = keep_mask[labeled_mask].astype(np.uint8)

    return filtered_mask

# NOTE: calculate_confidence_score() was removed - it was never called and would
# incorrectly override model-generated probability/uncertainty values.
# Models now set their own probability and uncertainty ranges directly.

def format_prediction_result(raw_prediction: Dict[str, Any],
                            include_metadata: bool = True) -> Dict[str, Any]:
    """
    Format raw prediction into standardized result format
    
    Args:
        raw_prediction: Raw model output
        include_metadata: Whether to include metadata
        
    Returns:
        Formatted prediction dictionary
    """
    result = {
        'anomaly_detected': raw_prediction.get('anomaly_detected', False),
        'confidence': raw_prediction.get('confidence', 0.0),
        'location': raw_prediction.get('location', {}),
    }
    
    if include_metadata:
        result['metadata'] = {
            'model_type': raw_prediction.get('model_type', 'unknown'),
            'timestamp': raw_prediction.get('timestamp', None)
        }
    
    return result

