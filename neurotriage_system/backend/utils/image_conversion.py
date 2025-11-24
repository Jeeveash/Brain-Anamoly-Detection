"""
Image conversion utilities for medical imaging
Handles conversion between JPG/PNG and NIfTI formats
"""

import numpy as np
import nibabel as nib
from PIL import Image
import os
from typing import List, Union
import cv2


def jpg_slices_to_nifti(slice_paths: List[str], output_path: str, spacing: tuple = (1.0, 1.0, 1.0)) -> str:
    """
    Convert a list of JPG/PNG slices to a 3D NIfTI volume
    
    Args:
        slice_paths: List of paths to 2D image slices (sorted by slice order)
        output_path: Output path for NIfTI file (.nii or .nii.gz)
        spacing: Voxel spacing in (x, y, z) - default 1mm isotropic
        
    Returns:
        Path to created NIfTI file
    """
    if not slice_paths:
        raise ValueError("No slice paths provided")
    
    # Load all slices
    slices = []
    for path in slice_paths:
        img = Image.open(path).convert('L')  # Convert to grayscale
        slices.append(np.array(img))
    
    # Stack into 3D volume (H, W, D)
    volume = np.stack(slices, axis=-1)
    
    # Create affine matrix with spacing
    affine = np.diag([spacing[0], spacing[1], spacing[2], 1.0])
    
    # Create NIfTI image
    nifti_img = nib.Nifti1Image(volume, affine)
    
    # Save
    nib.save(nifti_img, output_path)
    print(f"[INFO] Created NIfTI volume: {volume.shape} -> {output_path}")
    
    return output_path


def jpg_to_nifti_single(jpg_path: str, output_path: str = None) -> str:
    """
    Convert single JPG to NIfTI (for 2D or single slice)
    
    Args:
        jpg_path: Path to JPG image
        output_path: Output NIfTI path (optional, auto-generated if None)
        
    Returns:
        Path to created NIfTI file
    """
    if output_path is None:
        base = os.path.splitext(jpg_path)[0]
        output_path = f"{base}.nii.gz"
    
    # Load image
    img = Image.open(jpg_path).convert('L')
    data = np.array(img)
    
    # Add depth dimension for 3D format (H, W, 1)
    data = data[:, :, np.newaxis]
    
    # Create NIfTI with identity affine
    nifti_img = nib.Nifti1Image(data, np.eye(4))
    nib.save(nifti_img, output_path)
    
    return output_path


def normalize_intensity(image: np.ndarray, percentile_range: tuple = (1, 99)) -> np.ndarray:
    """
    Normalize image intensity to 0-255 range using percentile clipping
    
    Args:
        image: Input image array
        percentile_range: Tuple of (low, high) percentiles for clipping
        
    Returns:
        Normalized image in 0-255 range
    """
    p_low, p_high = np.percentile(image, percentile_range)
    image_clipped = np.clip(image, p_low, p_high)
    image_normalized = ((image_clipped - p_low) / (p_high - p_low) * 255).astype(np.uint8)
    return image_normalized


def multi_modal_jpgs_to_nifti(
    modality_paths: dict, 
    output_dir: str,
    modality_names: List[str] = None
) -> List[str]:
    """
    Convert multiple modalities (each with multiple slices) to NIfTI volumes
    
    Args:
        modality_paths: Dict mapping modality name to list of slice paths
                       e.g., {'t1': [slice1.jpg, ...], 't1ce': [...], ...}
        output_dir: Directory to save NIfTI files
        modality_names: Optional ordered list of modality names
        
    Returns:
        List of paths to created NIfTI files (in order of modality_names)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    output_paths = []
    modalities = modality_names if modality_names else list(modality_paths.keys())
    
    for modality in modalities:
        if modality not in modality_paths:
            raise ValueError(f"Modality '{modality}' not found in provided paths")
        
        slice_paths = modality_paths[modality]
        output_path = os.path.join(output_dir, f"{modality}.nii.gz")
        
        # Convert to NIfTI
        jpg_slices_to_nifti(slice_paths, output_path)
        output_paths.append(output_path)
    
    return output_paths


def resize_volume(volume: np.ndarray, target_shape: tuple, interpolation=cv2.INTER_LINEAR) -> np.ndarray:
    """
    Resize 3D volume to target shape
    
    Args:
        volume: 3D numpy array (H, W, D)
        target_shape: Target shape (H, W, D)
        interpolation: OpenCV interpolation method
        
    Returns:
        Resized volume
    """
    current_shape = volume.shape
    
    if current_shape == target_shape:
        return volume
    
    # Resize in-plane (H, W)
    if current_shape[:2] != target_shape[:2]:
        resized_slices = []
        for i in range(current_shape[2]):
            resized_slice = cv2.resize(
                volume[:, :, i], 
                (target_shape[1], target_shape[0]),
                interpolation=interpolation
            )
            resized_slices.append(resized_slice)
        volume = np.stack(resized_slices, axis=-1)
    
    # Resize along depth if needed
    if volume.shape[2] != target_shape[2]:
        # Use linear interpolation along z-axis
        z_old = np.arange(volume.shape[2])
        z_new = np.linspace(0, volume.shape[2] - 1, target_shape[2])
        
        resized_volume = np.zeros(target_shape, dtype=volume.dtype)
        for i in range(target_shape[0]):
            for j in range(target_shape[1]):
                resized_volume[i, j, :] = np.interp(z_new, z_old, volume[i, j, :])
        
        volume = resized_volume
    
    return volume


def create_mock_multimodal_from_single(
    single_modality_path: str,
    output_dir: str,
    num_modalities: int = 4
) -> List[str]:
    """
    Create mock multi-modal MRI from single modality by applying different intensity transforms
    Useful when only one modality is available but model expects 4
    
    Args:
        single_modality_path: Path to single modality NIfTI or list of JPGs
        output_dir: Output directory for generated modalities
        num_modalities: Number of modalities to generate (default 4 for BraTS)
        
    Returns:
        List of paths to generated NIfTI files
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Load base volume
    if single_modality_path.endswith('.nii') or single_modality_path.endswith('.nii.gz'):
        nifti = nib.load(single_modality_path)
        base_volume = nifti.get_fdata()
        affine = nifti.affine
    else:
        # Assume JPG - load as single slice volume
        img = Image.open(single_modality_path).convert('L')
        base_volume = np.array(img)[:, :, np.newaxis]
        affine = np.eye(4)
    
    # Normalize to 0-1
    base_volume = (base_volume - base_volume.min()) / (base_volume.max() - base_volume.min() + 1e-8)
    
    output_paths = []
    modality_names = ['t1', 't1ce', 't2', 'flair']
    
    # Apply different transforms to simulate modalities
    transforms = [
        lambda x: x,  # T1 - original
        lambda x: x * 1.2,  # T1CE - enhanced
        lambda x: 1.0 - x * 0.5,  # T2 - inverted contrast
        lambda x: x * 0.8 + 0.2,  # FLAIR - brightened
    ]
    
    for i in range(min(num_modalities, len(modality_names))):
        # Apply transform
        modality_volume = transforms[i](base_volume)
        modality_volume = np.clip(modality_volume * 255, 0, 255).astype(np.uint8)
        
        # Save as NIfTI
        output_path = os.path.join(output_dir, f"{modality_names[i]}.nii.gz")
        nifti_img = nib.Nifti1Image(modality_volume, affine)
        nib.save(nifti_img, output_path)
        
        output_paths.append(output_path)
        print(f"[INFO] Generated {modality_names[i]} -> {output_path}")
    
    return output_paths
