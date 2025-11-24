"""
Base model interface for brain anomaly detection models
Common interface that all model implementations should follow
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Union, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    try:
        import torch
    except ImportError:
        torch = None


class BaseMedicalModel(ABC):
    """
    Abstract base class for medical imaging models
    All medical models (TumorModel, HemorrhageModel, etc.) inherit from this class
    """
    
    def __init__(self, model_name: str):
        """
        Initialize the medical model
        
        Args:
            model_name: Name identifier for the model (e.g., 'tumor_detector', 'hemorrhage_detector')
        """
        self.model_name = model_name
        self.model = None
        self.model_loaded = False
    
    @abstractmethod
    def load_model(self) -> None:
        """
        Load the model from storage
        Subclasses should implement model loading logic based on model_name
        """
        pass
    
    @abstractmethod
    def preprocess(self, image_path: str) -> np.ndarray:
        """
        Preprocess medical image from file path
        
        Args:
            image_path: Path to the medical imaging file (DICOM, NIfTI, etc.)
            
        Returns:
            Preprocessed image as numpy array or tensor ready for inference
        """
        pass
    
    @abstractmethod
    def predict(self, input_tensor: Union[np.ndarray, Any]) -> Dict[str, Any]:
        """
        Perform prediction on input tensor
        
        Args:
            input_tensor: Preprocessed input tensor (numpy array or torch tensor)
            
        Returns:
            Dictionary containing:
            - 'probability': float - Probability of anomaly detection (0-1)
            - 'mask': np.ndarray - Segmentation mask or bounding box coordinates
            - 'uncertainty': float - Model uncertainty/confidence measure
        """
        pass
    
    @abstractmethod
    def postprocess(self, prediction: Dict[str, Any]) -> Dict[str, Any]:
        """
        Postprocess raw model predictions into interpretable results
        
        Args:
            prediction: Raw prediction dictionary from predict() method
            
        Returns:
            Postprocessed prediction dictionary with cleaned/filtered results
        """
        pass
    
    @abstractmethod
    def explain(self, input_tensor: Union[np.ndarray, Any]) -> np.ndarray:
        """
        Generate explanation (heatmap or attention mask) for the prediction
        
        Args:
            input_tensor: Preprocessed input tensor (numpy array or torch tensor)
            
        Returns:
            Explanation as numpy array:
            - Heatmap: Array with same spatial dimensions as input showing attention/importance
            - Mask: Binary or weighted mask highlighting important regions
        """
        pass
    
    def get_uncertainty(
        self, 
        input_tensor: Union[np.ndarray, Any],
        uncertainty_threshold: float = 0.25,
        method: str = "auto"
    ) -> Dict[str, Any]:
        """
        Compute prediction uncertainty and flag cases for human review
        
        For classification models: uses Monte Carlo dropout or ensemble averaging
        For segmentation models: computes entropy across softmax probabilities per voxel
        
        Args:
            input_tensor: Preprocessed input tensor
            uncertainty_threshold: Threshold above which to flag for human review (default: 0.25)
            method: Uncertainty method ('auto', 'mc_dropout', 'ensemble', 'entropy')
                   'auto' selects method based on model type
            
        Returns:
            Dictionary containing:
            - 'uncertainty': float or np.ndarray - Uncertainty measure(s)
            - 'mean_uncertainty': float - Mean uncertainty value
            - 'requires_review': bool - True if uncertainty > threshold
            - 'uncertainty_map': np.ndarray (optional) - Per-voxel uncertainty map for segmentation
        """
        # Default implementation - subclasses should override
        # This provides a fallback that uses basic entropy
        try:
            prediction = self.predict(input_tensor)
            uncertainty = prediction.get("uncertainty", 0.0)
            
            return {
                "uncertainty": uncertainty,
                "mean_uncertainty": float(uncertainty),
                "requires_review": uncertainty > uncertainty_threshold,
                "uncertainty_map": None,
            }
        except Exception as e:
            raise RuntimeError(f"Error computing uncertainty: {str(e)}")


# Legacy support: Keep BaseAnomalyModel for backward compatibility
class BaseAnomalyModel(ABC):
    """
    Abstract base class for brain anomaly detection models
    DEPRECATED: Use BaseMedicalModel instead
    All models (tumor, hemorrhage, etc.) should inherit from BaseMedicalModel
    """
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize the model
        
        Args:
            model_path: Path to the trained model file
        """
        self.model_path = model_path
        self.model = None
    
    @abstractmethod
    def load_model(self, model_path: str) -> None:
        """
        Load the model from the specified path
        
        Args:
            model_path: Path to the model file
        """
        pass
    
    @abstractmethod
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess the input image for model inference
        
        Args:
            image: Input brain scan image
            
        Returns:
            Preprocessed image ready for inference
        """
        pass
    
    @abstractmethod
    def predict(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Perform prediction on the preprocessed image
        
        Args:
            image: Preprocessed brain scan image
            
        Returns:
            Dictionary containing prediction results with keys:
            - 'anomaly_detected': bool
            - 'confidence': float
            - 'location': dict with bounding box or segmentation mask
            - 'severity': str (optional)
        """
        pass
    
    @abstractmethod
    def postprocess(self, prediction: Dict[str, Any]) -> Dict[str, Any]:
        """
        Postprocess raw model predictions into interpretable results
        
        Args:
            prediction: Raw model output
            
        Returns:
            Postprocessed prediction dictionary
        """
        pass

