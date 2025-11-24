"""
Advanced Uncertainty Quantification for Medical Imaging
Based on state-of-the-art methods for CT scan classification

References:
- Gal & Ghahramani (2016): "Dropout as a Bayesian Approximation"
- Lakshminarayanan et al. (2017): "Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles"
- Kurz et al. (2022): "Uncertainty Estimation in Medical Image Classification: Systematic Review"
- Ovadia et al. (2019): "Can You Trust Your Model's Uncertainty?"
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, Any, Tuple, List, Union, Optional
from scipy import stats


class UncertaintyQuantifier:
    """
    Advanced uncertainty quantification for medical image classification.
    
    Implements multiple state-of-the-art methods:
    1. Monte Carlo Dropout (MCDO) - Bayesian approximation
    2. Test-Time Augmentation (TTA) - Aleatoric uncertainty
    3. Multiple uncertainty metrics - Comprehensive evaluation
    4. Temperature Scaling - Calibration
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        device: torch.device,
        mc_samples: int = 30,
        temperature: float = 1.0
    ):
        """
        Initialize uncertainty quantifier.
        
        Args:
            model: The neural network model
            device: torch device (CPU or CUDA)
            mc_samples: Number of Monte Carlo samples for MCDO (default: 30)
            temperature: Temperature parameter for calibration (default: 1.0)
        """
        self.model = model
        self.device = device
        self.mc_samples = mc_samples
        self.temperature = temperature
    
    def enable_dropout(self):
        """Enable dropout layers during inference for MC Dropout."""
        for module in self.model.modules():
            if isinstance(module, torch.nn.Dropout) or isinstance(module, torch.nn.Dropout2d):
                module.train()  # Enable dropout
    
    def mc_dropout_predict(
        self,
        input_tensor: torch.Tensor,
        n_samples: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform Monte Carlo Dropout prediction.
        
        Args:
            input_tensor: Input tensor (B, C, H, W)
            n_samples: Number of MC samples (uses self.mc_samples if None)
            
        Returns:
            mean_probs: Mean probabilities across samples (B, num_classes)
            all_probs: All probability samples (n_samples, B, num_classes)
        """
        if n_samples is None:
            n_samples = self.mc_samples
        
        # Store original training state
        was_training = self.model.training
        
        # Set model to eval mode but enable dropout
        self.model.eval()
        self.enable_dropout()
        
        all_probs = []
        
        with torch.no_grad():
            for _ in range(n_samples):
                logits = self.model(input_tensor)
                # Apply temperature scaling
                probs = F.softmax(logits / self.temperature, dim=1)
                all_probs.append(probs.cpu().numpy())
        
        # Restore original training state
        if was_training:
            self.model.train()
        else:
            self.model.eval()
        
        all_probs = np.array(all_probs)  # (n_samples, B, num_classes)
        mean_probs = all_probs.mean(axis=0)  # (B, num_classes)
        
        return mean_probs, all_probs
    
    def compute_uncertainty_metrics(
        self,
        all_probs: np.ndarray
    ) -> Dict[str, float]:
        """
        Compute comprehensive uncertainty metrics.
        
        Args:
            all_probs: All probability samples (n_samples, B, num_classes)
            
        Returns:
            Dictionary with multiple uncertainty metrics
        """
        # Average probabilities across samples
        mean_probs = all_probs.mean(axis=0)  # (B, num_classes)
        
        # 1. Predictive Entropy (Total Uncertainty)
        # H[y|x] = -sum(mean_p(y|x) * log(mean_p(y|x)))
        eps = 1e-10
        predictive_entropy = -np.sum(
            mean_probs * np.log(mean_probs + eps), axis=1
        )
        
        # 2. Expected Entropy (Aleatoric Uncertainty)
        # E_w[H[y|x,w]] = mean(-sum(p(y|x,w) * log(p(y|x,w))))
        sample_entropies = -np.sum(
            all_probs * np.log(all_probs + eps), axis=2
        )  # (n_samples, B)
        expected_entropy = sample_entropies.mean(axis=0)  # (B,)
        
        # 3. Mutual Information (Epistemic Uncertainty)
        # I[y;w|x] = H[y|x] - E_w[H[y|x,w]]
        mutual_information = predictive_entropy - expected_entropy
        
        # 4. Variation Ratio (1 - frequency of most common prediction)
        # Measures disagreement
        predicted_classes = all_probs.argmax(axis=2)  # (n_samples, B)
        mode_count = np.array([
            np.bincount(predicted_classes[:, i]).max()
            for i in range(predicted_classes.shape[1])
        ])
        variation_ratio = 1.0 - (mode_count / all_probs.shape[0])
        
        # 5. Standard Deviation of Predictions
        std_probs = all_probs.std(axis=0)  # (B, num_classes)
        mean_std = std_probs.mean(axis=1)  # (B,)
        
        # 6. Coefficient of Variation (normalized uncertainty)
        # CV = std / mean (where mean is non-zero)
        max_probs = mean_probs.max(axis=1)
        coefficient_of_variation = mean_std / (max_probs + eps)
        
        # Aggregate for batch (take mean across batch dimension)
        metrics = {
            "predictive_entropy": float(predictive_entropy.mean()),
            "expected_entropy": float(expected_entropy.mean()),
            "mutual_information": float(mutual_information.mean()),
            "variation_ratio": float(variation_ratio.mean()),
            "mean_std": float(mean_std.mean()),
            "coefficient_of_variation": float(coefficient_of_variation.mean()),
            "max_probability": float(max_probs.mean()),
        }
        
        # Normalize metrics to [0, 1] for interpretability
        num_classes = all_probs.shape[2]
        max_entropy = np.log(num_classes)
        
        metrics["normalized_entropy"] = metrics["predictive_entropy"] / max_entropy
        metrics["normalized_mutual_info"] = metrics["mutual_information"] / max_entropy
        
        # Overall uncertainty score (combination of metrics)
        # Weight epistemic (mutual_information) more as it's more concerning
        overall_uncertainty = (
            0.4 * metrics["normalized_mutual_info"] +
            0.3 * metrics["normalized_entropy"] +
            0.3 * metrics["variation_ratio"]
        )
        metrics["overall_uncertainty"] = float(overall_uncertainty)
        
        return metrics
    
    def test_time_augmentation_predict(
        self,
        input_tensor: torch.Tensor,
        augmentation_fn: callable,
        n_augmentations: int = 5
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform Test-Time Augmentation prediction.
        
        Args:
            input_tensor: Input tensor (B, C, H, W)
            augmentation_fn: Function to apply random augmentations
            n_augmentations: Number of augmented predictions
            
        Returns:
            mean_probs: Mean probabilities across augmentations
            all_probs: All probability samples
        """
        was_training = self.model.training
        self.model.eval()
        
        all_probs = []
        
        with torch.no_grad():
            # Original prediction
            logits = self.model(input_tensor)
            probs = F.softmax(logits / self.temperature, dim=1)
            all_probs.append(probs.cpu().numpy())
            
            # Augmented predictions
            for _ in range(n_augmentations - 1):
                aug_input = augmentation_fn(input_tensor)
                logits = self.model(aug_input)
                probs = F.softmax(logits / self.temperature, dim=1)
                all_probs.append(probs.cpu().numpy())
        
        if was_training:
            self.model.train()
        
        all_probs = np.array(all_probs)
        mean_probs = all_probs.mean(axis=0)
        
        return mean_probs, all_probs
    
    def combined_uncertainty(
        self,
        input_tensor: torch.Tensor,
        use_mcdo: bool = True,
        use_tta: bool = False,
        augmentation_fn: Optional[callable] = None,
        n_tta: int = 5
    ) -> Dict[str, Any]:
        """
        Combined uncertainty estimation using multiple methods.
        
        Args:
            input_tensor: Input tensor (B, C, H, W)
            use_mcdo: Use Monte Carlo Dropout
            use_tta: Use Test-Time Augmentation
            augmentation_fn: Augmentation function for TTA
            n_tta: Number of TTA samples
            
        Returns:
            Dictionary with predictions and uncertainty estimates
        """
        all_probs_list = []
        methods_used = []
        
        # Monte Carlo Dropout
        if use_mcdo:
            mean_probs_mcdo, all_probs_mcdo = self.mc_dropout_predict(input_tensor)
            all_probs_list.append(all_probs_mcdo)
            methods_used.append("mcdo")
        
        # Test-Time Augmentation
        if use_tta and augmentation_fn is not None:
            mean_probs_tta, all_probs_tta = self.test_time_augmentation_predict(
                input_tensor, augmentation_fn, n_tta
            )
            all_probs_list.append(all_probs_tta)
            methods_used.append("tta")
        
        # Combine all probability samples
        if len(all_probs_list) > 1:
            # Concatenate along sample dimension
            all_probs = np.concatenate(all_probs_list, axis=0)
        else:
            all_probs = all_probs_list[0]
        
        # Compute final predictions
        mean_probs = all_probs.mean(axis=0)  # (B, num_classes)
        predicted_class = mean_probs.argmax(axis=1)
        confidence = mean_probs.max(axis=1)
        
        # Compute comprehensive uncertainty metrics
        uncertainty_metrics = self.compute_uncertainty_metrics(all_probs)
        
        result = {
            "probabilities": mean_probs,
            "predicted_class": predicted_class,
            "confidence": confidence,
            "uncertainty_metrics": uncertainty_metrics,
            "methods_used": methods_used,
            "n_samples": all_probs.shape[0],
            "all_samples": all_probs,  # For advanced analysis
        }
        
        return result
    
    def should_refer_to_expert(
        self,
        uncertainty_metrics: Dict[str, float],
        confidence_threshold: float = 0.7,
        uncertainty_threshold: float = 0.3,
        mutual_info_threshold: float = 0.2
    ) -> Dict[str, Any]:
        """
        Determine if prediction should be referred to human expert.
        
        Args:
            uncertainty_metrics: Dictionary of uncertainty metrics
            confidence_threshold: Minimum confidence for automatic decision
            uncertainty_threshold: Maximum overall uncertainty for automatic decision
            mutual_info_threshold: Maximum epistemic uncertainty for automatic decision
            
        Returns:
            Dictionary with referral decision and reasoning
        """
        max_prob = uncertainty_metrics["max_probability"]
        overall_unc = uncertainty_metrics["overall_uncertainty"]
        mutual_info = uncertainty_metrics["normalized_mutual_info"]
        
        refer = False
        reasons = []
        
        # Low confidence
        if max_prob < confidence_threshold:
            refer = True
            reasons.append(f"Low confidence: {max_prob:.3f} < {confidence_threshold}")
        
        # High overall uncertainty
        if overall_unc > uncertainty_threshold:
            refer = True
            reasons.append(f"High uncertainty: {overall_unc:.3f} > {uncertainty_threshold}")
        
        # High epistemic uncertainty (model doesn't know)
        if mutual_info > mutual_info_threshold:
            refer = True
            reasons.append(f"High model uncertainty: {mutual_info:.3f} > {mutual_info_threshold}")
        
        return {
            "refer_to_expert": refer,
            "reasons": reasons,
            "confidence_score": float(max_prob),
            "uncertainty_score": float(overall_unc),
            "epistemic_uncertainty": float(mutual_info),
        }


def get_default_tta_augmentation():
    """
    Get default TTA augmentation function for CT scans.
    
    Returns:
        Augmentation function
    """
    import torchvision.transforms.functional as TF
    
    def augment(tensor: torch.Tensor) -> torch.Tensor:
        """Apply random augmentations suitable for CT scans."""
        # Random rotation (small angles)
        angle = np.random.uniform(-5, 5)
        tensor = TF.rotate(tensor, angle)
        
        # Random brightness/contrast (subtle)
        brightness_factor = np.random.uniform(0.95, 1.05)
        tensor = tensor * brightness_factor
        
        # Random noise (very subtle)
        noise = torch.randn_like(tensor) * 0.01
        tensor = tensor + noise
        
        return torch.clamp(tensor, 0, 1)
    
    return augment
