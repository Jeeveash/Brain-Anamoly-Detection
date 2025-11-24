"""
Test script for nnU-Net tumor model integration
"""

import os
import sys
import torch
from pathlib import Path

def test_nnunet_import():
    """Test if nnU-Net can be imported"""
    print("\n=== Testing nnU-Net Import ===")
    try:
        from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
        print("✓ nnU-Net import successful")
        return True
    except ImportError as e:
        print(f"✗ nnU-Net import failed: {e}")
        return False

def test_model_weights_exist():
    """Test if model weights are present"""
    print("\n=== Testing Model Weights ===")
    
    weights_path = Path("../../nnU-Net Model Weights/Dataset002_BRATS19/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0")
    weights_path = weights_path.resolve()
    
    print(f"Looking for weights at: {weights_path}")
    
    if not weights_path.exists():
        print(f"✗ Model weights folder not found")
        return False
    
    checkpoint = weights_path / "checkpoint_final.pth"
    if not checkpoint.exists():
        print(f"✗ Checkpoint file not found: {checkpoint}")
        return False
    
    print(f"✓ Model weights found")
    print(f"  - Checkpoint size: {checkpoint.stat().st_size / (1024**2):.1f} MB")
    return True

def test_model_loading():
    """Test if model can be loaded"""
    print("\n=== Testing Model Loading ===")
    
    try:
        from models.nnunet_tumor_model import nnUNetTumorModel
        
        model = nnUNetTumorModel()
        print(f"✓ Model class instantiated")
        print(f"  - Device: {model.device}")
        print(f"  - Classes: {model.num_classes}")
        
        # Try to load model
        print("\nAttempting to load model weights...")
        model.load_model()
        print("✓ Model loaded successfully!")
        
        return True
        
    except Exception as e:
        print(f"✗ Model loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_model_registry():
    """Test if model is registered in app"""
    print("\n=== Testing Model Registry ===")
    
    try:
        # Import after adding backend to path
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from app import MODEL_REGISTRY
        
        print(f"✓ Registry loaded with {len(MODEL_REGISTRY)} models:")
        for name in MODEL_REGISTRY.keys():
            print(f"  - {name}")
        
        if "tumor_nnunet" in MODEL_REGISTRY:
            print("\n✓ nnU-Net model found in registry")
            return True
        else:
            print("\n✗ nnU-Net model NOT in registry")
            return False
            
    except Exception as e:
        print(f"✗ Registry test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_cuda_availability():
    """Test CUDA availability"""
    print("\n=== Testing CUDA ===")
    
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        print(f"✓ CUDA is available")
        print(f"  - Device count: {torch.cuda.device_count()}")
        print(f"  - Device name: {torch.cuda.get_device_name(0)}")
        print(f"  - CUDA version: {torch.version.cuda}")
    else:
        print("⚠ CUDA not available - will use CPU")
        print("  (This is fine but inference will be slower)")
    
    return True

def main():
    """Run all tests"""
    print("=" * 60)
    print("nnU-Net Integration Test Suite")
    print("=" * 60)
    
    results = []
    
    # Test 1: CUDA
    results.append(("CUDA", test_cuda_availability()))
    
    # Test 2: Import
    results.append(("Import", test_nnunet_import()))
    
    # Test 3: Weights
    results.append(("Weights", test_model_weights_exist()))
    
    # Test 4: Loading (only if import and weights passed)
    if results[1][1] and results[2][1]:
        results.append(("Loading", test_model_loading()))
    else:
        print("\n⊗ Skipping model loading test (prerequisites failed)")
        results.append(("Loading", False))
    
    # Test 5: Registry
    results.append(("Registry", test_model_registry()))
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{test_name:20s} {status}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! nnU-Net is ready to use.")
        return 0
    else:
        print(f"\n⚠ {total - passed} test(s) failed. Check errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
