#!/usr/bin/env python3
"""
Quick test script to verify PhaseNet installation and setup
"""
import sys
import torch
import numpy as np

def check_pytorch():
    """Check PyTorch installation"""
    print("=" * 60)
    print("PyTorch Installation Check")
    print("=" * 60)
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("  Note: Training will use CPU (slower)")
    
    # Check MPS (Apple Silicon)
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        print("MPS (Apple Silicon GPU) available: True")
    print()


def check_seisbench():
    """Check SeisBench installation"""
    print("=" * 60)
    print("SeisBench Installation Check")
    print("=" * 60)
    try:
        import seisbench
        print(f"SeisBench version: {seisbench.__version__}")
        
        # Try to list available models
        import seisbench.models as sbm
        print("\nAvailable PhaseNet models:")
        models = sbm.PhaseNet.list_pretrained()
        for model in models[:5]:  # Show first 5
            print(f"  - {model}")
        if len(models) > 5:
            print(f"  ... and {len(models) - 5} more")
        
        print("\n✓ SeisBench is properly installed")
    except ImportError as e:
        print(f"✗ SeisBench not found: {e}")
        print("  Install with: pip install seisbench")
    print()


def test_model_loading():
    """Test loading a pretrained model"""
    print("=" * 60)
    print("Model Loading Test")
    print("=" * 60)
    try:
        import seisbench.models as sbm
        print("Loading pretrained PhaseNet model (stead)...")
        model = sbm.PhaseNet.from_pretrained('stead')
        print(f"✓ Model loaded successfully")
        print(f"  Input channels: {model.in_channels}")
        print(f"  Output classes: {model.classes}")
        print(f"  Phases: {model.labels}")
        print(f"  Sampling rate: {model.sampling_rate} Hz")
        
        # Test forward pass
        print("\nTesting forward pass...")
        x = torch.randn(2, 3, 3001)  # batch=2, channels=3, samples=3001
        model.eval()
        with torch.no_grad():
            y = model(x)
        print(f"✓ Forward pass successful")
        print(f"  Input shape: {x.shape}")
        print(f"  Output shape: {y.shape}")
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        print(f"  Total parameters: {total_params:,}")
        
    except Exception as e:
        print(f"✗ Model loading failed: {e}")
    print()


def check_dependencies():
    """Check other dependencies"""
    print("=" * 60)
    print("Other Dependencies Check")
    print("=" * 60)
    
    dependencies = [
        ('numpy', 'NumPy'),
        ('pandas', 'Pandas'),
        ('matplotlib', 'Matplotlib'),
        ('obspy', 'ObsPy'),
        ('yaml', 'PyYAML'),
        ('h5py', 'h5py'),
        ('scipy', 'SciPy'),
    ]
    
    missing = []
    for module_name, display_name in dependencies:
        try:
            module = __import__(module_name)
            version = getattr(module, '__version__', 'unknown')
            print(f"✓ {display_name}: {version}")
        except ImportError:
            print(f"✗ {display_name}: Not installed")
            missing.append(display_name)
    
    if missing:
        print(f"\nMissing packages: {', '.join(missing)}")
        print("Install with: pip install -r requirements.txt")
    else:
        print("\n✓ All dependencies are installed")
    print()


def main():
    """Run all checks"""
    print("\n" + "=" * 60)
    print("PhaseNet Retraining Framework - Installation Check")
    print("=" * 60)
    print()
    
    check_pytorch()
    check_seisbench()
    check_dependencies()
    test_model_loading()

    print("=" * 60)
    print("Setup Check Complete!")
    print("=" * 60)
    print("\nNext steps (real pipeline, not the pytorch_lightning path):")
    print("1. Build training manifests: python scripts/build_training_dataset.py")
    print("2. Pick/edit a config in configs/finetune_jma_wc_global_v*.yaml")
    print("3. Run training: python scripts/finetune.py --config <config.yaml>")
    print("\nFor benchmark evaluation, see: notebooks/06_step_3_inference_evaluation.ipynb")
    print()


if __name__ == "__main__":
    main()
