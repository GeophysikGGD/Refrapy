#!/usr/bin/env python3
"""
Test script to verify GPU integration in Refrapy.
This script tests the GPU utilities without needing the full GUI.
"""

import sys
import numpy as np
from gpu_utils import (
    is_cuda_available, 
    get_cuda_info, 
    gpu_config, 
    GPUArray, 
    accelerate_griddata,
    accelerate_matrix_operations
)

def test_cuda_detection():
    """Test CUDA availability detection."""
    print("=" * 60)
    print("Testing CUDA Detection")
    print("=" * 60)
    
    cuda_available = is_cuda_available()
    cuda_info = get_cuda_info()
    
    print(f"CUDA Available: {cuda_available}")
    print(f"CUDA Info: {cuda_info}")
    
    if cuda_available:
        device_info = gpu_config.get_device_info()
        print(f"Device Info: {device_info}")
    
    print("✅ CUDA detection test passed\n")
    return cuda_available


def test_gpu_config():
    """Test GPU configuration management."""
    print("=" * 60)
    print("Testing GPU Configuration")
    print("=" * 60)
    
    # Test initial state
    assert gpu_config.is_gpu_enabled() == False, "GPU should be disabled by default"
    print("✓ GPU disabled by default")
    
    # Test enable (will fail if CUDA not available)
    result = gpu_config.enable_gpu()
    if is_cuda_available():
        assert result == True, "GPU enable should succeed when CUDA is available"
        assert gpu_config.is_gpu_enabled() == True, "GPU should be enabled"
        print("✓ GPU enabled successfully")
    else:
        assert result == False, "GPU enable should fail when CUDA is not available"
        print("✓ GPU enable correctly fails without CUDA")
    
    # Test disable
    gpu_config.disable_gpu()
    assert gpu_config.is_gpu_enabled() == False, "GPU should be disabled"
    print("✓ GPU disabled successfully")
    
    print("✅ GPU configuration test passed\n")


def test_gpu_array():
    """Test GPUArray wrapper class."""
    print("=" * 60)
    print("Testing GPUArray")
    print("=" * 60)
    
    # Create test data
    test_data = np.random.rand(10, 10)
    
    # Test CPU mode
    gpu_arr_cpu = GPUArray(test_data, use_gpu=False)
    result_cpu = gpu_arr_cpu.to_cpu()
    assert np.allclose(result_cpu, test_data), "CPU mode should preserve data"
    print("✓ GPUArray CPU mode works")
    
    # Test GPU mode (if available)
    if is_cuda_available():
        gpu_arr_gpu = GPUArray(test_data, use_gpu=True)
        result_gpu = gpu_arr_gpu.to_cpu()
        assert np.allclose(result_gpu, test_data), "GPU mode should preserve data"
        print("✓ GPUArray GPU mode works")
    else:
        print("⊘ GPU mode skipped (CUDA not available)")
    
    print("✅ GPUArray test passed\n")


def test_griddata_acceleration():
    """Test GPU-accelerated griddata interpolation."""
    print("=" * 60)
    print("Testing Griddata Acceleration")
    print("=" * 60)
    
    # Create test data (regular grid)
    x = np.linspace(0, 10, 20)
    y = np.linspace(0, 10, 20)
    X, Y = np.meshgrid(x, y)
    points = np.column_stack([X.ravel(), Y.ravel()])
    values = np.sin(X.ravel()) + np.cos(Y.ravel())
    
    # Interpolation grid
    xi_x, xi_y = np.meshgrid(np.linspace(0, 10, 50), np.linspace(0, 10, 50))
    
    # Test CPU mode
    result_cpu = accelerate_griddata(points, values, (xi_x, xi_y), 
                                     method='linear', use_gpu=False)
    assert result_cpu is not None, "CPU griddata should return result"
    assert not np.all(np.isnan(result_cpu)), "Result should contain valid data"
    print("✓ CPU griddata works")
    
    # Test GPU mode (if available)
    if is_cuda_available():
        result_gpu = accelerate_griddata(points, values, (xi_x, xi_y), 
                                         method='linear', use_gpu=True)
        # Note: GPU griddata currently falls back to CPU
        assert result_gpu is not None, "GPU griddata should return result"
        print("✓ GPU griddata works (currently uses CPU fallback)")
    else:
        print("⊘ GPU griddata skipped (CUDA not available)")
    
    print("✅ Griddata acceleration test passed\n")


def test_matrix_operations():
    """Test GPU-accelerated matrix operations."""
    print("=" * 60)
    print("Testing Matrix Operations")
    print("=" * 60)
    
    # Create test matrix
    matrix = np.random.rand(50, 50)
    matrix = matrix @ matrix.T  # Make it symmetric positive definite
    
    # Test CPU inversion
    result_cpu = accelerate_matrix_operations(matrix, operation='inv', use_gpu=False)
    assert result_cpu is not None, "CPU matrix inversion should return result"
    # Verify it's actually an inverse
    identity = matrix @ result_cpu
    assert np.allclose(identity, np.eye(50), atol=1e-10), "Should produce identity matrix"
    print("✓ CPU matrix inversion works")
    
    # Test GPU inversion (if available)
    if is_cuda_available():
        result_gpu = accelerate_matrix_operations(matrix, operation='inv', use_gpu=True)
        assert result_gpu is not None, "GPU matrix inversion should return result"
        identity_gpu = matrix @ result_gpu
        assert np.allclose(identity_gpu, np.eye(50), atol=1e-10), "Should produce identity matrix"
        print("✓ GPU matrix inversion works")
    else:
        print("⊘ GPU matrix operations skipped (CUDA not available)")
    
    print("✅ Matrix operations test passed\n")


def run_all_tests():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("Refrapy GPU Integration Test Suite")
    print("=" * 60 + "\n")
    
    cuda_available = test_cuda_detection()
    test_gpu_config()
    test_gpu_array()
    test_griddata_acceleration()
    test_matrix_operations()
    
    print("=" * 60)
    print("All Tests Passed!")
    print("=" * 60)
    
    if cuda_available:
        print("\n✅ CUDA is available - GPU acceleration is ready to use!")
    else:
        print("\n⚠️  CUDA is not available - running in CPU-only mode")
        print("   To enable GPU acceleration:")
        print("   1. Install CUDA Toolkit")
        print("   2. Install CuPy: pip install cupy-cuda11x")
        print("   (See GPU_SETUP.md for details)")


if __name__ == "__main__":
    try:
        run_all_tests()
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
