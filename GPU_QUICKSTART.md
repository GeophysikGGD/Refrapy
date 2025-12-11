# GPU Quick Start Guide

## 5-Minute Setup for GPU Acceleration

### Prerequisites Check
1. You have an NVIDIA GPU
2. You know your CUDA version (run `nvcc --version` or `nvidia-smi`)

### Installation

#### Step 1: Install CuPy (2 minutes)
```bash
# Check your CUDA version first
nvidia-smi

# Install matching CuPy version:
pip install cupy-cuda11x  # For CUDA 11.x
# OR
pip install cupy-cuda12x  # For CUDA 12.x
```

#### Step 2: Verify Installation (1 minute)
```bash
cd /path/to/Refrapy
python test_gpu_integration.py
```

If you see "✅ CUDA is available", you're good to go!

#### Step 3: Enable in Refrapy (30 seconds)
1. Launch Refrainv
2. Go to `Settings → GPU Settings`
3. Click `Enable GPU`
4. Done! 🚀

### Testing GPU Performance

Run a quick inversion and check the speed improvement. GPU typically provides 1.5-3x speedup for large datasets.

### Troubleshooting

**"CUDA not available"**
- Install CUDA Toolkit from: https://developer.nvidia.com/cuda-downloads
- Reboot after installation

**"CuPy import error"**
- Make sure CuPy version matches CUDA version
- Try: `pip uninstall cupy && pip install cupy-cuda11x`

**"Out of memory"**
- Reduce mesh resolution
- Close other GPU applications
- Or just use CPU mode (still works!)

For detailed setup and troubleshooting, see [GPU_SETUP.md](GPU_SETUP.md).
