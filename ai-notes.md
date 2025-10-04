# AI Development Environment Setup Notes

This document summarizes the key concepts and steps taken to set up a local AI development environment with full GPU acceleration on Linux (WSL) using Python, PyTorch, and CUDA. Each section includes background explanations, commands, and testing methods.

## Table of Contents
1. [Virtual Environments (`venv`)](#1-virtual-environments-venv)
2. [Package Management: `pip`, `setuptools`, `wheel`](#2-package-management-pip-setuptools-wheel)
3. [PyTorch](#3-pytorch)
4. [CUDA and GPU Compatibility](#4-cuda-and-gpu-compatibility)
5. [Testing PyTorch GPU Support](#5-testing-pytorch-gpu-support)
6. [Version Checks and Utilities](#6-version-checks-and-utilities)
7. [TorchVision and TorchAudio](#7-torchvision-and-torchaudio)
8. [Notes on Nightly Builds](#8-notes-on-nightly-builds)
9. [Summary of Key Learnings](#9-summary-of-key-learnings)
10. [End-to-End Setup Blueprint](#10-end-to-end-setup-blueprint)

---

## 1. Virtual Environments (`venv`)

**Purpose:**
A *virtual environment* is an isolated Python workspace that keeps your project dependencies separate from the system-wide Python installation.

**Why it matters:**
- Prevents version conflicts between projects.
- Keeps your global Python installation clean.
- Makes project sharing and reproducibility easier.

**Commands:**
```bash
# Create a new directory for your AI work
mkdir -p ~/projects/ai-lab
cd ~/projects/ai-lab

# Create a new virtual environment
python3 -m venv .venv

# Activate the environment
source .venv/bin/activate

# Check which Python interpreter is used
which python
```
Expected output:
```
/home/<user>/projects/ai-lab/.venv/bin/python
```
This confirms you are inside the virtual environment (notice the `.venv` prefix in your shell prompt).

---

## 2. Package Management: `pip`, `setuptools`, `wheel`

**Purpose:**
These are the essential tools for Python package installation.
- **pip**: Installs Python packages from PyPI or custom repositories.
- **setuptools**: Handles building and installing Python projects.
- **wheel**: Supports the binary package format `.whl`, which installs faster than source builds.

**Command to upgrade them:**
```bash
python -m pip install --upgrade pip setuptools wheel
```
Expected output:
```
Successfully installed pip-25.x setuptools-80.x wheel-0.45.x
```
This ensures compatibility with the latest library versions.

---

## 3. PyTorch

**Purpose:**
[PyTorch](https://pytorch.org/) is a machine learning framework widely used for deep learning and AI model development. It supports both CPU and GPU computations.

**Installation (with CUDA support):**
To install PyTorch with GPU acceleration, use the following command:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```
- `torch`: Core deep learning library.
- `torchvision`: Image-related utilities (datasets, transformations, pretrained models).
- `torchaudio`: Audio-related utilities and datasets.
- `cu128`: Specifies CUDA 12.8, matching your NVIDIA driver and GPU architecture.

**Nightly Builds:**
If your GPU or CUDA version is newer than the stable PyTorch release, you can install a *nightly build* (development version):
```bash
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
```
Nightly builds are often necessary to support new GPUs (like the RTX 5080).

---

## 4. CUDA and GPU Compatibility

**CUDA** (Compute Unified Device Architecture) is NVIDIA’s framework that allows software (like PyTorch) to use GPUs for computation.

- Each GPU architecture has a unique **SM (Streaming Multiprocessor)** version (e.g., `sm_90`, `sm_120`).
- CUDA and PyTorch must support your GPU’s SM version.

**SM Compatibility Table (simplified):**
| Architecture | GPU Series         | SM Version |
|---------------|--------------------|-------------|
| Volta         | Tesla V100         | sm_70       |
| Turing        | RTX 20xx           | sm_75       |
| Ampere        | RTX 30xx           | sm_80       |
| Ada Lovelace  | RTX 40xx           | sm_89/90    |
| Blackwell     | RTX 50xx           | sm_120      |

If your GPU shows as *unsupported*, upgrade to a newer CUDA toolkit or a nightly PyTorch build.

---

## 5. Testing PyTorch GPU Support

You can verify GPU detection and CUDA functionality with this script:

```bash
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("arch list:", torch.cuda.get_arch_list())
    print("device name:", torch.cuda.get_device_name(0))
    x = torch.randn(256, 256, device="cuda")
    y = torch.matmul(x, x)
    print("successful multiply", y.shape)
PY
```

**Expected Output:**
```
torch: 2.1.0.dev20251002+cu128
cuda available: True
arch list: ['sm_70', 'sm_75', 'sm_80', 'sm_86', 'sm_90', 'sm_100', 'sm_120']
device name: NVIDIA GeForce RTX 5080
successful multiply torch.Size([256, 256])
```
This confirms:
- PyTorch is running with CUDA 12.8.
- GPU recognized as RTX 5080.
- The test computation successfully ran on the GPU.

---

## 6. Version Checks and Utilities

**Check CUDA Toolkit Version:**
```bash
nvcc --version
```

**Check NVIDIA Driver & GPU Info:**
```bash
nvidia-smi
```

**Check PyTorch CUDA Version:**
```bash
python -c "import torch; print(torch.version.cuda)"
```

**Check PyTorch GPU Detection:**
```bash
python -c "import torch; print(torch.cuda.get_device_name(0))"
```

**Force reinstall latest nightly build:**
```bash
pip install --upgrade --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
```

---

## 7. TorchVision and TorchAudio

- **torchvision** adds prebuilt datasets (e.g., CIFAR, ImageNet), image transformations, and pretrained CNN models.
- **torchaudio** provides similar functionality for audio datasets and signal processing.

Example test:
```bash
python - <<'PY'
import torchvision, torchaudio
print("torchvision:", torchvision.__version__)
print("torchaudio:", torchaudio.__version__)
PY
```
Expected output:
```
torchvision: 0.20.x
torchaudio: 2.2.x
```

---

## 8. Notes on Nightly Builds

- Nightly versions (`--pre`) are built automatically from the latest commits.
- They may include early support for new GPUs or CUDA toolkits.
- Slight instability is possible, but they’re ideal for testing and cutting-edge hardware.

To downgrade to a stable version later:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128 --force-reinstall
```

---

## 9. Summary of Key Learnings

| Concept | Description |
|----------|--------------|
| **venv** | Isolated Python environment for dependencies |
| **pip / setuptools / wheel** | Core Python packaging tools |
| **PyTorch** | Machine learning framework supporting GPU acceleration |
| **CUDA** | NVIDIA's framework to run GPU computations |
| **Nightly Builds** | Developer builds with cutting-edge features |
| **SM (Streaming Multiprocessor)** | GPU architecture version identifier |
| **torchvision / torchaudio** | PyTorch extensions for image and audio processing |

---

## 10. End-to-End Setup Blueprint

This chapter condenses the essential actions from Sections 1–9 into a practical workflow that can be followed from a clean system to a validated GPU-enabled PyTorch environment.

1. **Prepare the workspace** by creating a project directory and activating a Python virtual environment to isolate dependencies and guarantee reproducibility.
2. **Upgrade foundational packaging tools** (`pip`, `setuptools`, and `wheel`) to ensure compatibility with the latest binary distributions and to speed up installations.
3. **Install PyTorch with matching CUDA support**, optionally selecting the nightly channel when targeting cutting-edge GPUs or driver/toolkit combinations.
4. **Verify CUDA and driver alignment** by checking GPU architecture support tables, confirming the installed CUDA toolkit, and reviewing driver status with `nvidia-smi`.
5. **Run GPU functionality tests** in PyTorch to confirm device discovery, architecture availability, and successful execution of tensor operations on the GPU.
6. **Validate auxiliary libraries** such as TorchVision and TorchAudio to ensure that vision and audio tooling versions align with the installed PyTorch build.
7. **Document key learnings**—including SM mappings, packaging utilities, and nightly build considerations—to retain institutional knowledge for future setup iterations.

Use this blueprint whenever you need a succinct, end-to-end reminder of the environment setup process; each bullet links conceptually back to the detailed chapters above.

---

*Document generated to summarize AI environment setup for PyTorch + CUDA 12.8 with RTX 5080.*
