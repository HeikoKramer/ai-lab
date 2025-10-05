# AI Development Environment Setup Notes

This document summarizes the key concepts and steps taken to set up a local AI development environment with full GPU acceleration on Linux (WSL) using Python, PyTorch, and CUDA. Each section includes background explanations, commands, and testing methods.

## Table of Contents
1. [AI Environment Setup for PyTorch + CUDA 12.8 (RTX 5080)](#ai-environment-setup-for-pytorch--cuda-128-rtx-5080)
   - [1. Virtual Environments (`venv`)](#1-virtual-environments-venv)
   - [2. Package Management: `pip`, `setuptools`, `wheel`](#2-package-management-pip-setuptools-wheel)
   - [3. PyTorch](#3-pytorch)
   - [4. CUDA and GPU Compatibility](#4-cuda-and-gpu-compatibility)
   - [5. Testing PyTorch GPU Support](#5-testing-pytorch-gpu-support)
   - [6. Version Checks and Utilities](#6-version-checks-and-utilities)
   - [7. TorchVision and TorchAudio](#7-torchvision-and-torchaudio)
   - [8. Notes on Nightly Builds](#8-notes-on-nightly-builds)
   - [9. Summary of Key Learnings](#9-summary-of-key-learnings)
   - [10. End-to-End Setup Blueprint](#10-end-to-end-setup-blueprint)
2. [Working with Hugging Face](#working-with-hugging-face)
   - [Models](#models)
     - [1. Transformers Overview](#1-transformers-overview)
     - [2. Install the Library](#2-install-the-library)
     - [3. Navigating the Model Hub](#3-navigating-the-model-hub)
     - [4. Pipelines Quickstart](#4-pipelines-quickstart)
     - [5. Pipeline Execution Internals](#5-pipeline-execution-internals)
     - [6. Model Management and Caching](#6-model-management-and-caching)
     - [7. Version Checks and Quick Tests](#7-version-checks-and-quick-tests)
   - [Datasets](#datasets)
     - [1. Load a Dataset](#1-load-a-dataset)
     - [2. Standard Splits](#2-standard-splits)
     - [3. Dataset Cards](#3-dataset-cards)
     - [4. Apache Arrow Storage](#4-apache-arrow-storage)
     - [5. Working with Partitions](#5-working-with-partitions)

---

## AI Environment Setup for PyTorch + CUDA 12.8 (RTX 5080)

This chapter consolidates the foundational concepts required to build a GPU-enabled PyTorch environment on an RTX 5080 with CUDA 12.8 support.

### 1. Virtual Environments (`venv`)

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

### 2. Package Management: `pip`, `setuptools`, `wheel`

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

### 3. PyTorch

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

### 4. CUDA and GPU Compatibility

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

### 5. Testing PyTorch GPU Support

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

### 6. Version Checks and Utilities

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

### 7. TorchVision and TorchAudio

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

### 8. Notes on Nightly Builds

- Nightly versions (`--pre`) are built automatically from the latest commits.
- They may include early support for new GPUs or CUDA toolkits.
- Slight instability is possible, but they’re ideal for testing and cutting-edge hardware.

To downgrade to a stable version later:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128 --force-reinstall
```

---

### 9. Summary of Key Learnings

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

### 10. End-to-End Setup Blueprint

This subsection condenses the essential actions from Sections 1–9 into a practical workflow that can be followed from a clean system to a validated GPU-enabled PyTorch environment.

1. **Prepare the workspace** by creating a project directory and activating a Python virtual environment to isolate dependencies and guarantee reproducibility.
2. **Upgrade foundational packaging tools** (`pip`, `setuptools`, and `wheel`) to ensure compatibility with the latest binary distributions and to speed up installations.
3. **Install PyTorch with matching CUDA support**, optionally selecting the nightly channel when targeting cutting-edge GPUs or driver/toolkit combinations.
4. **Verify CUDA and driver alignment** by checking GPU architecture support tables, confirming the installed CUDA toolkit, and reviewing driver status with `nvidia-smi`.
5. **Run GPU functionality tests** in PyTorch to confirm device discovery, architecture availability, and successful execution of tensor operations on the GPU.
6. **Validate auxiliary libraries** such as TorchVision and TorchAudio to ensure that vision and audio tooling versions align with the installed PyTorch build.
7. **Document key learnings**—including SM mappings, packaging utilities, and nightly build considerations—to retain institutional knowledge for future setup iterations.

Use this blueprint whenever you need a succinct, end-to-end reminder of the environment setup process; each bullet links conceptually back to the detailed chapters above.

---

## Working with Hugging Face

This chapter summarizes the key workflows for exploring models on Hugging Face and running inference with the `transformers` library.

### Models

#### 1. Transformers Overview

**Purpose:**
The `transformers` library centralizes state-of-the-art Transformer architectures (e.g., GPT, BERT, T5, Whisper, Stable Diffusion) with ready-to-use APIs.

**What it handles for you:**
- Downloading and loading pretrained models and tokenizers.
- Converting raw text, audio, or images into tensors the models can process.
- Running inference across CPUs and GPUs with device-aware optimizations.
- Formatting and post-processing model outputs into human-readable results.

#### 2. Install the Library

Install or upgrade to the latest release of `transformers`:

```bash
pip install --upgrade transformers
```

This ensures you receive current model definitions, tokenizer updates, and pipeline improvements.

#### 3. Navigating the Model Hub

The **Models** tab on [huggingface.co](https://huggingface.co/models) is the central directory for community and official checkpoints.

- Use filters for task, dataset, library, language, license, and hardware requirements.
- Sort models by Most Downloads, Trending, or Most Liked to surface popular checkpoints.
- Each **Model Card** provides: overview and intended use, example code snippets, evaluation metrics (per dataset or benchmark), training data references, licensing terms, and known limitations or ethical considerations.
- KPIs to watch include download counts, likes, last modified date, supported tasks, and compatible libraries.

#### 4. Pipelines Quickstart

`pipeline` offers a high-level interface for rapid experimentation:

```python
from transformers import pipeline

generator = pipeline("text-generation", model="gpt2")
result = generator("The future of open science", max_new_tokens=20)
print(result[0]["generated_text"])
```

- Automatically selects the appropriate tokenizer and model classes for the task.
- Accepts text, audio, or vision inputs depending on the pipeline type.
- Returns structured outputs such as generated text, classification labels, or transcription segments.

#### 5. Pipeline Execution Internals

When a pipeline runs, it orchestrates several steps:

1. Loads the matching `AutoTokenizer` and `AutoModel` (or task-specific subclasses).
2. Downloads the latest weights from the Hugging Face Hub if they are not already cached locally.
3. Tokenizes the input prompt and maps it to tensors for the target device (CPU or GPU).
4. Streams the tensors through the model to produce logits and decoded outputs.
5. Applies task-specific post-processing (e.g., text decoding, probability sorting, or audio chunk stitching).

#### 6. Model Management and Caching

- Authenticate once with `huggingface-cli login` to access private models or higher rate limits.
- By default, models and tokenizers are cached under `~/.cache/huggingface`; reuse of the same model avoids repeated downloads.
- Remove a directory within the cache to force a fresh download when updated weights are released.
- Set the `HF_HOME` environment variable if you prefer a custom cache location.

#### 7. Version Checks and Quick Tests

Keep tooling aligned and validate that everything runs on the intended hardware:

```bash
pip install --upgrade "transformers[torch]" huggingface-hub
```

```bash
python - <<'PY'
import torch

print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
    print("cuda toolkit:", torch.version.cuda)
PY
```

```python
from transformers import pipeline

generator = pipeline("text-generation", model="gpt2", device_map="auto")
output = generator("This AI assistant always", max_new_tokens=20)
print(output[0]["generated_text"])
```

The first command synchronizes core dependencies, the second verifies PyTorch GPU support, and the final snippet confirms that a Hugging Face model can be loaded and executed locally.

### Datasets

#### 1. Load a Dataset

**Purpose:**
`datasets` lets you pull curated datasets from the Hugging Face Hub with a single call.

**Command:**
```python
from datasets import load_dataset

dataset = load_dataset("TWI-RM/BIOMERT_Italian")
print(type(dataset))
```

**Expected output:**
```
DatasetDict
```

The loader returns an Arrow-backed `DatasetDict` (splits as keys, `Dataset` objects as values) ready for inspection or iteration.

#### 2. Standard Splits

Most Hub datasets ship with named splits so you can immediately separate training and evaluation data.

| Split | Primary use | Typical share |
|-------|-------------|----------------|
| `train` | Model fitting | 70–80% |
| `validation` | Hyperparameter tuning and early stopping | 10–15% |
| `test` | Final unbiased evaluation | 10–15% |

You can load an individual split on demand:

```python
train = load_dataset("TWI-RM/BIOMERT_Italian", split="train")
print(len(train))
```

#### 3. Dataset Cards

Each dataset has a Dataset Card that documents the source, schema, licensing, and recommended uses. Read it before integrating a dataset to confirm label definitions, preprocessing steps, and ethical considerations. The card also links to benchmarks and related work that inform how to evaluate your model.

#### 4. Apache Arrow Storage

Hugging Face stores tables in Apache Arrow for fast columnar access:

- Arrow is a binary column store that reads efficiently on CPU and GPU.
- It is interoperable with frameworks like Pandas, PyTorch, and TensorFlow.
- Datasets stream from disk via memory mapping, keeping the memory footprint low.
- Reusing the same dataset does not require a reload because Arrow caching persists between sessions.

Loaders return Arrow-native blocks automatically, so you can treat each split like an in-memory dataset while benefiting from lazy loading.

#### 5. Working with Partitions

**What is a partition?**
A *partition* is any subset of records that together form a complete, non-overlapping division of a dataset. The Hugging Face `DatasetDict` treats the familiar `train`, `validation`, and `test` splits as first-class partitions, but you can create as many additional groups as you need (for example, a `holdout` set for long-term regression testing or an `inference` slice for demo traffic).

**Why partitions matter:**
- They keep evaluation honest by ensuring models never see the held-out examples during training.
- They let you tailor preprocessing: heavy augmentations might apply only to the training partition, while deterministic tokenization is shared across all partitions.
- They support reproducibility; storing the slicing logic in code guarantees that teammates regenerate identical partitions.

**Standard partitioning workflows:**
1. **Named split selection** – Load the predefined partitions the dataset author published. This is the default when you call `load_dataset("<name>")`.
2. **Percentage-based slicing** – Request custom ratios (e.g., 80/10/10) when the dataset ships as a single table. This is the most common option for personal or internal corpora.
3. **Stratified sampling** – Balance the class distribution across partitions by filtering before you slice or by using `train_test_split` with the `stratify_by_column` argument.
4. **Deterministic shards** – Fix a random seed so that partitions stay stable across runs, which is essential for comparing experiments over time.

**Example: create percentage-based partitions**

```python
custom = load_dataset(
    "mc4",
    split={
        "train": "train[:90%]",
        "validation": "train[90%:95%]",
        "test": "train[95%:]",
    },
)
print({name: len(part) for name, part in custom.items()})
```

Each partition behaves like a regular `Dataset`, so you can inspect schema metadata, iterate through examples, or combine subsets. For example, you can temporarily merge the training and validation partitions for a larger fine-tuning run, then re-separate them as needed:

```python
from datasets import concatenate_datasets

train = custom["train"]
validation = custom["validation"]
merged = concatenate_datasets([train, validation])
print(train.column_names)
print(merged.num_rows)
```

When you need stratified or random splits, reach for the built-in helper:

```python
balanced = train.train_test_split(
    test_size=0.2,
    seed=42,
    stratify_by_column="label",
)
print({name: len(part) for name, part in balanced.items()})
```

This workflow keeps preprocessing centralized while letting you express advanced partitioning strategies—random shuffles, class-balanced samples, or experiment-specific holdouts—directly in code.

---

*Document generated to summarize AI environment setup for PyTorch + CUDA 12.8 with RTX 5080 and core Hugging Face workflows.*
