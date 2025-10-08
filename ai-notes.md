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
     - [4. Model Repositories, Classes, and Checkpoints](#4-model-repositories-classes-and-checkpoints)
     - [5. Pipelines Quickstart](#5-pipelines-quickstart)
     - [6. Pipeline Execution Internals](#6-pipeline-execution-internals)
     - [7. Model Management and Caching](#7-model-management-and-caching)
     - [8. Version Checks and Quick Tests](#8-version-checks-and-quick-tests)
     - [9. Task Selection and Model Matching](#9-task-selection-and-model-matching)
   - [Datasets](#datasets)
     - [1. Load a Dataset](#1-load-a-dataset)
     - [2. Standard Splits](#2-standard-splits)
     - [3. Dataset Cards](#3-dataset-cards)
     - [4. Apache Arrow Storage](#4-apache-arrow-storage)
     - [5. Working with Partitions](#5-working-with-partitions)
   - [Text Classification](#text-classification)
     - [1. Sentiment Analysis](#1-sentiment-analysis)
     - [2. Grammatical Correctness](#2-grammatical-correctness)
     - [3. Question-Answering Natural Language Inference (QNLI)](#3-question-answering-natural-language-inference-qnli)
     - [4. Dynamic Category Assignment](#4-dynamic-category-assignment)
     - [5. Challenges of Text Classification](#5-challenges-of-text-classification)
     - [6. Model Landscape Playbook](#6-model-landscape-playbook)
   - [Text Summarization](#text-summarization)
     - [1. Summarization Overview](#1-summarization-overview)
     - [2. Extractive vs. Abstractive Approaches](#2-extractive-vs-abstractive-approaches)
     - [3. Use Cases for Extractive Summarization](#3-use-cases-for-extractive-summarization)
     - [4. Extractive Summarization in Action](#4-extractive-summarization-in-action)
     - [5. Use Cases for Abstractive Summarization](#5-use-cases-for-abstractive-summarization)
     - [6. Abstractive Summarization in Action](#6-abstractive-summarization-in-action)
     - [7. Controlling Summary Length with Token Parameters](#7-controlling-summary-length-with-token-parameters)
     - [8. Interpreting Token Length Effects](#8-interpreting-token-length-effects)
     - [9. Model Landscape Playbook](#9-model-landscape-playbook)
   - [Document Q&A](#document-qa)
     - [1. Concept Overview](#1-concept-overview)
     - [2. Sample Scenario](#2-sample-scenario)
     - [3. Build a Minimal Pipeline](#3-build-a-minimal-pipeline)
     - [4. Verification Checklist](#4-verification-checklist)
     - [5. Model Landscape Playbook](#5-model-landscape-playbook)
   - [Auto Models and Tokenizers](#auto-models-and-tokenizers)
     - [1. AutoModel Essentials](#1-automodel-essentials)
     - [2. AutoTokenizer Workflow](#2-autotokenizer-workflow)
     - [3. Matching Models and Tokenizers](#3-matching-models-and-tokenizers)
     - [4. Uploading Artifacts to Hugging Face](#4-uploading-artifacts-to-hugging-face)
     - [5. Common Pitfalls Checklist](#5-common-pitfalls-checklist)
   - [The Hub API](#the-hub-api)
     - [1. Setup and Authentication](#1-setup-and-authentication)
     - [2. Cheatsheet: High-Value Hub API Calls](#2-cheatsheet-high-value-hub-api-calls)
     - [3. Find Popular Models for a Task](#3-find-popular-models-for-a-task)
     - [4. Inspect Model Metadata](#4-inspect-model-metadata)
     - [5. Enumerate Available Tasks](#5-enumerate-available-tasks)
     - [Preprocessing different modalities](#preprocessing-different-modalities)
       - [Preprocessing text](#preprocessing-text)
       - [Preprocessing images](#preprocessing-images)
       - [Preprocessing audio](#preprocessing-audio)
   - [Pipeline tasks and evaluations](#pipeline-tasks-and-evaluations)
     - [1. Pipeline vs. model components](#1-pipeline-vs-model-components)
     - [2. Selecting pipelines for common tasks](#2-selecting-pipelines-for-common-tasks)
     - [3. Evaluating pipeline performance](#3-evaluating-pipeline-performance)
   - [Computer vision](#computer-vision)
     - [1. Vision model building blocks](#1-vision-model-building-blocks)
     - [2. Image classification workflow](#2-image-classification-workflow)
     - [3. Object detection pipeline](#3-object-detection-pipeline)
     - [4. Segmentation playbook](#4-segmentation-playbook)
     - [5. Fine-tuning computer vision models](#5-fine-tuning-computer-vision-models)
     - [6. Model Landscape Quick Reference](#6-model-landscape-quick-reference)
   - [Speech recognition and audio generation](#speech-recognition-and-audio-generation)
     - [1. Task overview](#1-task-overview)
     - [2. Model landscape overview](#2-model-landscape-overview)
     - [3. End-to-end workflow example](#3-end-to-end-workflow-example)
     - [4. Implementation checklist](#4-implementation-checklist)

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

More tokenizer-specific preprocessing steps are unpacked in [Preprocessing text](#preprocessing-text) so you can see the full normalization and padding workflow in action.

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

#### 4. Model Repositories, Classes, and Checkpoints

Every Hugging Face model you load is the combination of three layers of abstraction. Keeping them straight prevents confusion when reading tutorials or translating code between architectures.

| Layer | What it represents | Whisper example |
| --- | --- | --- |
| **Model repository** | The Hub container that stores config files, tokenizer assets, and one or more checkpoints. Addressed by `<owner>/<model-name>`. | `openai/whisper-tiny` |
| **Model class** | The Python class that implements the architecture for a specific task head. Determines which forward pass signature is available. | `WhisperForConditionalGeneration` |
| **Checkpoint** | A snapshot of learned weights and biases at a specific training step. Tells the model class how to initialize its parameters. | The weights referenced by `from_pretrained("openai/whisper-tiny")` |

```python
from transformers import WhisperForConditionalGeneration, AutoProcessor

processor = AutoProcessor.from_pretrained("openai/whisper-tiny")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny")
```

Here is what happens in detail:

1. The repository name `openai/whisper-tiny` directs both the processor and the model class to the same Hub container.
2. `AutoProcessor` downloads tokenizer files such as `vocabulary.json`, `merges.txt`, and `preprocessor_config.json` so inputs match the training regime.
3. `WhisperForConditionalGeneration` picks the encoder–decoder architecture that Whisper uses for transcription tasks.
4. `from_pretrained` loads the checkpoint weights (saved under files such as `pytorch_model.bin`) into that class.
5. Once instantiated, you can call `model.generate()` to produce text tokens, and the processor will decode them back into readable text.

> **General Rule:** When experimenting with a new checkpoint, inspect both the model card and the repository files; mismatched model classes and checkpoints are the fastest path to runtime errors.

**Quick checklist before loading a model:**
- Confirm the repository ID you intend to use (public vs. private fork).
- Verify that the model class aligns with the task (e.g., `WhisperForConditionalGeneration` for ASR generation vs. `WhisperModel` for encoder-only use).
- Review the latest checkpoint date to ensure you are not pinning an outdated snapshot.

#### 5. Pipelines Quickstart

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

#### 6. Pipeline Execution Internals

When a pipeline runs, it orchestrates several steps:

1. Loads the matching `AutoTokenizer` and `AutoModel` (or task-specific subclasses).
2. Downloads the latest weights from the Hugging Face Hub if they are not already cached locally.
3. Tokenizes the input prompt and maps it to tensors for the target device (CPU or GPU).
4. Streams the tensors through the model to produce logits and decoded outputs.
5. Applies task-specific post-processing (e.g., text decoding, probability sorting, or audio chunk stitching).

More details on how raw text is normalized, tokenized, and padded before modeling are provided in [Preprocessing text](#preprocessing-text).

#### 7. Model Management and Caching

- Authenticate once with `huggingface-cli login` to access private models or higher rate limits.
- By default, models and tokenizers are cached under `~/.cache/huggingface`; reuse of the same model avoids repeated downloads.
- Remove a directory within the cache to force a fresh download when updated weights are released.
- Set the `HF_HOME` environment variable if you prefer a custom cache location.

#### 8. Version Checks and Quick Tests

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

#### 9. Task Selection and Model Matching

**Purpose:**
Reduce guesswork when choosing a checkpoint by mapping your use case to the right task, filtering the Hub effectively, and validating that the model meets latency and accuracy expectations.

**Common tasks and starter checkpoints:**

| Task | Typical Input/Output | Lightweight baseline | High-capacity option |
| --- | --- | --- | --- |
| `text-generation` | Prompt → extended narrative | `distilgpt2` (fast CPU inference) | `meta-llama/Meta-Llama-3-8B-Instruct` (better reasoning) |
| `question-answering` | Context passage + question → answer span | `distilbert-base-cased-distilled-squad` | `deepset/roberta-base-squad2` |
| `summarization` | Long article → concise abstract | `sshleifer/distilbart-cnn-12-6` | `facebook/bart-large-cnn` |
| `translation` | Source language text → target language text | `Helsinki-NLP/opus-mt-de-en` | `facebook/nllb-200-distilled-600M` |
| `sentence-similarity` | Pair of texts → similarity score | `sentence-transformers/all-MiniLM-L6-v2` | `sentence-transformers/all-mpnet-base-v2` |

**Selection checklist:**

1. **Define the task category** before browsing. The Hub’s task filter mirrors the `pipeline` task argument, so locking this down narrows results instantly.
2. **Set constraints** on model size, quantization format, and hardware (CPU vs. GPU vs. MPS). Smaller distilled models keep latency low; larger instruction-tuned checkpoints improve accuracy at the cost of memory.
3. **Compare evaluation metrics** in the model card (e.g., F1 for QA, ROUGE for summarization). Favor models evaluated on datasets that resemble your target domain.
4. **Review training data and licenses** to ensure compliance with your deployment scenario, especially for commercial or proprietary contexts.
5. **Prototype with a pipeline** using the short-listed model to validate answers on your own samples. Measure runtime and output quality before promoting it to production.

**Tip:** Save curated model shortlists in a shared note or spreadsheet with columns for task, model ID, parameters, eval metrics, and qualitative observations. This institutional memory speeds up future evaluations.

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


### Text Classification

This chapter outlines common text-classification tasks available through Hugging Face pipelines, emphasizing how to interpret predictions across customer feedback, education, and question-answering scenarios.

#### 1. Sentiment Analysis

**Purpose:** Assigns a sentiment label (positive, negative, neutral) to subjective text so teams can monitor customer or user reactions at scale.

**Applications:** Review moderation, social listening dashboards, support-ticket triage.

**Example: DistilBERT sentiment pipeline**

```python
from transformers import pipeline

my_pipeline = pipeline(
    "text-classification",
    model="distilbert-base-uncased-finetuned-sst-2-english",
)
print(my_pipeline("Wi-Fi is slower than a snail today!"))
```

**Expected output:**

```
[{'label': 'NEGATIVE', 'score': 0.99}]
```

**Interpretation:** The model is highly confident the sentence expresses a negative sentiment about the Wi-Fi experience.

#### 2. Grammatical Correctness

**Purpose:** Evaluates whether text follows grammatical rules, helping educators and content reviewers surface errors quickly.

**Applications:** Language-learning tools, editorial review, quality assurance for autogenerated content.

**Example: English grammar checker**

```python
from transformers import pipeline

grammar_checker = pipeline(
    task="text-classification",
    model="abdulmattinomtoso/English_Grammar_Checker",
)

print(grammar_checker("He eat pizza every day."))
```

**Expected output:**

```
[{'label': 'LABEL_0', 'score': 0.99}]
```

**Interpretation:** The pipeline flags the sentence as unacceptable (LABEL_0) with strong confidence because the verb tense is incorrect.

#### 3. Question-Answering Natural Language Inference (QNLI)

**Purpose:** Determines whether a statement (premise) supports the answer to a question, which is essential for fact-checking and knowledge retrieval systems.

**Applications:** Q&A assistants, automated support bots, content validation pipelines.

**Example: Cross-encoder QNLI classifier**

```python
from transformers import pipeline

classifier = pipeline(
    task="text-classification",
    model="cross-encoder/qnli-electra-base",
)

text = "Where is Seattle located? Seattle is located in Washington state."
print(classifier(text))
```

**Expected output:**

```
[{'label': 'LABEL_0', 'score': 0.997}]
```

**Interpretation:** LABEL_0 denotes entailment, so the model confirms the premise supports the answer to the question.

#### 4. Dynamic Category Assignment

**Purpose:** Performs zero-shot classification by mapping free-form text to whichever labels you provide at runtime, enabling flexible routing without task-specific fine-tuning.

**Applications:** Content moderation, intelligent routing for support queues, personalized recommendation flows.

**Example: Routing user request categories**

```python
from transformers import pipeline

classifier = pipeline(
    task="zero-shot-classification",
    model="facebook/bart-large-mnli",
)

text = "Hi team, the analytics dashboard shows an error every time I log in."
categories = ["Billing Question", "Product Feedback", "Technical Support"]

output = classifier(text, candidate_labels=categories)
print({"labels": output["labels"], "scores": [round(score, 3) for score in output["scores"]]})
```

**Expected output:**

```
{'labels': ['Technical Support', 'Product Feedback', 'Billing Question'], 'scores': [0.934, 0.045, 0.021]}
```

**Interpretation:** The request is confidently routed to the Technical Support queue, while other categories receive negligible probability.

#### 5. Challenges of Text Classification

Text classifiers must contend with several linguistic hurdles:

- **Ambiguity:** The same phrase can carry different meanings depending on context or domain jargon.
- **Sarcasm and irony:** Surface-level wording may oppose the intended sentiment, confusing literal models.
- **Multilingual input:** Classifiers trained on one language can misinterpret code-switching or regional expressions.

#### 6. Model Landscape Playbook

The pipelines above map directly to widely adopted checkpoints on the Hub. Start evaluations with the following short list before branching into niche fine-tunes.

| Model | Primary Use Case | Strengths | Limitations |
|-------|------------------|-----------|-------------|
| `facebook/bart-large-mnli` | Zero-shot or label-scarce classification | Handles arbitrary label sets via natural-language prompts; strong zero-shot baseline | Slower inference than distilled models; prompt phrasing impacts accuracy |
| `distilbert-base-uncased-finetuned-sst-2-english` | Sentiment analysis for English text | Lightweight with fast inference and solid sentiment accuracy | Limited to binary sentiment; English-only |
| `roberta-large` fine-tuned on domain data | Domain-specific multi-class classification | High accuracy when fine-tuned; robust contextual understanding | Requires substantial compute and labeled data for fine-tuning |

> **General Rule:** Benchmark a lightweight distilled model first to establish latency and accuracy baselines before scaling to larger encoders.

**Next steps:**
- Capture qualitative notes alongside accuracy metrics so business stakeholders understand trade-offs.
- When performance plateaus, explore adapters or LoRA fine-tunes on top of the shortlisted checkpoints.
- Document any prompt templates used for zero-shot models; phrasing changes can shift outputs materially.

---

### Text Summarization

This chapter demonstrates how Hugging Face pipelines condense source text into shorter narratives, contrasting extractive and abstractive workflows and highlighting how token parameters influence output length.

#### 1. Summarization Overview

**Purpose:** Reduce lengthy passages to focused summaries that preserve the key message while omitting redundant or tangential details.

**Process:**
- Accept full sentences or short paragraphs as input.
- Compress them into concise statements or bullet-style highlights.
- Maintain factual accuracy so downstream teams can make decisions quickly.

#### 2. Extractive vs. Abstractive Approaches

| Approach     | How it works                                                      | Strengths     | Considerations                                |
|--------------|-------------------------------------------------------------------|--------------------------------------------|-----------------------------------------------|
| Extractive   | Selects the most informative sentences directly from the source.  | Fast, factual, minimal hallucination risk. | May keep awkward phrasing or irrelevant detail |
| Abstractive  | Generates new text that paraphrases the source content.           | Fluent, coherent, adapts to narrative tone. | Requires guardrails to avoid fabrication       |

#### 3. Use Cases for Extractive Summarization

- **Legal brief triage:** Surface pivotal clauses so attorneys can skim precedence quickly.
- **Financial alerts:** Lift sentences describing revenue swings or guidance updates for analysts.

#### 4. Extractive Summarization in Action

```python
from transformers import pipeline

summarizer = pipeline("summarization", model="myamuda/extractive-summarization")

text = (
    "City council members met for a short session to approve the river cleanup budget. "
    "They highlighted that volunteer turnout doubled compared to last year, "
    "and the plan prioritizes removing plastic debris along the south bank."
)

summary = summarizer(text)
print(summary[0]["summary_text"])
```

**Expected output:**

```
City council members met for a short session to approve the river cleanup budget.
```

**Interpretation:** The extractive pipeline selects the most informative sentence verbatim, preserving factual wording from the source paragraph.

#### 5. Use Cases for Abstractive Summarization

- **News digests:** Provide concise recaps of local events without repeating the entire article.
- **Content recommendations:** Rephrase long-form reviews into short hooks for readers.

#### 6. Abstractive Summarization in Action

```python
from transformers import pipeline

summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

text = (
    "The neighborhood gardening club planted a pollinator garden beside the library. "
    "Volunteers logged the soil quality, chose low-maintenance flowers, and scheduled "
    "weekly watering shifts so families can learn about native plants."
)

summary = summarizer(text)
print(summary[0]["summary_text"])
```

**Expected output:**

```
The club organized a pollinator garden project near the library and set up a family-friendly maintenance plan.
```

**Interpretation:** The abstractive pipeline rewrites the source into a fluent sentence that blends the key actions into a single narrative.

#### 7. Controlling Summary Length with Token Parameters

```python
from transformers import pipeline

summarizer = pipeline(
    task="summarization",
    model="sshleifer/distilbart-cnn-12-6",
    min_new_tokens=15,
    max_new_tokens=40,
)

text = (
    "A high school robotics club built a solar-powered rover for the science expo. "
    "Students documented the build process, trained volunteers to operate the rover, "
    "and presented test results that showed longer battery life than last year's design."
)

shorter_summary = pipeline(
    task="summarization",
    model="sshleifer/distilbart-cnn-12-6",
    min_new_tokens=5,
    max_new_tokens=15,
)(text)[0]["summary_text"]

medium_summary = summarizer(text)[0]["summary_text"]

longer_summary = pipeline(
    task="summarization",
    model="sshleifer/distilbart-cnn-12-6",
    min_new_tokens=30,
    max_new_tokens=60,
)(text)[0]["summary_text"]

print("Short:", shorter_summary)
print("Medium:", medium_summary)
print("Long:", longer_summary)
```

**Expected output:**

```
Short: Students built a solar rover and proved it lasts longer.
Medium: The robotics club built a solar-powered rover, trained volunteers, and reported longer battery life than last year.
Long: Students in the robotics club built a solar-powered rover, documented each step, trained volunteers to drive it, and shared test data showing the new model outperforms last year's design.
```

#### 8. Interpreting Token Length Effects

- **Shorter setting:** Lower `min_new_tokens` and `max_new_tokens` force a brief highlight at the expense of nuance.
- **Medium setting:** Balanced token bounds retain the main actions without unnecessary detail.
- **Longer setting:** Higher token limits allow the model to elaborate on supporting context while staying within the summarization scope.

#### 9. Model Landscape Playbook

Match the extractive and abstractive techniques above with the Hub checkpoints most teams rely on before exploring custom fine-tunes.

| Model | Primary Use Case | Strengths | Limitations |
|-------|------------------|-----------|-------------|
| `facebook/bart-large-cnn` | Abstractive summarization of news and reports | Produces fluent, human-like summaries; strong ROUGE scores | May hallucinate facts; input length capped at 1024 tokens |
| `google/pegasus-large` | High-quality abstractive summarization for long-form documents | Trained on diverse summarization corpora; excels at abstractive paraphrasing | Heavy GPU memory footprint; slower decoding |
| `philschmid/bart-large-cnn-samsum` | Dialogue and meeting transcript summarization | Fine-tuned on conversational data; captures speaker turns | Less effective on formal prose or technical documents |

> **General Rule:** Apply factuality checks (e.g., entailment classifiers) to abstractive outputs before publishing summaries externally.

**Evaluation tips:**
- Test candidate models on a mix of structured reports and conversational transcripts to see where domain drift appears.
- Track latency alongside ROUGE or BERTScore so deployment teams understand the trade-offs between accuracy and throughput.
- When hallucinations persist, pair the summarizer with retrieval-augmented prompts that anchor it to verbatim passages.

---

### Document Q&A

#### 1. Concept Overview

- **Objective:** Answer natural-language questions by grounding responses in source documents such as PDFs, manuals, or policy guides.
- **Inputs:** A reference document (often chunked into passages) and a user question.
- **Output:** A concise answer plus optional metadata (confidence score, source span) that points back to the document.
- **Why it matters:** Document-grounded answers reduce hallucinations and keep HR, finance, or support teams aligned with approved content.

#### 2. Sample Scenario

Imagine an internal HR handbook stored as a PDF. We will work with a file named `US-Employee_Policy.pdf`, where page 7 contains the volunteer-time policy. Before we can pass the content to a QA model, we need to extract the text from the PDF and confirm exactly which page holds the relevant paragraph.

```python
from pypdf import PdfReader

reader = PdfReader("US-Employee_Policy.pdf")
page_number = 6  # zero-based index; page 7 in the printed handbook
document_text = reader.pages[page_number].extract_text()

print(f"US-Employee_Policy.pdf — Page {page_number + 1}\n" + "-" * 50)
print(document_text)
```

The output confirms that page 7 includes the passage we need:

```
US-Employee_Policy.pdf — Page 7
--------------------------------------------------
Annual volunteer day program
• Each full-time employee is eligible for 12 hours of volunteer time off per calendar year.
• Requests must be submitted two weeks in advance through the HR portal.
• Managers will confirm coverage needs before approvals are issued.
```

This extraction step matters for two reasons:

1. **Traceability:** When HR reviews the answer later, they can open the PDF to page 7 and verify the wording line by line.
2. **Reliable preprocessing:** QA pipelines depend on clean, machine-readable text. Automating the PDF-to-text step avoids manual copy/paste errors and keeps long documents searchable.

If an employee asks, "How many volunteer hours do we receive each year?", the answer should be verifiable against the highlighted bullet. With the `document_text` string captured above, we can now feed the exact PDF content into a QA pipeline.

#### 3. Build a Minimal Pipeline

```python
from transformers import pipeline

qa_pipeline = pipeline(
    task="question-answering",
    model="distilbert-base-cased-distilled-squad",
)

context = document_text

question = "How many volunteer hours do we receive each year?"

result = qa_pipeline(question=question, context=context)

print("Answer:", result["answer"])
print("Confidence:", round(result["score"], 3))
```

**What to expect:**

```
Answer: 12 hours
Confidence: 0.842
```

The model extracts the phrase "12 hours" because it matches the question’s semantics and appears in the provided context segment.

#### 4. Verification Checklist

1. **Traceability:** Confirm that the predicted answer text exactly matches a span in the extracted passage. Here, "12 hours" aligns with the second bullet in the PDF excerpt.
2. **Context coverage:** Ensure the context string contains all sentences needed to interpret the question. Add surrounding bullets or headings when meaning could be ambiguous.
3. **Chunking strategy:** For long PDFs, split pages or sections into overlapping chunks (~200–400 tokens) and feed each chunk to the pipeline before selecting the highest-scoring answer.
4. **Model suitability:** If answers sound uncertain or hallucinated, try a larger QA model (e.g., `deepset/roberta-base-squad2`) or fine-tune on domain-specific Q&A pairs.
5. **Human validation:** Keep a manual review step for high-impact answers so stakeholders can compare the model output with the authoritative source.

#### 5. Model Landscape Playbook

The retrieval and QA workflow above pairs best with complementary checkpoints that specialize in either extracting spans or generating natural-language answers.

| Model | Primary Use Case | Strengths | Limitations |
|-------|------------------|-----------|-------------|
| `deepset/roberta-base-squad2` | Extractive QA over short passages | High accuracy on span extraction; well-documented pipeline support | Requires relevant passage upfront; struggles with multi-hop reasoning |
| `facebook/dpr-question_encoder-single-nq-base` + `facebook/dpr-ctx_encoder-single-nq-base` | Dense retrieval for large corpora | Retrieves semantically similar passages at scale; integrates with Haystack | Needs vector index infrastructure; sensitive to domain shift |
| `google/flan-t5-large` | Generative QA with instructions | Handles abstractive answers and follow-up questions; supports few-shot prompts | May hallucinate if retrieval context is missing; larger deployment footprint |

> **General Rule:** Pair generative QA models with explicit retrieval context and guardrails to minimize unsupported answers.

**Operational advice:**
- Cache DPR embeddings so nightly re-indexing jobs finish quickly even as the document base grows.
- Track answer provenance by logging which chunk ID or retrieval score produced the final response.
- Escalate low-confidence answers to a human reviewer instead of returning a guess—especially for policy or compliance topics.

---

### Auto Models and Tokenizers

This chapter translates the automatic class machinery in `transformers` into actionable steps. Auto classes remove guesswork when pairing checkpoints with the exact Python implementations that know how to run them.

#### 1. AutoModel Essentials

- **Core idea:** Each AutoModel class inspects the model card metadata (`config.json`) and instantiates the concrete architecture (e.g., `BertForSequenceClassification`) without you hardcoding the class name.
- **Task-specific loaders:**

  | AutoModel class | Typical task | Notes |
  | --- | --- | --- |
  | `AutoModelForSequenceClassification` | Text classification, sentiment analysis, intent detection | Adds classification heads with logits over labels. |
  | `AutoModelForTokenClassification` | Named entity recognition, part-of-speech tagging | Produces token-level logits. |
  | `AutoModelForCausalLM` | Text generation, chat models | Supports sampling/beam search over next-token probabilities. |
  | `AutoModelForSeq2SeqLM` | Translation, abstractive summarization | Handles encoder-decoder checkpoints. |
  | `AutoModel` | Backbone-only representation | Returns hidden states without task heads. |

- **Usage pattern:**

  ```python
  from transformers import AutoModelForSequenceClassification

  model = AutoModelForSequenceClassification.from_pretrained(
      "distilbert-base-uncased-finetuned-sst-2-english"
  )
  logits = model(**tokenized_batch).logits
  ```

- **Pipelines vs. auto classes:** Pipelines deliver a turnkey experience (single function call, implicit device handling), while auto classes expose every intermediate step—ideal for custom training loops, gradient inspection, and advanced batching.

#### 2. AutoTokenizer Workflow

AutoTokenizers are the text-processing counterparts to AutoModels. They clean input, break it into tokens, and convert those tokens into numerical IDs that the model understands.

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
tokens = tokenizer("AI helps humans", return_tensors="pt")
```

**Flow of text through an AutoTokenizer:**

```
Raw text
  ↓ normalize (lowercase, strip accents, apply rules)
Normalized text
  ↓ tokenize (WordPiece, BPE, SentencePiece)
Tokens ("ai", "helps", "humans")
  ↓ map to IDs via vocabulary
Token IDs ([993, 1256, 4820])
  ↓ package inputs (add CLS/SEP, pad, create attention masks)
Model-ready batch
```

More details and examples on normalization, ID conversion, and padding appear in [Preprocessing text](#preprocessing-text).

- **Pairing matters:** Always load the tokenizer with the exact identifier you use for the model. Checkpoints can introduce new special tokens or change normalization rules; the paired tokenizer knows about those updates.
- **Cased vs. uncased checkpoints:**
  - *Uncased* models (e.g., `distilbert-base-uncased`) lowercase tokens, which boosts robustness on informal text and social media content.
  - *Cased* models (e.g., `bert-base-cased`) preserve capitalization for languages or domains where case carries meaning (named entities, German nouns).
- **Byte pair vs. WordPiece:** DistilBERT/ BERT rely on WordPiece (prefix `##` for subwords), while RoBERTa variants use BPE (`Ġ` to mark spaces). The AutoTokenizer selects the proper algorithm automatically.

#### 3. Matching Models and Tokenizers

- **One metadata source:** The tokenizer is resolved via the model's configuration; there is no separate "AutoTokenizer per model" registry you must maintain.
- **Consequences of mismatches:** Using `AutoTokenizer.from_pretrained("roberta-base")` with a DistilBERT checkpoint yields incompatible token IDs, similar to sending Morse code to a Braille reader—decoding fails because token IDs map to different embeddings.
- **Quick compatibility check:** After loading, compare `tokenizer.vocab_size` with the model's `config.vocab_size`. A mismatch indicates the wrong tokenizer or an outdated cache.
- **Selecting the right Auto class:** `AutoTokenizer` dispatches to the correct concrete implementation (`BertTokenizer`, `RobertaTokenizerFast`, etc.) based on configuration flags. You rarely need to import these specialized classes manually.

#### 4. Uploading Artifacts to Hugging Face

When you train a model locally (as an individual, a company, or a research lab), pushing to the Hugging Face Hub collects all necessary artifacts:

1. Run `huggingface-cli repo create <namespace>/<model-name>` (or create via the web UI).
2. Place the following files in your local repository:
   - `config.json`: architecture hyperparameters.
   - `pytorch_model.bin` (or `safetensors`): trained weights.
   - `tokenizer.json`, `tokenizer_config.json`, `special_tokens_map.json`, merges/vocab files as required by the tokenizer.
3. Use `AutoTokenizer.from_pretrained(<path>)` and `AutoModel.from_pretrained(<path>)` locally to validate loading before publishing.
4. Push with `huggingface-cli upload` or `git push` (after adding the Hub remote). The Hub auto-detects tokenizer assets and wires them to the model card so downstream users get the paired setup via a single identifier.

#### 5. Common Pitfalls Checklist

| Scenario | Symptom | Fix |
| --- | --- | --- |
| Tokenizer not paired with model | `RuntimeError: The tokenizer class you are using is not the same as the one used during training.` | Re-download with the same repo ID and clear stale cache entries. |
| Using pipelines for complex training loops | Limited control over batching, gradients, and logging. | Switch to explicit `AutoModel` + `AutoTokenizer` usage to own the data flow. |
| Forgetting case handling | Lowercase text fed into a cased model performs poorly on names. | Match preprocessing to the checkpoint (keep case for cased models). |
| Missing files when pushing to Hub | Model card shows "Tokenizer missing" or "Config missing" warnings. | Include tokenizer configuration files and rerun `huggingface-cli upload`. |
| Custom special tokens ignored | Generation outputs skip task-specific markers. | Call `tokenizer.add_special_tokens(...)` **before** resizing the model embeddings and retraining/fine-tuning. |


### The Hub API

#### 1. Setup and Authentication

- Install the tooling once so both the Python client and CLI are available:

  ```bash
  pip install --upgrade "huggingface_hub[cli]"
  ```

- Authenticate to unlock higher rate limits, gated models, and write access:

  ```bash
  huggingface-cli login
  ```

  Paste a [user access token](https://huggingface.co/settings/tokens) with the minimal scopes you need (typically `read` for exploration, `write` for uploading checkpoints).

- Environment variables to keep handy:
  - `HF_HOME`: set a custom cache root when working on small disks.
  - `HUGGINGFACEHUB_API_TOKEN`: export this for non-interactive scripts or CI runs that rely on the Hub API.

#### 2. Cheatsheet: High-Value Hub API Calls

| Goal | Key Call | Essential Filters / Arguments |
| --- | --- | --- |
| List models matching a task, framework, or author | `HfApi().list_models(filter=ModelFilter(...), limit=...)` | `task`, `library`, `author`, `language`, `license`, `model_name`, `trained_dataset`, `tags`, `limit`, `sort`, `direction`, `cardData` |
| Inspect a single repository in depth | `HfApi().model_info(repo_id, files_metadata=True)` | `files_metadata=True` returns size, checksum, and path for every file; `revision` pins to a commit or tag. |
| Quickly download a specific artifact | `hf_hub_download(repo_id, filename, revision=None)` | Combine with `local_dir` / `local_dir_use_symlinks` to control caching; provide `token` when working with private repos. |
| Search curated datasets in the same manner | `HfApi().list_datasets(filter=DatasetFilter(...))` | Filter by `task_categories`, `languages`, `license`, `size_categories`, `download` / `likes` sorting. |
| Discover Spaces for interactive demos | `HfApi().list_spaces(filter=SpaceFilter(...))` | Filter on `task`, `sdk`, `runtime`, `hardware`. |

The `ModelFilter`, `DatasetFilter`, and `SpaceFilter` helper classes accept keyword arguments directly; passing multiple filters applies a logical AND.

#### 3. Find Popular Models for a Task

```python
from huggingface_hub import HfApi, ModelFilter

api = HfApi()

top_summarizers = api.list_models(
    filter=ModelFilter(task="summarization", library="transformers"),
    sort="downloads",      # also accepts "likes" or "last_modified"
    direction=-1,           # -1 for descending, 1 for ascending
    limit=5,
    cardData=True,          # pull model card metadata for richer context
)

for model in top_summarizers:
    print(
        f"{model.modelId} | downloads={model.downloads} | likes={model.likes} | "
        f"pipeline={model.pipeline_tag} | datasets={model.cardData.get('datasets', [])}"
    )
```

Key insights when ranking models programmatically:

- `sort` accepts `"downloads"`, `"likes"`, `"trending"`, or `"last_modified"`. Pair it with `direction` to toggle ascending vs. descending order.
- Use additional filters like `ModelFilter(author="facebook")`, `ModelFilter(language="en")`, or `ModelFilter(trained_dataset="cnn_dailymail")` to tailor the shortlist.
- Set `full=True` to retrieve extended metadata (tags, model card fields, library compatibility), which is essential when you need to confirm quantization formats or inference frameworks.
- Combine results with `api.list_models` for multiple tasks by iterating through a list of `ModelFilter` definitions and merging the outputs into your own ranking logic.

#### 4. Inspect Model Metadata

```python
from huggingface_hub import HfApi

api = HfApi()
model_id = "facebook/bart-large-cnn"

info = api.model_info(model_id, files_metadata=True)

print("Model:", info.modelId)
print("Private:", info.private)
print("Last modified:", info.lastModified)
print("Sha:", info.sha)
print("Downloads:", info.downloads)
print("Tags:", info.tags)
print("License:", info.cardData.get("license"))
print("Example input:", info.cardData.get("widget", [{}])[0].get("inputs"))

for file in info.siblings[:3]:
    print(f"{file.rfilename} — {file.size} bytes — {file.lfs is not None and 'LFS' or 'regular'}")
```

Helpful metadata fields when triaging candidates for production:

- **`downloads` and `likes`** signal community adoption and can hint at maintenance activity.
- **`pipeline_tag` and `tags`** confirm the supported tasks, modalities, and frameworks.
- **`cardData`** exposes structured model card details such as training datasets, license, and evaluation metrics; combine them with your own quality gates.
- **`siblings`** lists every file in the repository, enabling checks for `safetensors` availability, quantized variants, or missing tokenizer assets before scheduling downloads.

#### 5. Enumerate Available Tasks

```python
import requests

headers = {
    "Accept": "application/json",
    "User-Agent": "ai-lab-notes/1.0",
}

response = requests.get("https://huggingface.co/api/tasks", headers=headers, timeout=30)
response.raise_for_status()
tasks = response.json()

print(f"Total tasks: {len(tasks)}")

for task_id, task_info in list(tasks.items())[:8]:
    description = task_info.get("description", "").strip().replace("\n", " ")
    print(f"{task_id} → {task_info['label']} | {description}")

# Access structured metadata on demand
audio_tasks = [tid for tid, info in tasks.items() if info.get("type") == "audio"]
print("Audio-centric tasks:", audio_tasks)
```

The `/api/tasks` endpoint returns a JSON dictionary keyed by task identifiers. Each entry includes:

- `label`: human-friendly name suitable for menus or logs.
- `type`: coarse modality grouping (`text`, `audio`, `vision`, `tabular`, etc.).
- `description`: concise explanation of the task’s goal (verify its presence before displaying it; older tasks may omit the field).
- `widget_models`: curated example repositories you can plug directly into demos.

Use the returned keys to build selection UIs or to validate that a requested task matches the Hub’s canonical taxonomy before issuing `list_models` queries.

#### 6. Search and Filter Datasets Programmatically

```python
from huggingface_hub import HfApi, DatasetFilter

api = HfApi()

top_text_classification = api.list_datasets(
    filter=DatasetFilter(
        task_categories=["text-classification"],
        languages=["en", "de"],
        size_categories=["10K<n<100K"],
        license="apache-2.0",
    ),
    sort="downloads",
    direction=-1,
    limit=5,
    full=True,
)

for dataset in top_text_classification:
    print(
        f"{dataset.id} | downloads={dataset.downloads} | likes={dataset.likes} | "
        f"languages={dataset.cardData.get('languages')} | configs={dataset.cardData.get('configs')}"
    )

# Inspect the richest candidate in more detail
selected = top_text_classification[0].id
details = api.dataset_info(selected, files_metadata=True)

print("First files:", [s.rfilename for s in list(details.siblings)[:3]])
print("Preview splits:", details.cardData.get("splits"))
print("Preview features:", details.cardData.get("features"))
```

Key dataset filters and helpers to keep in mind:

| Need | Function | Essential Arguments / Notes |
| --- | --- | --- |
| Enumerate public datasets with compound filters | `HfApi().list_datasets(filter=DatasetFilter(...))` | Combine `task_categories`, `languages`, `size_categories`, `license`, `tags`, `author`, `benchmark` for fine-grained slicing; `full=True` hydrates `cardData`. |
| Quickly iterate over dataset metadata | `DatasetFilter(...)` | Accepts both scalars and lists; omit parameters to broaden the query; chaining multiple filters results in an AND logic. |
| Pull the authoritative dataset card and file manifest | `HfApi().dataset_info(repo_id, files_metadata=True)` | Surfaces configs, splits, and schema snippets stored in the card plus every artifact under `siblings`; set `revision` to lock to a snapshot. |
| Extract reusable taxonomy metadata | `HfApi().get_dataset_tags()` | Returns the available filter values (tasks, languages, licenses, sizes) so you can populate dropdowns without hard-coding choices. |

When designing dataset discovery workflows, start with `get_dataset_tags()` to present valid options, pass the user’s selections into `DatasetFilter`, and fall back to `dataset_info()` to hydrate UI cards or validation logs before downloads begin.

#### 7. CLI vs API: Verified Comparison

| Task | CLI (`hf` / `huggingface-cli`) | API (`huggingface_hub`) |
| --- | --- | --- |
| Models & datasets search | No native search command; quickest option is still the web UI or piping API responses through `curl`/`jq` manually once you know the exact repo IDs. | `list_models()` / `list_datasets()` provide structured filtering (task, language, size, license) and ready-to-use metadata objects. |
| Clone or download repositories | `hf download <repo-id> [files...]` fetches targeted assets without Git; pair with `--repo-type dataset` for dataset payloads. | `hf_hub_download()` mirrors the same capability but gives you Python control over caching, retries, and post-processing. |
| Inspect metadata / configs | CLI can show environment info (`hf env`) but cannot expose dataset cards or split schemas directly. | `dataset_info()` and `model_info()` expose card data, file manifests, and download counts for automated triage. |
| Automations & batch processing | Shell scripts can chain `hf download` or `hf upload`, yet handling pagination, retries, or conditional logic remains manual. | Python API surfaces paginated generators, structured responses, and first-class auth handling—ideal for complex filters, scoring, and orchestration. |
| Token handling & auth | `hf auth login` stores tokens locally and is ideal for quick terminals or CI bootstrap. | Pass `token=` to API calls or set `HUGGINGFACEHUB_API_TOKEN` for headless workflows; integrate with secrets managers and retries easily. |

#### 8. Summary: When to Prefer CLI vs API

- **Reach for the CLI** when you already know the repository slug, need to grab or upload artifacts quickly, or want a one-off login without writing code.
- **Reach for the API (Python)** when you must rank or filter datasets/models programmatically, enrich dashboards with metadata, or integrate Hub queries into automation pipelines.
- Mix both when prototyping: sketch the filter logic with `list_datasets()` in a notebook, then bake the final repo IDs into shell scripts powered by `hf download` for reproducible deployments.


---

### Preprocessing different modalities

Preprocessing turns raw user input into the normalized tensors that pretrained checkpoints expect. Without these transformations, the model receives signals that differ from the data it saw during training, which leads to unstable predictions or outright failures.

> **General Rule:** Mirror the preprocessing recipe that the checkpoint used during training so your runtime inputs follow the same distribution as the model’s training data.

#### Preprocessing text

Text preprocessing standardizes string inputs before tokenization so every batch has a consistent representation.

```
+-----------------------------------------+
|                 Raw text                |
+-----------------------------------------+
                     |
                     v
+-----------------------------------------+
|        Normalize casing / remove        |
|            special characters           |
+-----------------------------------------+
                     |
                     v
+-----------------------------------------+
|      Pre-tokenize words or subwords     |
+-----------------------------------------+
                     |
                     v
+-----------------------------------------+
|     Convert tokens to vocabulary IDs    |
+-----------------------------------------+
                     |
                     v
+-----------------------------------------+
|  Apply padding to reach uniform length  |
+-----------------------------------------+
```

- **Tokenizer:** Maps text to the model input space by applying normalization rules, splitting text into tokens, and translating tokens into integer IDs.
- **Normalization:** Handles lowercasing, punctuation stripping, whitespace cleanup, or accent folding so semantically equivalent variants collapse to the same representation.
- **Pre-tokenization:** Breaks text into manageable pieces (words, subwords, characters) that downstream algorithms can convert to IDs.
- **ID conversion:** Looks up each token inside the model vocabulary to produce a tensor of token IDs.
- **Padding:** Adds special tokens (for example, `<pad>`) so that shorter sequences match the maximum length in the batch. Padding is crucial because GPUs prefer rectangular tensors; without it, batching would require costly ragged structures and attention masks could not reliably mask irrelevant positions.

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
normalized = tokenizer.normalizer("Do you need more eclairs?")
tokens = tokenizer("Do you need more eclairs?", return_tensors="pt", padding=True)

print(normalized)
print(tokens["input_ids"])
```

This snippet reveals the normalization output and shows how padding creates a batch-friendly tensor.

#### Preprocessing images

Moving from text to images introduces intensity normalization, consistent sizing, and processor utilities that coordinate the transformations.

```
+-----------------------------------------+
|                Raw image                |
+-----------------------------------------+
                     |
                     v
+-----------------------------------------+
|       Normalize pixel intensities       |
+-----------------------------------------+
                     |
                     v
+-----------------------------------------+
|   Resize to the model's expected shape  |
+-----------------------------------------+
                     |
                     v
+-----------------------------------------+
|   Apply processor-specific transforms   |
+-----------------------------------------+
                     |
                     v
+-----------------------------------------+
|  Deliver tensors to the vision encoder  |
|          or multimodal backbone         |
+-----------------------------------------+
```

- **Normalization:** Scales pixel values (for example, from 0–255 to 0–1 or to mean/variance pairs) to match the distribution used during training.
- **Resize:** Ensures that every image matches the input shape that the vision model requires, preventing convolution or attention layers from receiving unexpected dimensions.
- **Processor pipeline:** Applies the same augmentations or tensor formatting that the model card specifies (channel order, center cropping, text prompt pairing, etc.).

```python
from transformers import BlipForConditionalGeneration, BlipProcessor
from PIL import Image

model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")

image = Image.open("samples/prompt.jpg")
inputs = processor(images=image, return_tensors="pt")
generated_ids = model.generate(**inputs)
caption = processor.decode(generated_ids[0], skip_special_tokens=True)

print(caption)
```

The `BlipProcessor` bundles image preprocessing (resizing, normalization, pixel value scaling) with optional text tokenization so the BLIP model ingests aligned vision and language signals. Without it, you would have to manually replicate every transformation that keeps the image tensor and prompt embeddings synchronized.

> **General Rule:** Reuse the processor provided with a checkpoint—especially for multimodal models—so that pixel statistics, crop strategies, and text prompts stay consistent with the pretrained weights.

#### Preprocessing audio

Audio preprocessing converts waveforms into structured sequences that capture frequency information while respecting the model’s sample rate expectations.

```
+-----------------------------------------+
|               Raw waveform              |
+-----------------------------------------+
                     |
                     v
+-----------------------------------------+
|    Resample or validate sampling rate   |
+-----------------------------------------+
                     |
                     v
+-----------------------------------------+
|  Apply audio preprocessing (filtering,  |
|                 padding)                |
+-----------------------------------------+
                     |
                     v
+-----------------------------------------+
|      Extract spectrogram or feature     |
|                embeddings               |
+-----------------------------------------+
                     |
                     v
+-----------------------------------------+
|   Feed tensors into the acoustic model  |
+-----------------------------------------+
```

- **Audio feature extraction:** Transforms the waveform into log-Mel spectrograms or similar frequency-domain representations that speech models consume.
- **Padding:** Extends shorter clips with silence so the batch forms a rectangle, mirroring the requirement described in the text workflow.
- **Sampling rate alignment:** Ensures that the audio tensor matches the training configuration; mismatched sampling rates cause drift in temporal resolution and degrade recognition quality.

```python
from datasets import load_dataset
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq

dataset = load_dataset("CSTR-Edinburgh/vctk", split="train")
processor = AutoProcessor.from_pretrained("openai/whisper-small")
model = AutoModelForSpeechSeq2Seq.from_pretrained("openai/whisper-small")

sample = dataset[0]
features = processor(audio=sample["audio"], sampling_rate=16000, return_tensors="pt")
generated_ids = model.generate(**features)
transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)

print(transcription[0])
```

> **General Rule:** Keep the waveform sample rate and feature extraction parameters identical to the model card’s recommendations to avoid timing distortions and transcription errors.

### Pipeline tasks and evaluations

This new chapter connects the high-level `pipeline` interface with deeper evaluation routines so you can decide when to stay with a pipeline abstraction and when to drop to individual model components.

#### 1. Pipeline vs. model components

`pipeline(task, model=...)` wraps tokenization, model execution, and post-processing into a single callable. Building your own inference loop with `AutoTokenizer.from_pretrained()` and `AutoModel.from_pretrained()` exposes each stage, letting you customize batching, quantization, or decoding. The comparison below highlights when each option shines.

| Scenario | Pipeline advantage | When to prefer model components |
|----------|--------------------|----------------------------------|
| Rapid sentiment demo on a dozen support tickets | One function call handles tokenization, batching, and label mapping without boilerplate. | When you need to attach a custom classification head or average logits across multiple augmentations. |
| Batch caption generation for a marketing mock-up | Pipelines chain the processor (e.g., `BlipProcessor`) and model with minimal configuration. | When you must adjust beam search, apply nucleus sampling, or cache image embeddings for reuse. |
| QA bot prototype for an internal knowledge base | Pipelines fetch the best-matching QA model and deliver start/end spans immediately. | When you integrate retrieval-augmented generation and need to merge context passages manually. |
| Speech transcription on short audio clips | Processor + pipeline coordinates feature extraction and decoding in a single call. | When you want streaming transcription, chunk-level timestamps, or mixed precision control. |

**Practical rule:** Use a pipeline to explore a task, validate model quality, or ship a lightweight proof of concept. Switch to explicit model classes once you need architectural tweaks, multi-stage post-processing, or enterprise deployment controls (quantization, logging, retries).

#### 2. Selecting pipelines for common tasks

The cheat sheet below maps popular Hugging Face tasks to suitable pipeline arguments and input preparation reminders. Start from these defaults, then iterate with manual components as your accuracy or latency goals tighten.

| Task goal | Suggested `pipeline` call | Data preparation tips | Ideal follow-up customization |
|-----------|--------------------------|-----------------------|------------------------------|
| Flag risky customer reviews | `pipeline("text-classification", model="cardiffnlp/twitter-roberta-base-sentiment")` | Normalize casing and strip HTML before inference. | Calibrate thresholds per channel (support vs. app store). |
| Describe catalog photos | `pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")` | Resize images to the processor’s expected square resolution. | Fine-tune decoder to match brand terminology. |
| Summarize weekly stand-up notes | `pipeline("summarization", model="philschmid/bart-large-cnn-samsum")` | Chunk transcripts so each fits within the max token window. | Swap in a domain-specific checkpoint or enforce controlled vocabularies. |
| Draft melody descriptors from audio stems | `pipeline("audio-classification", model="MIT/ast-finetuned-audioset-10-10-0.4593")` | Convert audio to mono and resample to 16 kHz. | Add custom label smoothing or hierarchical genre aggregation. |

```text
+------------------------------+     +------------------------------+     +------------------------------+     +------------------------------+
|      Define pipeline task      | --> |   Select evaluation data   | --> |  Configure metrics & kwargs  | --> |   Run inference & aggregate   |
+------------------------------+     +------------------------------+     +------------------------------+     +------------------------------+
                                                                                                                  |
                                                                                                                  v
+------------------------------+
|  Interpret results & iterate  |
+------------------------------+
```

The flowchart reinforces that evaluation planning starts with the task definition and ends with interpretation. Centering these steps prevents ad-hoc metric choices later in the workflow.

#### 3. Evaluating pipeline performance

The `evaluate` library and pipeline outputs work together: you collect raw predictions, feed them to metrics, and translate numbers back to user impact.

```python
from evaluate import load
from transformers import pipeline

pipe = pipeline("image-classification", model="microsoft/resnet-50")
metric = load("precision")

example = pipe("dog_beach.png")[0]
score = metric.compute(
    predictions=[example["label"]],
    references=["Labrador retriever"],
)
print(score)
```

- **Positive example (high alignment):** Suppose precision and recall both exceed 0.9 on a validation set of beach dog photos. The balance indicates the labels your users care about are frequently correct, so you can safely promote the model to a limited beta. The small standard deviation in latency also signals that the pipeline’s internal batching suits your workload.
- **Negative example (dataset shift):** If recall drops to 0.45 when you switch to night-time images, interpret it as missed detections rather than random noise. At this point, graduate from the pipeline to manual components so you can add brightness augmentation, adjust confidence thresholds, or fine-tune with nocturnal samples.

When reading pipeline metrics:

1. **Match metric type to the task.** `accuracy` fits single-label classification; `wer` or `cer` better reflect transcription quality. Misaligned metrics hide real-world issues.
2. **Compare against baselines.** Always log a trivial baseline (e.g., majority class accuracy) so you can tell whether your pipeline genuinely outperforms naive heuristics.
3. **Watch batch-level variance.** Pipelines batch internally; spikes in latency or memory hint that manual batching would give you tighter control.
4. **Translate to user outcomes.** Convert F1 into statements such as “9 out of 10 urgent tickets are flagged,” which grounds decisions in business impact.

> **General Rule:** Treat pipeline metrics as the north star for product acceptance, then use low-level model access to squeeze out the incremental gains your edge cases demand.

### Computer vision

Computer vision pipelines convert pixel tensors into structured predictions such as labels, bounding boxes, or pixel-level masks. Pair this chapter with [Preprocessing images](#preprocessing-images) for normalization recipes that keep inputs compatible with pretrained checkpoints.

#### 1. Vision model building blocks

**Purpose:** Understand how vision models convert pixels into predictions so you can match the right pipeline to the problem.

**Key components:**
- **Backbone:** A convolutional or transformer network (e.g., ResNet, ViT, ConvNeXt, DETR) that extracts feature maps from the image.
- **Head:** A task-specific layer that maps features to labels, bounding boxes, or pixel masks.
- **Post-processing:** Functions that translate raw outputs into human-friendly results—top-1 labels, bounding box coordinates, or color-coded masks.

> **General Rule:** Keep the input resolution, channel order, and normalization values exactly as the model card specifies. Deviating from those details distorts activations and produces off-target detections.

**Cheat sheet: core computer-vision tasks**

| Task | What the pipeline returns | Representative model | Popular use cases |
| --- | --- | --- | --- |
| Image classification | Ranked labels with confidence scores | `microsoft/resnet-50`, `google/vit-base-patch16-224` | Quality inspection dashboards, content tagging |
| Object detection | Bounding boxes + labels per instance | `facebook/detr-resnet-50`, `YOLOv8` (via `ultralytics/yolov8m`) | Safety monitoring, shelf analytics |
| Semantic segmentation | Per-pixel label map | `briaai/RMBG-1.4`, `nvidia/segformer-b3-finetuned-ade-512-512` | Background removal, urban-scene planning |

#### 2. Image classification workflow

Image classification predicts the dominant class in a picture. The pipeline handles resizing, normalization, and batching so you can focus on interpretation.

```python
from datasets import load_dataset
from transformers import pipeline

dataset = load_dataset("lambdalabs/pokemon-blip-captions", split="train")
classifier = pipeline("image-classification", model="microsoft/resnet-50")

example = dataset[0]["image"]
prediction = classifier(example)[0]
print(prediction)
```

- **Why load a dataset?** `datasets` ensures consistent Pillow image objects, making it easy to iterate through samples or compare labels.
- **Interpreting results:** Inspect `prediction["label"]` and `prediction["score"]` to confirm the model recognizes the primary subject. High-confidence mislabels often reveal domain gaps.
- **Typical deployment targets:** Lightweight CNNs (e.g., MobileNetV2) excel on edge devices, while larger transformers (e.g., ViT-L) power cloud classification APIs.

```text
+----------------------------+     +--------------------------+     +------------------------------+
|       Select dataset       | --> |   Run classification     | --> |  Review label + confidence   |
+----------------------------+     +--------------------------+     +------------------------------+
```

Use this flow to validate each step before scaling to batch inference.

#### 3. Object detection pipeline

Object detection returns multiple labels, each tied to a bounding box. DETR-style models simplify the workflow by skipping anchor boxes and handling post-processing internally.

```python
from datasets import load_dataset
from transformers import pipeline

dataset = load_dataset("ashudeep/ff13k", split="test")
detector = pipeline(
    task="object-detection",
    model="facebook/detr-resnet-50",
    threshold=0.5,
)

image = dataset[42]["image"]
predictions = detector(image)
for item in predictions:
    print(item["label"], item["score"], item["box"])
```

- **Box format:** Each `box` dictionary exposes `xmin`, `ymin`, `xmax`, and `ymax` pixel coordinates. Convert them to integers before drawing overlays.
- **Confidence filtering:** Adjust `threshold` to balance recall and precision. Lower values surface more boxes but risk false positives.
- **Why detectors find people automatically:** Models such as DETR are trained on datasets like COCO that include 80+ everyday categories. Even if you only plan to keep “person” detections, the model still learns visual cues for animals, vehicles, and furniture because those labels appeared during pretraining. When an image only contains people, every high-confidence prediction falls into that category, giving the impression that the model was single-purpose.
- **Model selection guide:** DETR offers strong accuracy for multi-object scenes, while YOLO variants prioritize real-time inference on video streams.

To visualize detections, pair the pipeline with Matplotlib patches:

```python
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

fig, ax = plt.subplots(figsize=(8, 6))
ax.imshow(image)
for item in predictions:
    xmin, ymin = item["box"]["xmin"], item["box"]["ymin"]
    width = item["box"]["xmax"] - xmin
    height = item["box"]["ymax"] - ymin
    ax.add_patch(
        Rectangle((xmin, ymin), width, height, fill=False, edgecolor="lime", linewidth=2)
    )
    ax.text(xmin, ymin - 2, f"{item['label']} ({item['score']:.2f})", color="lime")
plt.axis("off")
plt.show()
```

> **General Rule:** Always compare detections against ground-truth annotations or a calibrated human review process. False negatives carry a higher cost in safety monitoring than low-confidence extra boxes.

#### 4. Segmentation playbook

Segmentation assigns a label to each pixel. The `image-segmentation` pipeline can remove backgrounds or highlight specific regions without manual masking.

```python
from transformers import pipeline
import matplotlib.pyplot as plt

segmenter = pipeline(
    task="image-segmentation",
    model="briaai/RMBG-1.4",
    trust_remote_code=True,
)

outputs = segmenter(image)
plt.imshow(outputs)
plt.axis("off")
plt.show()
```

- **Why `trust_remote_code=True`?** Some segmentation models ship custom decoding logic; this flag allows Hugging Face to execute the necessary helper functions.
- **Output format:** Pipelines return a Pillow image or NumPy array with the background removed. Compose it with the original image to blend or replace backgrounds.
- **Common model picks:** Use `facebook/mask2former-swin-large-ade` when you need dense scene labeling and `briaai/RMBG-1.4` for e-commerce cutouts.

> **General Rule:** Validate segmentation masks against multiple lighting conditions; models trained on bright scenes often underperform on low-light or motion-blurred inputs.

**Next steps:**
- Overlay masks on the original frame to assess edge quality.
- Export masks as PNGs for downstream editing or to feed into compositing software.
- Benchmark runtime on representative hardware; segmentation heads can be heavier than classifiers.

#### 5. Fine-tuning computer vision models

Fine-tuning adapts a pretrained vision model to a narrower domain, such as distinguishing between look-alike products or recognizing a new imaging modality. The typical workflow mirrors the Hugging Face end-to-end tutorials but can be summarized as follows.

1. **Adjust the model head:** Swap in a new classification or detection head sized for your label set, or configure the detection head to output additional categories. Models like `google/mobilenet_v2_1.0_224` are popular starting points when you need mobile-friendly deployments.
2. **Prepare the dataset:** Load images with `datasets`, split into train/test sets, and apply consistent transforms. Normalize pixel values using the processor tied to your checkpoint so training statistics match the pretrained backbone.
3. **Configure training:** Define `TrainingArguments` (learning rate, batch size, epochs) and a data collator. Monitor metrics such as accuracy or mean Average Precision (mAP) to verify progress.
4. **Train and evaluate:** Call `trainer.train()` to update weights, then `trainer.predict()` or `trainer.evaluate()` on the held-out split. Expect large jumps from baseline accuracy (e.g., 0.45 → 0.90) once the head specializes to your labels.

> **General Rule:** Freeze most backbone layers for small datasets to avoid catastrophic forgetting, and only unfreeze additional blocks after the head converges.

**Practical tips:**
- **Data augmentation:** Incorporate flips, crops, and color jitter to mimic real-world variety, especially if you only have a few hundred labeled images.
- **Label coverage:** Even when you only care about one class, include negative examples so the model learns to distinguish “not the target” cases. This is why pretrained detectors recognize people, vehicles, and props—they saw all of them during COCO training.
- **Evaluation cadence:** Track metrics on every epoch and save the best checkpoint with `load_best_model_at_end=True` to simplify deployment.

#### 6. Model Landscape Quick Reference

Anchor the workflows above to a short list of Hub checkpoints that balance quality, speed, and ease of deployment.

| Model | Primary Use Case | Strengths | Limitations |
|-------|------------------|-----------|-------------|
| `google/vit-base-patch16-224` | Image classification | Strong accuracy with minimal architectural tuning; benefits from fine-tuning | Requires substantial data augmentation; transformer backbone can be compute-heavy |
| `facebook/detr-resnet-50` | Object detection | End-to-end detection without anchor tuning; solid performance out of the box | Slower inference than YOLO-family detectors; moderate memory use |
| `briaai/RMBG-1.4` | Background removal / segmentation | Optimized for crisp cutouts; readily usable via pipelines | Narrow scope; relies on custom code (`trust_remote_code=True`) |

> **General Rule:** Profile inference on target hardware early—vision transformers and detection transformers have markedly different latency profiles on CPU vs. GPU.

**Adoption notes:**
- ViT backbones respond well to mixed-precision (`fp16`) inference; profile both CPU and GPU paths when planning deployment.
- DETR outputs benefit from post-processing thresholds tuned to your tolerance for false positives versus misses.
- RMBG models often require color correction after compositing; keep a simple post-processing step (e.g., gamma adjustment) in the pipeline.

### Speech recognition and audio generation

#### 1. Task overview

This chapter complements the earlier [Preprocessing audio](#preprocessing-audio) notes by focusing on end-to-end speech recognition (ASR) and speech synthesis (TTS) workflows within the Hugging Face ecosystem. The objective is to convert raw speech into text, extract voice characteristics, and regenerate speech that preserves the original speaker’s timbre.

#### 2. Model landscape overview

| Model | Segment | Primary Use Case | Strengths | Limitations |
|-------|---------|------------------|-----------|-------------|
| `openai/whisper-large-v3` | ASR | Multilingual transcription and translation | Robust to accents, noise, and code-switching; handles >50 languages | Requires GPU acceleration for real-time processing; large model footprint |
| `facebook/wav2vec2-large-960h` | ASR | English transcription with fine-tuning flexibility | High accuracy after domain fine-tuning; efficient streaming variants available | Performs best on clean audio; needs labeled audio-text pairs for adaptation |
| `speechbrain/spkrec-ecapa-voxceleb` | Speaker embedding | Voice print extraction for cloning or diarization | Compact embeddings with strong speaker discrimination | Additional smoothing needed for emotional prosody; sensitive to microphone mismatch |
| `microsoft/speecht5_tts` + `microsoft/speecht5_hifigan` | TTS | Neural text-to-speech with speaker adaptation | Supports speaker embeddings for voice cloning; modular vocoder | Requires high-quality speaker embeddings; inference latency higher than lightweight vocoders |
| `coqui/XTTS-v2` | TTS | Multilingual voice cloning with expressive prosody | Handles cross-language synthesis; community fine-tunes widely | Needs substantial VRAM; quality depends on speaker embedding quality |

> **General Rule:** Always secure consent and legal rights before cloning or recreating a speaker’s voice, and retain audit logs of input samples.

#### 3. End-to-end workflow example

The following pipeline demonstrates how to transcribe a speaker’s audio, derive reusable voice embeddings, and regenerate speech with matching characteristics.

**Step-by-step flow:**
1. **Capture raw audio** at 16 kHz or higher and normalize levels to avoid clipping.
2. **Transcribe speech** using Whisper (or a comparable ASR model) to obtain time-stamped text.
3. **Extract speaker embeddings** with ECAPA-TDNN or similar models to capture timbre and prosody.
4. **Prepare synthesis input** by pairing the transcript (or new text) with the speaker embedding vector.
5. **Synthesize speech** through SpeechT5 (encoder-decoder) and a neural vocoder such as HiFi-GAN.
6. **Post-process audio** for loudness normalization and export to deployment formats (e.g., WAV, MP3).

```
  +------------------------+     +----------------------+     +---------------------------+     +----------------------+     +-----------------------+
  |    Raw Audio Input     | --> |   Whisper ASR Text    | --> |  Speaker Embedding Model  | --> |  SpeechT5 Synthesis   | --> |  HiFi-GAN Vocoder      |
  +------------------------+     +----------------------+     +---------------------------+     +----------------------+     +-----------------------+
```

**Illustrative code snippet:**

```python
from transformers import pipeline, SpeechT5Processor, SpeechT5ForTextToSpeech
from datasets import load_dataset
import torch

# 1. Load sample audio
dataset = load_dataset("lj_speech", split="validation[:1]")
audio_sample = dataset[0]["audio"]

# 2. Speech recognition
asr = pipeline("automatic-speech-recognition", model="openai/whisper-large-v3")
transcription = asr(audio_sample["array"], sampling_rate=audio_sample["sampling_rate"])

# 3. Speaker embedding
from speechbrain.inference.speaker import EncoderClassifier
spk_encoder = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb")
with torch.no_grad():
    speaker_embedding = spk_encoder.encode_batch(torch.tensor(audio_sample["array"]).unsqueeze(0))

# 4. Text-to-speech with cloned voice
processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
inputs = processor(text=transcription["text"], return_tensors="pt")
speech = model.generate_speech(inputs["input_ids"], speaker_embeddings=speaker_embedding)

# 5. Vocoder post-processing (using the paired HiFi-GAN checkpoint)
from transformers import SpeechT5HifiGan
vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")
final_audio = vocoder(speech).cpu().numpy()
```

**Key safeguards:**
- Validate transcription timestamps before alignment; Whisper provides segment-level timing useful for lip-sync or captioning.
- Maintain a secure store for speaker embeddings; treat them as biometric data.
- Apply loudness normalization (e.g., ITU-R BS.1770) before distribution to ensure consistent playback volume.

#### 4. Implementation checklist

- **Data governance:** Confirm consent, retention policies, and encryption for raw and processed audio.
- **Evaluation metrics:** Track word error rate (WER) for ASR and mean opinion score (MOS) or cosine similarity for TTS output quality.
- **Latency profiling:** Measure per-stage latency (ASR, embedding, synthesis, vocoder) to ensure the pipeline meets real-time or batch throughput targets.
- **Fallback planning:** Keep a default synthetic voice ready in case speaker embedding extraction fails or drifts over time.

---

*Document generated to summarize AI environment setup for PyTorch + CUDA 12.8 with RTX 5080, core Hugging Face workflows, key text classification pipelines, text summarization techniques, document question-answering patterns, preprocessing strategies for text, images, audio, computer-vision pipelines, pipeline task evaluation guidelines, and speech-focused generation workflows.*


