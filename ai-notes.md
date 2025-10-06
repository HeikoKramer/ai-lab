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
3. [Text Classification](#text-classification)
   - [1. Sentiment Analysis](#1-sentiment-analysis)
   - [2. Grammatical Correctness](#2-grammatical-correctness)
   - [3. Question-Answering Natural Language Inference (QNLI)](#3-question-answering-natural-language-inference-qnli)
   - [4. Dynamic Category Assignment](#4-dynamic-category-assignment)
   - [5. Challenges of Text Classification](#5-challenges-of-text-classification)
4. [Text Summarization](#text-summarization)
   - [1. Summarization Overview](#1-summarization-overview)
   - [2. Extractive vs. Abstractive Approaches](#2-extractive-vs-abstractive-approaches)
   - [3. Use Cases for Extractive Summarization](#3-use-cases-for-extractive-summarization)
   - [4. Extractive Summarization in Action](#4-extractive-summarization-in-action)
   - [5. Use Cases for Abstractive Summarization](#5-use-cases-for-abstractive-summarization)
   - [6. Abstractive Summarization in Action](#6-abstractive-summarization-in-action)
   - [7. Controlling Summary Length with Token Parameters](#7-controlling-summary-length-with-token-parameters)
   - [8. Interpreting Token Length Effects](#8-interpreting-token-length-effects)

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

## Text Classification

This chapter outlines common text-classification tasks available through Hugging Face pipelines, emphasizing how to interpret predictions across customer feedback, education, and question-answering scenarios.

### 1. Sentiment Analysis

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

### 2. Grammatical Correctness

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

### 3. Question-Answering Natural Language Inference (QNLI)

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

### 4. Dynamic Category Assignment

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

### 5. Challenges of Text Classification

Text classifiers must contend with several linguistic hurdles:

- **Ambiguity:** The same phrase can carry different meanings depending on context or domain jargon.
- **Sarcasm and irony:** Surface-level wording may oppose the intended sentiment, confusing literal models.
- **Multilingual input:** Classifiers trained on one language can misinterpret code-switching or regional expressions.

---

## Text Summarization

This chapter demonstrates how Hugging Face pipelines condense source text into shorter narratives, contrasting extractive and abstractive workflows and highlighting how token parameters influence output length.

### 1. Summarization Overview

**Purpose:** Reduce lengthy passages to focused summaries that preserve the key message while omitting redundant or tangential details.

**Process:**
- Accept full sentences or short paragraphs as input.
- Compress them into concise statements or bullet-style highlights.
- Maintain factual accuracy so downstream teams can make decisions quickly.

### 2. Extractive vs. Abstractive Approaches

| Approach     | How it works                                                      | Strengths                                       | Considerations                                |
|--------------|-------------------------------------------------------------------|-------------------------------------------------|-----------------------------------------------|
| Extractive   | Selects the most informative sentences directly from the source.  | Fast, factual, minimal hallucination risk.      | May keep awkward phrasing or irrelevant detail |
| Abstractive  | Generates new text that paraphrases the source content.           | Fluent, coherent, adapts to narrative tone.     | Requires guardrails to avoid fabrication       |

### 3. Use Cases for Extractive Summarization

- **Legal brief triage:** Surface pivotal clauses so attorneys can skim precedence quickly.
- **Financial alerts:** Lift sentences describing revenue swings or guidance updates for analysts.

### 4. Extractive Summarization in Action

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

### 5. Use Cases for Abstractive Summarization

- **News digests:** Provide concise recaps of local events without repeating the entire article.
- **Content recommendations:** Rephrase long-form reviews into short hooks for readers.

### 6. Abstractive Summarization in Action

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

### 7. Controlling Summary Length with Token Parameters

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

medium_summary = summarizer(text)[0]["summary_text"]

shorter_summary = pipeline(
    task="summarization",
    model="sshleifer/distilbart-cnn-12-6",
    min_new_tokens=5,
    max_new_tokens=15,
)(text)[0]["summary_text"]

longer_summary = pipeline(
    task="summarization",
    model="sshleifer/distilbart-cnn-12-6",
    min_new_tokens=30,
    max_new_tokens=60,
)(text)[0]["summary_text"]

print("Medium:", medium_summary)
print("Shorter:", shorter_summary)
print("Longer:", longer_summary)
```

**Expected output:**

```
Medium: The robotics club built a solar-powered rover, trained volunteers, and reported longer battery life than last year.
Shorter: Students built a solar rover and proved it lasts longer.
Longer: Students in the robotics club built a solar-powered rover, documented each step, trained volunteers to drive it, and shared test data showing the new model outperforms last year's design.
```

### 8. Interpreting Token Length Effects

- **Shorter setting:** Lower `min_new_tokens` and `max_new_tokens` force a brief highlight at the expense of nuance.
- **Medium setting:** Balanced token bounds retain the main actions without unnecessary detail.
- **Longer setting:** Higher token limits allow the model to elaborate on supporting context while staying within the summarization scope.

---

*Document generated to summarize AI environment setup for PyTorch + CUDA 12.8 with RTX 5080, core Hugging Face workflows, key text classification pipelines, and text summarization techniques.*
