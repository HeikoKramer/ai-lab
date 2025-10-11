# AGENTS Template

## Running System Description
- Primary GPU: NVIDIA RTX 5080 with TensorFloat-32 optimizations aligned to CUDA 12.8.
- CUDA 12.8 toolchain with matching drivers is available across all runtimes.
- Python environments are provisioned with virtual-environment support for isolated workflows.
- Core deep learning stack ships with PyTorch, torchvision, and torchaudio wheels built against CUDA 12.8.
- Hugging Face tooling (transformers, diffusers, accelerate, safetensors, sentencepiece) is preinstalled for model development and deployment.
- Continuous environment health checks ensure CUDA 12.8 is detected and GPU execution remains enabled; CPU fallbacks are treated as configuration issues.

## General Repository Rules
- Use English across code comments, documentation, commit messages, and configuration files; omit emojis unless a task explicitly requests them or they are part of an established documentation pattern (e.g., navigation aids in a README) that should be preserved.
- Keep operational or process notes intended for automation confined to `AGENTS.md`; do not surface them in public-facing documentation.
- Avoid teaser or placeholder sentences that promise future content.
- Apply these guidelines repository-wide; allow more specific `AGENTS.md` files in subdirectories to override details when necessary.
- When documenting APIs, workflows, or library collections, add cheat sheet style summaries that highlight the most relevant commands, filters, or functions.
- Express multi-step processes with ASCII flowcharts when they improve clarity. Arrange nodes left-to-right when possible, otherwise top-to-bottom, and center the text within each bordered node.
- Maintain backlog documents in a vision → strategy → prioritized tasks sequence and update them as progress evolves.
- When a repository intentionally uses emojis within canonical docs like the README to improve scannability, retain them and avoid stripping the expressive formatting.

## Coding Standards
- Prefer GPU-accelerated code paths that capitalize on the RTX 5080; explicitly configure libraries to target CUDA 12.8.
- Structure scripts to detect environment prerequisites early (driver, CUDA, Python package versions) and fail fast with actionable messages.
- Keep repository-specific automation details separated from reusable code so the templates remain portable between projects.

## Testing Standards
- Automate GPU availability checks (e.g., `torch.cuda.is_available()`) within test suites to prevent regressions that fall back to CPU execution.
- Record the exact commands required to validate CUDA 12.8 compatibility and PyTorch functionality; keep them synchronized with the supported hardware stack.
- Treat environment validation as part of continuous integration: capture logs that confirm driver versions, CUDA toolkit revisions, and GPU memory visibility.

## Documentation Standards
- Use a concise, action-oriented tone for procedural chapters; treat the primary environment setup guide as the stylistic baseline for future sections.
- Break large chapters into meaningful sub-sections when the material benefits from segmentation.
- Maintain an always-current table of contents that links to major sections within long-form documentation.
- Append new knowledge chronologically instead of rearranging established sections unless a deliberate restructuring is requested.
- Whenever similar topics already exist, cross-link them with one-sentence pointers so readers can navigate quickly between related content.
- Emphasize widely applicable best practices by highlighting them (e.g., bold text) instead of leaving them implied.
- When documenting technology-specific domains (such as model families or third-party services), list the widely adopted options with succinct notes on use cases, strengths, and limitations, and reference existing blueprint-style summaries rather than duplicating them.
