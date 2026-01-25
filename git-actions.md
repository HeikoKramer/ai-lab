# Git Restoration Action Plan

This document outlines the steps to restore the `ai-lab` repository structure to match commit `195cb67` for specific files, while preserving the new progress in `naruto_talker` and `qwen3_tts_demo`.

## Prerequisites
- [x] `.gitignore` has been updated to exclude `debug_outputs`, `venv`, and other build artifacts.

## Restoration Steps

The following steps should be executed in the repository root: `/home/heiko/projects/ai-lab`.

### 1. Fetch Remote Updates
Ensure we have the latest information from the remote repository (though we are targeting a specific commit).
```bash
git fetch origin
```

### 2. Restore Specific Files
Restore the critical files `ai-notes.md`, `ai-projects_backlog.md`, and `AGENTS.md` to their state at commit `195cb67`.
```bash
git checkout 195cb67 -- ai-notes.md ai-projects_backlog.md AGENTS.md
```
> **Note:** This command immediately overwrites these files in your working directory with the versions from the specified commit and stages them for commit.

### 3. Stage New Projects
Stage the new directories that we want to preserve and add to the repository.
```bash
git add naruto_talker qwen3_tts_demo
```

### 4. Stage .gitignore Update
Ensure the updated `.gitignore` is included.
```bash
git add .gitignore
```

### 5. Review Status
Check the status to ensure:
- `ai-notes.md`, `ai-projects_backlog.md`, `AGENTS.md` are staged (modified/new).
- `naruto_talker` and `qwen3_tts_demo` content is staged (new files).
- `.gitignore` is staged (modified).
- Ignored files (debug outputs) are NOT staged.

```bash
git status
```

### 6. Commit
Commit the changes with a descriptive message.
```bash
git commit -m "Restore core structure from 195cb67 and add naruto_talker & qwen3_tts_demo"
```

## Post-Check
After committing, you should verify:
- `AGENTS.md` and `ai-projects_backlog.md` exist and have content.
- `ai-notes.md` is reverted to the previous version.
- `naruto_talker` and `qwen3_tts_demo` are part of the repo.
