# Repository Guidelines

- **Language:** All repository content, including documentation, code comments, commit messages, and configuration entries, must be written in English. Do not use emojis unless explicitly requested.
- **Operational notes stay internal:** Do not copy process directions or instructions addressed to AI agents into the general documentation. Keep those details only inside `AGENTS.md` files.
- **No teaser sentences:** Do not include forward-looking placeholder lines such as "More hands-on agent build recipes will follow" in user-facing documentation. These internal planning cues belong only in `AGENTS.md` files.
- **Scope:** These guidelines apply to the entire repository unless superseded by a more specific `AGENTS.md` in a subdirectory.
- **Cheatsheet summaries:** When documenting APIs, database queries, libraries, or notable function groups, include cheat sheet-style summaries of the essential filters or functions relevant to the immediate use case whenever it adds clarity.
- **Flowcharts:** Use flowcharts whenever they clarify multi-step processes. Arrange steps from left to right when space allows; otherwise, stack steps vertically. Always enclose each step in a bordered container so the chart reads visually as a flow diagram. Render flowcharts as ASCII art and center the text inside each node for consistent readability.
- **`ai-projects_backlog.md`:** Keep all backlog entries in English, structuring each project with a clear vision, strategy, and prioritized tasks.
- **Working with Hugging Face topics:** For every topic documented under the "Working with Hugging Face" chapter in `ai-notes.md`, verify whether specialized and widely adopted Hugging Face models exist for the topic's primary use cases. When they do, append a concise overview to the chapter that lists each model with its core use case, strengths, and limitations.
- **Chapter structure in `ai-notes.md`:** When adding a new chapter, organize the content into meaningful sub-chapters instead of leaving everything in a single block whenever the material can be segmented for clarity.

## File-Specific Expectations

### README.md
- Treat this file as the table of contents for the repository.
- Provide concise descriptions of important locations, projects, and documentation, each with links to the relevant files or folders.
- Keep the overview up to date when new major areas are added.

### ai-notes.md
- Document the current understanding and key explanations from the AI development journey.
- Focus on insights, commands, and verification steps rather than upcoming tasks.
- Maintain a table of contents that links to the major sections.
- Treat the "AI Environment Setup for PyTorch + CUDA 12.8 (RTX 5080)" chapter as the stylistic reference for future additions; new chapters should match its concise, action-oriented tone and structure.
- Refer to the entire "AI Environment Setup for PyTorch + CUDA 12.8 (RTX 5080)" chapter when a blueprint-style summary is needed; do not maintain a standalone blueprint chapter in `ai-notes.md`.
- Append new documentation to the end of `ai-notes.md`. Do not reorder or reposition existing chapters unless a separate request specifically asks for a restructuring.
- Before adding new content, check whether the topic already exists elsewhere in the document. If it does, add a cross-link with a one-sentence pointer (e.g., "More details and examples in this chapter") so readers can navigate between related sections.
- Highlight general rules or best-practice guidance visually (for example, by using bold text in a dedicated sentence). Whenever a well-known general rule or best practice is relevant, incorporate it into the documentation rather than leaving it implied.

### next-steps.md
- Record actionable next steps for the overall repository.
- Keep entries current and remove items once completed or no longer relevant.
- Mark completed items using Markdown strikethrough (e.g., `~~Task~~`). Items 1–3 are already completed and should remain crossed out.
