# Refactoring & Translation Scope of Work (SoW)

This document outlines the step-by-step plan to refactor the Application UI/Backend and integrate automated translation services.

**Goal:** Transform the application from a "Wrapper" to a "Direct Control" architecture, while ensuring user ease-of-use through automated translation of German inputs into English (for Models) or the Target Language (for Speech).

---

## Phase 1: Infrastructure & Translation Setup

**Objective:** Add specific translation capabilities to the project.
**Dependencies:** `deep_translator` (needs installation).

### Step 1.1: Install Dependencies
*   **Action:** Add `deep_translator` to `requirements.txt` (or install via pip).
*   **Acceptance Criteria:** `pip show deep_translator` returns valid info.

### Step 1.2: Translation Utility
*   **Action:** Create `src/translation.py`.
*   **Features:**
    *   Function `translate_text(text, target_lang="en")`.
    *   Error handling (fallback to original text if offline/error).
*   **Acceptance Criteria:**
    *   Input: "Eine raue Stimme" -> Output: "A rough voice" (approx).
*   **Verification Method:**
    *   Create script `tests/test_translation_unit.py` that calls the function and asserts output is English-like.

---

## Phase 2: UI Redesign (Frontend/Gradio)

**Objective:** Complete overhaul of `app.py` UI layout.

### Step 2.1: Remove Legacy Controls
*   **Action:** Remove `Pitch` (Slider), `Speed` (Slider), `Gender` (Radio), `Emotion` (Checkbox/Dropdown).
*   **Acceptance Criteria:** App launches without errors; old components are gone.

### Step 2.2: Implement "Character Studio" Layout
*   **Action:** Add new Grouped Components:
    *   **Visual Identity:** `Visual Prompt` (TextArea), `Negative Prompt` (Input), `Seed` (Number).
    *   **Voice Identity:** `Voice Prompt` (TextArea), `Reference Audio` (Audio Upload + Mic).
    *   **Action:** `Dialogue Script` (TextArea), `Output Language` (Dropdown).
    *   **Technical:** `LipSync Offset` (Slider), `Restoration Strength` (Slider).
*   **Acceptance Criteria:**
    *   All new components are visible.
    *   `Reference Audio` allows upload and recording.
*   **Verification Method:**
    *   Visual User Verification (Screenshot).
    *   Headless check: Parse `app.ui` object to confirm component IDs exist.

---

## Phase 3: Backend Refactoring (Wiring & Logic)

**Objective:** Update pipelines to accept new raw inputs and handle routing.

### Step 3.1: Translation Integration in `app.py`
*   **Action:** Integrate `src/translation.py` into the `generate_btn` click event.
*   **Logic:**
    *   `Visual Prompt` (DE) -> **Translate to EN** -> SDXL.
    *   `Voice Prompt` (DE) -> **Translate to EN** -> Qwen-TTS.
    *   `Dialogue Script` (Any) -> **Translate to [Output Language]** -> TTS.
*   **Acceptance Criteria:**
    *   Logs show: "Translating 'Eine Katze' to 'A cat'".
    *   Models receive English prompts.

### Step 3.2: TTS Pipeline Upgrade (`src/tts.py`)
*   **Action:** Update `generate()` signature.
*   **Logic:**
    *   Support `voice_design` (Prompt-based) AND `voice_cloning` (Audio-based).
    *   If `ref_audio` is provided, prioritize it over `voice_prompt`.
*   **Acceptance Criteria:**
    *   Passing `voice_prompt` uses `generate_voice_design`.
    *   Passing `ref_audio` uses `generate_voice_clone`.
*   **Verification Method:**
    *   `tests/test_tts_routing.py`: Mock the model and verify which internal method is called based on inputs.

### Step 3.3: LipSync & Restoration Upgrade
*   **Action:** Pass `bbox_shift` and `restoration_strength` from App -> LipSync Pipeline -> Restoration.
*   **Acceptance Criteria:**
    *   `bbox_shift` changes the coordinate logic in `lipsync.py`.
    *   `restoration_strength` is passed to the enhancer.
*   **Verification Method:**
    *   Dry run checking logs for parameter values.

---

## Phase 4: Verification & Final Polish

**Objective:** End-to-End testing.

### Step 4.1: End-to-End Test Script
*   **Action:** Update `run_full_pipeline.py` to match new arguments.
*   **Acceptance Criteria:** Script runs from start to finish producing `final_result.mp4`.
*   **Verification Method:**
    *   Execute `run_full_pipeline.py`.
    *   Check `final_result.mp4` properties.

---

## Acceptance Criteria for Success (DoD)
1.  **UI:** New "Character Studio" layout is active.
2.  **Input:** User can type German in all fields.
3.  **Process:** Logs confirm translation to English for system prompts.
4.  **Output:** Video is generated with correct Voice Style (cloned or designed) and correct Visuals.
