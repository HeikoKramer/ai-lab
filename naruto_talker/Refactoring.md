# Refactoring & UI Redesign Plan

## 1. Introduction: AS-IS Architecture (The "Wrapper" Approach)
Currently, our application acts as a "Translator" or "Wrapper". It defines high-level, abstract user inputs (like sliders for "Pitch" or "Speed", checkboxes for "Gender") and attempts to translate them into the specific prompt strings expected by the AI models.

**Current Data Flow:**
1.  **User Input**: `Gender="Female"`, `Pitch=0.4` (Slider), `Speed=1.1`, `Emotion="Happy"`.
2.  **App Logic (`app.py`)**:
    *   Constructs SDXL Prompt: *"portrait close-up of [Visual Description], [Gender]"*
    *   Constructs TTS Prompt: *"Eine fröhliche, energetische [Gender] Stimme, hohe tonlage..."* (Hardcoded string concatenation).
3.  **Model Execution**: Models receive these constructed strings.

**The Problem:**
*   **Lossy Translation**: A "Pitch" slider doesn't really exist in Qwen-TTS. We are just appending text like "high pitched". This is imprecise.
*   **Hidden Capabilities**: Models like Qwen-TTS supports Voice Cloning (Reference Audio) or very specific natural language instructions ("A raspy, smoking addiction voice"), which our current UI makes impossible to use.
*   **Black Box**: The user doesn't see what is actually sent to the model.

---

## 2. TO-BE Architecture: The "Character Sheet" (Direct Control)
We will move to a "Direct Control" architecture. The UI will expose the models' native capabilities directly. The "Magic Glue" is removed in favor of "Power User Control".

### 2.1 Backend Refactoring Strategy
*   **Language Handling**:
    *   **SDXL (Image)**: Native language is **English**. German prompts often result in worse quality or ignored keywords.
        *   *Strategy*: We will implement a hidden "DeepL / Translation" helper, OR simply instruct the user "English works best for visuals". Given the requested "Native" feel, we should probably allow German input but auto-translate it silently to English for the Visual Prompt only.
    *   **Qwen-TTS (Audio)**: Native language is **Multilingual**. It understands German descriptions ("Eine raue Stimme"), but often English descriptions work better for *style* definition even if the output language is German.
        *   *Strategy*: Pass the user's input raw. If they write German, Qwen usually handles it.
*   **Routing**:
    *   Removal of `Pitch`/`Speed` sliders.
    *   New Input routes: `Voice Text Prompt` and `Reference Audio Path`.

---

## 3. UI Design Proposal: "The Character Studio"
The interface will be reorganized into a single "Character Card" view.

### **Section A: Visual Identity (The Look)**
*Goal: Define the SDXL-Turbo Inputs.*

1.  **Character Portrait Prompt** (Text Area, 3 lines)
    *   *Label*: "Visuelle Beschreibung (Visual Prompt)"
    *   *Explanation*: Beschreiben Sie genau, wie der Charakter aussehen soll. Englisch funktioniert am besten.
    *   *Example*: `portrait of a cyberpunk hacker, neon lights, purple hair, detailed eyes, 8k, close up`
    *   *Impact*: Direct input to SDXL. Controls the generated image.
2.  **Negative Prompt** (Text Input, Collapsible)
    *   *Label*: "Was nicht enthalten sein soll (Negative)"
    *   *Explanation*: Elemente, die vermieden werden sollen.
    *   *Example*: `blurry, bad anatomy, three eyes, painting, drawing`
    *   *Impact*: Filters out unwanted artifacts.
3.  **Seed** (Number Input)
    *   *Label*: "Portrait Seed"
    *   *Explanation*: Festlegung des "Zufalls". Gleicher Seed + Gleicher Prompt = Gleiches Bild. Wichtig für Konsistenz.
    *   *Default*: `-1` (Random) or `42`.

### **Section B: Voice Identity (The Sound)**
*Goal: Define the Qwen/VoiceDesign Inputs without 'Interpretation'.*

1.  **Voice Design Prompt** (Text Area, 2 lines)
    *   *Label*: "Stimm-Beschreibung (Voice Prompt)"
    *   *Explanation*: Beschreiben Sie die Stimme mit natürlichen Adjektiven. Funktioniert wie ChatGPT für Stimmen.
    *   *Example*: `Eine tiefe, rauchige Stimme eines alten Erzählers, ruhig und langsam.` OR `A high-pitched, energetic anime girl voice, very excited.`
    *   *Impact*: Replaces the "Pitch/Gender" sliders. This string is sent DIRECTLY to Qwen-TTS.
2.  **Reference Audio (Voice Cloning)** (Audio Upload / Mic) - *New Feature!*
    *   *Label*: "Stimm-Referenz (Optional)"
    *   *Explanation*: Laden Sie eine 5-10 sekündige Aufnahme einer Stimme hoch, die geklont werden soll. Überschreibt die "Stimm-Beschreibung".
    *   *Impact*: Unlocks Qwen's specific "Voice Cloning" mode. Extremely powerful for consistency.

### **Section C: Action (The Script)**
*Goal: What happens in this specific take.*

1.  **Dialogue Script** (Text Area)
    *   *Label*: "Text (Was wird gesagt?)"
    *   *Explanation*: Der Text, den der Charakter sprechen soll.
    *   *Example*: `Hallo! Ich bin der neue Avatar. Wie findest du meine Stimme?`
2.  **Output Language** (Dropdown)
    *   *Label*: "Sprache (Language)"
    *   *Values*: `German`, `English`, `French`, `Japanese`...
    *   *Impact*: Tells the TTS model which phonemes to generate. Independent of the Voice Description! (You can have a "French Voice" speaking "German").

### **Section D: Technical Tuning (Advanced)**
*Goal: Debugging and Fine-Tuning parameters.*

1.  **LipSync Padding (BBox Shift)** (Slider: -20 to +20)
    *   *Label*: "Mund-Position Korrektur (Offset)"
    *   *Explanation*: Verschiebt den erkannten Mund-Bereich nach oben/unten.
    *   *Why*: Manchmal erkennt MuseTalk den Mund zu tief (Kinn) oder zu hoch (Nase).
    *   *Default*: `0`.
2.  **Restoration Strength** (Slider: 0.1 to 1.0)
    *   *Label*: "Gesichtskorrektur Stärke"
    *   *Explanation*: Wie stark soll der "Mouth Super-Resolution" Filter arbeiten?
    *   *Why*: Zu stark = künstlich. Zu schwach = unscharf.
    *   *Default*: `1.0` (Current hardcoded setting).

---

## 4. Implementation Steps

1.  **Refactor `app.py` UI**: 
    *   Remove `gr.Slider` for pitch/speed.
    *   Add `gr.Audio(sources=['upload', 'microphone'])`.
    *   Add Translation Helper function (Optional, for Visual Prompt).
2.  **Refactor `src/tts.py`**:
    *   Update `generate` method to accept `ref_audio_path`.
    *   Logic: IF `ref_audio_path` exists -> Call `model.generate_voice_clone`. ELSE -> Call `model.generate_voice_design`.
3.  **Refactor `src/lipsync.py`**:
    *   Expose `bbox_shift` as an argument in `inference()`.

## 5. Summary
This design moves responsibility from the *Developer* (us guessing what "Happy" means) to the *User* (describing exactly what they want). It aligns perfectly with the capabilities of Generative AI.
