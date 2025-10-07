# AI Projects Backlog

## Project: Interactive Audio Biography "Oma erzählt"

### Vision
- Preserve Oma's authentic storytelling voice so family members can engage with her memories in an interactive, emotionally resonant experience.
- Allow listeners to ask questions and receive responses generated from real anecdotes, delivered in Oma's familiar speech patterns.

### Strategic Approach
1. **Data Preparation**
   - Gather all high-quality studio recordings (5–6 hours) and segment them into 3–12 second clips.
   - Annotate clips with topics, emotions, and contextual metadata; normalize audio to consistent mono, 22 kHz format.
2. **Voice Modeling**
   - Train a modern TTS/voice-cloning architecture (e.g., Coqui XTTS v2 or VITS) using the curated dataset.
   - Capture Oma's timbre, speech rhythm, and prosody to produce a natural, narrative delivery rather than an interview cadence.
3. **Narrative Structuring**
   - Break interviews into thematic blocks (childhood, war experiences, family life, anecdotes) with cross-referenced story notes.
   - Build an enriched textual corpus that includes recurring phrases and signature expressions for more faithful responses.
4. **Interactive Layer**
   - Design an AI dialogue system that interprets user prompts, maps them to relevant story segments, and crafts empathetic replies.
   - Ensure responses draw from verified memories; include guardrails to avoid fabrications and maintain respectful tone.
5. **Experience Enhancement**
   - Implement background ambience, subtle accent cues, and mood-based variations to reinforce immersion.
   - Add pacing controls so users can adjust tempo while keeping Oma's speech natural.

### Backlog
| Priority | Task | Description | Owner | Status |
| --- | --- | --- | --- | --- |
| High | Collect and segment raw audio | Consolidate existing recordings, apply noise reduction, and export normalized segments. | TBD | Not Started |
| High | Draft metadata schema | Define tagging structure for themes, emotions, and timeline references. | TBD | Not Started |
| High | Select voice model stack | Evaluate Coqui XTTS v2 vs. VITS for multilingual nuance and fine-tuning support. | TBD | Not Started |
| Medium | Curate narrative knowledge base | Transcribe interviews, identify recurring stories, and map follow-up questions. | TBD | Not Started |
| Medium | Prototype dialogue manager | Build prompt-routing logic that matches user queries to story clusters. | TBD | Not Started |
| Medium | Guardrail policy design | Document constraints to prevent speculative or historically inaccurate outputs. | TBD | Not Started |
| Low | Audio ambience exploration | Test gentle background soundscapes aligned with memory themes. | TBD | Not Started |
| Low | User interface concept | Sketch interactive experience flow for web deployment. | TBD | Not Started |
| Low | Accessibility review | Plan captions, transcripts, and alternative interaction modes. | TBD | Not Started |

### Milestones
1. **MVP Voice Prototype** – Deliver a short interactive demo with at least three validated story paths using Oma's cloned voice.
2. **Narrative Knowledge Graph** – Complete tagging and cross-referencing of all interview material for semantic retrieval.
3. **Public Beta Experience** – Launch updated oma.heikokraemer.de with interactive storytelling, feedback capture, and analytics.

### Risks & Mitigations
- **Ethical authenticity:** Maintain transparency that the experience is an homage, not a live conversation; include disclaimers and consent records.
- **Model hallucination:** Use retrieval-augmented generation and human review for new prompts before public release.
- **Emotional sensitivity:** Offer content warnings for potentially triggering stories and include user-controlled pacing.

### Open Questions
- What governance model will oversee updates to Oma's digital voice over time?
- Which hosting stack best supports low-latency, on-demand synthesis without compromising privacy?
- How can family members contribute new memories while preserving narrative coherence?

