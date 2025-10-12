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

## Project: NeoMutt Agent Automation

### Vision
- Transform NeoMutt into an intelligent inbox that anticipates time-sensitive actions and surfaces them proactively.
- Keep the user in control by logging each automated step and providing simple override mechanisms.

### Strategic Approach
1. **Workflow Research**
   - Catalogue high-value email scenarios (verification codes, account activations, calendar invites) and define desired automations.
   - Map privacy expectations, clipboard timeouts, and notification preferences.
2. **Agent Architecture**
   - Select an extensible framework (e.g., Python daemon with IMAP hooks) that can observe NeoMutt events without blocking the UI.
   - Define pluggable action modules for clipboard updates, link launching, and scripted replies.
3. **Security & Safety**
   - Introduce allow-lists, confirmation prompts for risky actions, and encrypted clipboard handling.
   - Log every automated step with timestamps for auditability.
4. **User Experience**
   - Surface contextual status updates inside NeoMutt (e.g., sidebar widget) and optional desktop notifications.
   - Provide pause/resume controls and quick command bindings.

### Backlog
| Priority | Task | Description | Owner | Status |
| --- | --- | --- | --- | --- |
| High | Define automation playbook | Document supported email scenarios, triggers, and required guardrails. | TBD | Not Started |
| High | Prototype IMAP listener | Build a daemon that monitors new messages and emits structured events to the agent. | TBD | Not Started |
| High | Implement OTP extractor | Detect one-time passwords, copy them to the clipboard with automatic expiration, and notify the user. | TBD | Not Started |
| Medium | Verification link handler | Parse trusted senders and open confirmation links in the default browser after user approval. | TBD | Not Started |
| Medium | Activity logging dashboard | Record agent actions and display them inside NeoMutt or a companion TUI panel. | TBD | Not Started |
| Low | Scripted reply templates | Add quick-reply actions for common responses that can be triggered via shortcuts. | TBD | Not Started |
| Low | Pause/resume command bindings | Map agent control toggles to NeoMutt keybindings for rapid overrides. | TBD | Not Started |

### Milestones
1. **Agent Foundations** – Event listener and OTP automation running locally with audit logs.
2. **Trusted Action Suite** – Verified support for clipboard handling, link approvals, and notification workflows.
3. **User-Controlled Launch** – Polished configuration, documentation, and safeguards ready for daily use.

### Risks & Mitigations
- **Security exposure:** Restrict automation to allow-listed senders and enforce short clipboard lifetimes for sensitive data.
- **Workflow disruption:** Provide immediate pause controls and verbose logging to rebuild trust when automations misfire.
- **Compliance concerns:** Ensure the agent respects corporate policies by allowing per-account configuration and opt-outs.

### Open Questions
- Which NeoMutt integration path (post-hook scripts vs. mailbox polling) offers the best balance of responsiveness and stability?
- How should clipboard expiration and notifications behave across desktop environments?
- What approval flow is needed before the agent opens external links automatically?

