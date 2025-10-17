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

## Project: Siri Shortcut Maker

### Vision
- Let users describe their desired iPhone automation in plain language and instantly receive a ready-to-run Siri Shortcut.
- Streamline the delivery pipeline so validated shortcuts land on the target device with minimal manual intervention.

### Strategic Approach
1. **Prompt Understanding**
   - Build a structured schema that captures intents, required inputs, triggers, and device constraints from natural-language prompts.
   - Maintain a reusable library of shortcut patterns (e.g., text processing, device control, API integrations) for rapid composition.
2. **Shortcut Synthesis Engine**
   - Translate structured intents into Shortcut actions using Apple's shortcut serialization format or Shortcut Builder APIs.
   - Incorporate validation rules to catch missing permissions, unsupported actions, or conflicting triggers before deployment.
3. **Delivery Pipeline**
   - Implement authenticated communication with the user's iCloud account or device for pushing the finished shortcut package.
   - Provide staging previews (JSON + visual tree) so users can approve the automation before installation.
4. **Safety & Feedback Loop**
   - Track execution outcomes and user overrides to refine prompt parsing and action recommendations.
   - Log deployment history for rollback and compliance auditing.

### Flow Overview
```
+----------------------------+    +------------------------------+    +------------------------------+    +----------------------------+
|   Natural Language Prompt  | -> |   Intent & Pattern Matching   | -> |    Shortcut Construction     | -> |   Device Deployment & UX   |
+----------------------------+    +------------------------------+    +------------------------------+    +----------------------------+
```

### Backlog
| Priority | Task | Description | Owner | Status |
| --- | --- | --- | --- | --- |
| High | Define prompt schema | Document required fields, validation rules, and mapping to Shortcut actions. | TBD | Not Started |
| High | Build action pattern library | Catalogue reusable Shortcut templates for messaging, automation, and API calls. | TBD | Not Started |
| High | Prototype Shortcut generator | Convert schema instances into executable Shortcut files and surface validation errors. | TBD | Not Started |
| Medium | Implement device push service | Establish secure session with iCloud or a companion app to deliver shortcuts. | TBD | Not Started |
| Medium | Design approval preview UI | Render shortcut tree and permissions summary for user confirmation. | TBD | Not Started |
| Low | Telemetry dashboard | Visualize deployment history, success metrics, and rollback options. | TBD | Not Started |
| Low | Feedback ingestion pipeline | Collect user satisfaction data to refine parsing heuristics. | TBD | Not Started |

### Milestones
1. **Schema & Pattern Alpha** – Prompt schema, validation rules, and core shortcut templates published for internal testing.
2. **End-to-End Generator Beta** – Natural language prompt converted to a verified shortcut file with preview and error reporting.
3. **Device Deployment Launch** – Push-to-device workflow live with approval UI, telemetry, and rollback safeguards.

### Risks & Mitigations
- **Apple platform restrictions:** Monitor Shortcut API changes and maintain fallback export paths (e.g., share links) when direct pushes fail.
- **Permission misconfiguration:** Enforce approval gates that highlight sensitive actions (location, contacts, automation triggers) before installation.
- **Prompt ambiguity:** Provide interactive disambiguation questions when required fields are missing or conflicting.

### Open Questions
- Which Apple authentication method (personal automation profile vs. companion app) offers the most reliable shortcut delivery?
- How can we sandbox third-party API keys or secrets embedded in user prompts?
- What level of versioning is needed to support iterative shortcut refinements and rollbacks?

## Project: AI Agent Trainer Setup

### Vision
- Build an orchestration layer of cooperating AI agents that can translate a target training outcome into a full model training and fine-tuning plan.
- Provide guided recommendations for model families, datasets, and evaluation routines so teams can move from idea to validated model rapidly.

### Strategic Approach
1. **Outcome Translation**
   - Implement a requirements parser that converts desired product outcomes into concrete training objectives, metrics, and constraints.
2. **Model & Dataset Advisory**
   - Catalog available foundation models and public/private datasets; match them to objectives using capability scoring and licensing checks.
3. **Training Pipeline Automation**
   - Generate end-to-end training recipes that include environment setup, hyperparameter schedules, and resource allocation plans.
4. **Evaluation & Iteration**
   - Run validation suites against target benchmarks, analyze gaps, and trigger fine-tuning rounds until success criteria are met.
5. **Governance & Reporting**
   - Maintain audit logs, reproducibility metadata, and executive summaries for each training engagement.

### Flow Overview
```
+---------------------------+    +---------------------------+    +-------------------------------+    +------------------------+
|   Desired Outcome Brief   | -> |   Objective Translation   | -> |   Model & Dataset Selection   | -> |   Training Plan Draft  |
+---------------------------+    +---------------------------+    +-------------------------------+    +------------------------+
                                                                          |
                                                                          v
                                                           +-------------------------------+
                                                           |   Evaluation & Iteration Loop |
                                                           +-------------------------------+
                                                                          |
                                                                          v
                                                           +-------------------------------+
                                                           |   Final Report & Handoff      |
                                                           +-------------------------------+
```

### Backlog
| Priority | Task | Description | Owner | Status |
| --- | --- | --- | --- | --- |
| High | Define outcome schema | Specify required inputs (KPIs, constraints, deployment context) for the Trainer intake form. | TBD | Not Started |
| High | Build model advisory engine | Implement scoring logic that ranks candidate base models with reasoning traces. | TBD | Not Started |
| High | Automate training recipe generator | Produce reproducible training scripts and infrastructure manifests from selected configurations. | TBD | Not Started |
| Medium | Integrate dataset discovery | Connect to public catalogs and internal registries to surface compatible datasets with quality tags. | TBD | Not Started |
| Medium | Implement evaluation harness | Assemble metric dashboards and regression tests aligned with each objective. | TBD | Not Started |
| Low | Create governance dashboard | Visualize audit logs, approval checkpoints, and reproducibility metadata for stakeholders. | TBD | Not Started |
| Low | Develop continuous learning loop | Enable agents to incorporate post-deployment feedback into future training recommendations. | TBD | Not Started |

### Milestones
1. **Training Blueprint Alpha** – Outcome schema, advisory engine, and recipe generator produce a validated dry-run plan.
2. **Automated Fine-Tuning Beta** – Trainer executes training runs with evaluation harness feedback and iterative adjustments.
3. **Operational Launch** – Governance dashboard live with reproducibility reports and continuous learning loop activated.

### Risks & Mitigations
- **Model or data misuse:** Enforce licensing checks and red-team reviews before recommending restricted assets.
- **Resource overruns:** Introduce budget-aware scheduling and early-stop policies tied to outcome metrics.
- **Evaluation blind spots:** Maintain a library of domain-specific tests and solicit SME feedback to update coverage.

### Open Questions
- Which orchestration framework (e.g., LangGraph, CrewAI) best supports multi-agent planning with robust observability?
- How should the Trainer balance automated decision-making with human approvals for high-impact deployments?
- What metadata is required to guarantee reproducibility across different infrastructure providers?

## Project: Food Storage Management System

### Vision
- Provide a seamless way to capture new grocery items with minimal friction while maintaining an accurate, searchable household pantry inventory.
- Surface timely freshness insights so households can plan meals around items nearing expiration and reduce food waste.

### Strategic Approach
1. **Capture Interfaces**
   - Build mobile and Pi-camera capture flows that support barcode scans, object recognition, and manual overrides.
   - Add quick prompts for quantity, unit, and optional "opened" status when automation confidence is low.
2. **Recognition & Metadata Extraction**
   - Combine barcode lookup services with computer vision classification to identify items and map them to a product catalog.
   - Extract or prompt for expiration dates, packaging sizes, and storage categories; store unit conversions for consistent inventory math.
3. **Inventory Management Core**
   - Maintain a normalized item database with quantity tracking, opened/closed state, and consumption history.
   - Implement search, filtering, and criticality scoring rules that weigh product type, freshness, and spoilage risk.
4. **Insights & Notifications**
   - Build dashboards for expiring or expired items, grouped by criticality with suggested actions.
   - Offer notification channels (mobile push, email, or smart display cards) configurable by urgency thresholds.
5. **Consumption & Deletion Flows**
   - Support manual checkout, capture of empty packaging, or automated deduction from smart appliance signals.
   - Provide auditing tools to reconcile discrepancies and adjust quantities quickly.

### Flow Overview
```
+-------------------+    +-------------------------+    +-----------------------+    +---------------------------+
|   Item Capture    | -> |   Recognition & Lookup  | -> |   Inventory Update    | -> |   Alerts & Dashboards     |
+-------------------+    +-------------------------+    +-----------------------+    +---------------------------+
                              |                                       ^                       |
                              v                                       |                       v
                      +--------------------+                          |               +-----------------+
                      | Manual Overrides   | -------------------------+--------------- | Consumption Log |
                      +--------------------+                                          +-----------------+
```

### Backlog
| Priority | Task | Description | Owner | Status |
| --- | --- | --- | --- | --- |
| High | Define product catalog schema | Model item attributes (name, categories, shelf life factors, units) and barcode links. | TBD | Not Started |
| High | Prototype capture pipeline | Implement mobile and Pi-camera capture with barcode scanning and fallback photo uploads. | TBD | Not Started |
| High | Build expiration extraction service | Parse OCR results, packaging text, and user prompts to record best-before dates accurately. | TBD | Not Started |
| Medium | Develop criticality scoring engine | Weight item types by spoilage risk and elapsed time past expiry for dashboard sorting. | TBD | Not Started |
| Medium | Implement searchable inventory UI | Deliver filters by storage location, category, and opened status with responsive search. | TBD | Not Started |
| Medium | Add consumption workflows | Support manual checkout, empty-package recognition, and quantity adjustments with audit logs. | TBD | Not Started |
| Low | Integrate smart appliance hooks | Explore signals from connected fridges or scales to automate inventory updates. | TBD | Not Started |
| Low | Design notification preferences | Configure cadence, channels, and urgency thresholds for freshness alerts. | TBD | Not Started |

### Milestones
1. **Capture & Recognition Alpha** – End-to-end flow from photo or barcode scan to catalog match with manual override safety.
2. **Inventory Intelligence Beta** – Criticality scoring, expiring dashboard, and search UI running with real household data.
3. **Household Launch** – Notification preferences, consumption logging, and reconciliation tooling validated in daily use.

### Risks & Mitigations
- **Recognition inaccuracies:** Maintain human-in-the-loop overrides and confidence thresholds before committing inventory updates.
- **Data privacy concerns:** Store captures locally or encrypt cloud storage; provide deletion and export tools for household members.
- **User adoption friction:** Offer rapid entry shortcuts, reusable shopping lists, and optional voice commands to minimize manual typing.

### Open Questions
- Which regional barcode databases or grocery APIs offer the best coverage for local products?
- How should criticality scoring adapt to user-defined freshness tolerances or dietary restrictions?
- What signals (e.g., opened packaging photos, smart sensor data) are reliable enough for automated consumption logging?

### Future Expansion
- Dedicated recipe recommendation service that leverages current inventory, soon-to-expire items, household taste profiles, and available kitchen appliances.
- Integration hooks for spice tracking, utensil availability, and personalized meal-planning workflows as a follow-on project.

## Project: Codex CLI Autonomous Agent

### Vision
- Launch an autonomous development loop where a supervising agent orchestrates Codex CLI workers through every roadmap phase without manual prompting.
- Guarantee that generated code is validated and version-controlled so each completed work package unlocks the next stage confidently.

### Strategic Approach
1. **Supervising Orchestrator**
   - Parse the project roadmap, plan work packages, and assign them to Codex CLI workers with clear acceptance criteria.
   - Track execution status, trigger evaluations, and decide when to advance to the next phase.
2. **Codex CLI Worker Agent**
   - Translate work-package briefs into Codex CLI commands, manage local workspace context, and hand back diffs plus logs.
   - Commit validated outputs and request guidance whenever acceptance conditions are unmet.
3. **Evaluation & Feedback Loop**
   - Run automated tests, linting, and static checks to grade each Codex deliverable.
   - Feed test outcomes to the orchestrator for remediation or approval decisions.
4. **Roadmap & Status Management**
   - Maintain a structured roadmap definition, execution history, and audit logs for traceable progress reporting.

### Architecture Overview
```
+-----------------------------+      +-----------------------------+      +-----------------------------+
|       Roadmap Registry      | ---> |       Supervising Agent     | ---> |        Evaluation Harness   |
+-----------------------------+      +-------------+---------------+      +-----------------------------+
                                     ^             |
                                     |             | Dispatch
                                     |             v
                           +-------------------------------+
                           |        Codex CLI Worker       |
                           +-------------------------------+
                                     |
                                     v
                           +-------------------------------+
                           |      Repository & Reports     |
                           +-------------------------------+
```

### Backlog
| Priority | Task | Description | Owner | Status |
| --- | --- | --- | --- | --- |
| High | Define system requirements | Consolidate objectives, agent responsibilities, and hand-off rules into a baseline spec. | TBD | Not Started |
| High | Model roadmap schema | Choose a machine-readable roadmap format and build parser utilities for work-package retrieval. | TBD | Not Started |
| High | Implement supervising orchestrator | Develop planning, dispatch, and monitoring logic with hooks for external oversight models. | TBD | Not Started |
| High | Build Codex CLI worker wrapper | Encapsulate Codex CLI execution, workspace preparation, and result collation. | TBD | Not Started |
| Medium | Establish evaluation pipeline | Automate unit tests, linting, and result grading with standardized feedback channels. | TBD | Not Started |
| Medium | Add status persistence & reporting | Store execution history and surface progress dashboards or summaries. | TBD | Not Started |
| Low | Prototype operator controls | Create pause/resume and override interfaces for human supervisors. | TBD | Not Started |

### Milestones
1. **Orchestrator Foundations** – Roadmap parser, supervising agent skeleton, and Codex worker integration validated on sample tasks.
2. **Autonomous Delivery Loop** – Evaluation harness gating work-package promotion with automated remediation triggers.
3. **Operational Readiness** – Reporting, operator controls, and audit logs supporting continuous multi-phase delivery.

### Risks & Mitigations
- **Unreliable test coverage:** Invest in comprehensive automated checks before promoting work packages; add manual gates for high-impact phases.
- **Context drift between agents:** Normalize workspace state and share explicit briefs to keep Codex outputs aligned with roadmap intent.
- **API or rate limitations:** Cache prompts, batch executions where possible, and fall back to queued retries when Codex CLI is throttled.

### Open Questions
- Which supervising model or framework offers the best balance of planning depth and operational cost?
- How should the system prioritize between retrying failed work packages and escalating them for human review?
- What telemetry is required to audit Codex CLI interactions for compliance and reproducibility?

