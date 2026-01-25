# Spezifikation: Mau Mau - Tokyo Night Grid

### Visuelle Identität
- **Farbschema:** Tokyo Night (Hintergrund: #1a1b26, Akzente: #7dcfff (Cyan) & #f7768e (Magenta)).
- **Karten-Stil:** ASCII-Art-Symbole, Ubuntu Mono Font, Glow-Effekte.

### Spielmechanik (Classic Mau Mau)
- 32 Karten (7 bis Ass).
- **Sonderregeln:**
  - 7: Nächster Spieler zieht 2 Karten.
  - 8: Nächster Spieler setzt aus.
  - Bube: Wünscht sich eine Farbe (darf auf fast alles gelegt werden, außer Bube auf Bube).
  - Mau/Mau Mau: Pflichtankündigung bei letzter/vorletzter Karte.

### KI-Architektur (16GB VRAM Limit)
- **Logic & Referee LLM:** Mistral NeMo 12B (Q4 Quantisierung).
- **Audio/Voice:** Qwen2-Audio (TTS).
- **Funktion:** Die KI ist Gegner und Schiedsrichter zugleich. Sie kommentiert, erklärt Regeln und weist illegale Züge des Spielers verbal zurück.

### Technischer Stack
- Frontend: HTML5/JS (Vanilla ES Modules für einfache lokale Ausführung ohne komplexe Build-Chain empfohlen, Komponentenstruktur via Klassen).
- Backend (Local AI): API-Calls an Localhost (Ollama/vLLM).
