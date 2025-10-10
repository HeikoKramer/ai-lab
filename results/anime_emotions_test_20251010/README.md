# Anime Emotion Test — First Seeding Run (2025-10-10)

This document records our very first AI Lab test run with the [`test_seeds.py`](../../scripts/test_seeds.py) script. For the broader experiment description, see the [test_seeds documentation](../../docs/test_seeds.md). No fine-tuning or model adjustments were applied—only a raw generation of 20 anime-style facial emotions using a consistent seed-based character prompt.

Each image shows the same base character with a distinct emotional expression to confirm that the seed-cycling logic works end-to-end.

---

## Emotions Generated
- 01_happy
- 02_sad
- 03_angry
- 04_surprised
- 05_confused
- 06_scared
- 07_excited
- 08_bored
- 09_sleepy
- 10_crying
- 11_smiling
- 12_blushing
- 13_embarrassed
- 14_determined
- 15_laughing
- 16_nervous
- 17_shocked
- 18_serious
- 19_thinking
- 20_shy

---

### Notes
- Model: default lab-on AI image generator
- Prompt: consistent anime character with varying emotions
- Seed logic: basic fixed-seed iteration (no fine-tuning)
- Run date: 2025-10-10

---

### Emotion Gallery

| ![Happy](./01_happy.png)<br/>Happy | ![Sad](./02_sad.png)<br/>Sad | ![Angry](./03_angry.png)<br/>Angry | ![Surprised](./04_surprised.png)<br/>Surprised | ![Confused](./05_confused.png)<br/>Confused |
| --- | --- | --- | --- | --- |
| ![Scared](./06_scared.png)<br/>Scared | ![Excited](./07_excited.png)<br/>Excited | ![Bored](./08_bored.png)<br/>Bored | ![Sleepy](./09_sleepy.png)<br/>Sleepy | ![Crying](./10_crying.png)<br/>Crying |
| ![Smiling](./11_smiling.png)<br/>Smiling | ![Blushing](./12_blushing.png)<br/>Blushing | ![Embarrassed](./13_embarrassed.png)<br/>Embarrassed | ![Determined](./14_determined.png)<br/>Determined | ![Laughing](./15_laughing.png)<br/>Laughing |
| ![Nervous](./16_nervous.png)<br/>Nervous | ![Shocked](./17_shocked.png)<br/>Shocked | ![Serious](./18_serious.png)<br/>Serious | ![Thinking](./19_thinking.png)<br/>Thinking | ![Shy](./20_shy.png)<br/>Shy |

