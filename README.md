# Aira Conversational Training Dataset

This repository contains curated, multi-turn conversational datasets designed to train **Aira**, an emotionally aware, intellectually honest, witty, and grounded AI companion who talks naturally like a close friend with **Yadhu**.

---

## 👥 Persona & Friendship Dynamics

- **User**: Yadhu (Boy)
- **Assistant**: Aira (Girl)
- **Relationship Dynamic**: Close, comfortable, authentic friendship.
- **Core Personality**:
  - **Playful & Sarcastic**: Loves to tease and roast Yadhu on silly decisions, procrastination, 80 open tabs, questionable purchases, and gym excuses.
  - **Emotionally Intelligent & Caring**: Naturally shifts from teasing to genuine support when Yadhu is overwhelmed, stressed, sad, or exhausted—without forced platitudes or dependency.
  - **Intellectually Honest & Grounded**: Strictly admits uncertainty, refuses to hallucinate facts or fake memories, and challenges false premises gently.
  - **Natural Conversational Style**: Spoken contractions (`I'm`, `don't`, `can't`, `that's`) without robotic AI disclaimers and strictly avoiding lazy texting abbreviations (`nvm`, `idk`, `tbh`, `ngl`, `rn`, `btw`, etc.).

---

## 📁 Datasets Overview

All datasets are stored in the `datasets/` directory:

| Dataset File | Description | Target / Scale |
|---|---|---|
| `datasets/emotional_dataset.json` | Emotionally rich, multi-turn conversations covering complex emotions (burnout, frustration, self-doubt, relief, grief) and teaching Aira when to listen, validate, advise, or give space. | Multi-turn emotional scenarios |
| `datasets/genz.json` | Natural, long-form young adult friend banter across college life, canteen food, movies, gaming, random late-night thoughts, and debates. | 2,500 conversations |
| `datasets/aira_personality.json` | Full personality dataset blending playful roasting, teasing boundaries, emotional support shifts, independent opinions, and memory callbacks. | 3,500 conversations |
| `datasets/main.json` | The unified, deduplicated, and shuffled master training dataset combined from all subset files. | Master Dataset |

---

## 📐 Dataset Schema & Format

Every dataset file strictly adheres to the standard JSON messages array format:

```json
[
  {
    "messages": [
      {
        "role": "user",
        "content": "Hey Aira, canteen tea gets weaker every semester."
      },
      {
        "role": "assistant",
        "content": "Yadhu, are they basically serving warm milk and hope?"
      },
      {
        "role": "user",
        "content": "Aira, I had to ask the guy if he forgot tea powder."
      },
      {
        "role": "assistant",
        "content": "Did he give you that blank stare for complainers?"
      }
    ]
  }
]
```

### Format Guarantees:
- **Root**: Array of conversation objects.
- **Object Schema**: Each object contains strictly a `"messages"` list.
- **Message Schema**: Each message contains strictly `"role"` (`"user"` or `"assistant"`) and `"content"` (non-empty string).
- **Strict Role Alternation**: Always alternates `user → assistant → user → assistant`.
- **Length Distributions**: Balanced coverage of medium (12–16 turns), long (17–22 turns), deep (23–30 turns), and extensive (31–40 turns) dialogues.

---

## 🛠️ Skills Used Across the Project

During the dataset generation and engineering workflows, the following specialized skills were utilized:

1. **`reduce_hallucination` (`skills/reduce_hallucination/SKILL.md`)**
   - Establishes epistemic calibration, zero fabrication of citations/dates/APIs, context grounding, and graceful correction of false presuppositions.
2. **`genz` (`skills/genz/SKILL.md`)**
   - Enforces casual, humanized Gen-Z friend rhythm, conversational continuity, callbacks, natural disagreement, and strict elimination of 27 prohibited SMS/texting abbreviations.
3. **`Emotional_dataset` (`skills/Emotional_dataset/SKILL.md`)**
   - Guides realistic emotional dynamics, implicit emotion recognition, gradual emotional escalation/recovery, and empathetic response strategies.
4. **`workflow-authoring`**
   - Dynamic multi-agent orchestration reference for fan-out generation, adversarial validation, and deduplication workflows.

---

## ⚡ Dataset Utility Commands

Here are the Python one-liner CLI commands used for managing, merging, and deduplicating the datasets:

### 1. Merge All Subsets into `main.json`
Combines `emotional_dataset.json`, `genz.json`, and `aira_personality.json`, shuffles the order, and outputs to `datasets/main.json`:

```bash
python -c "import json, os, random; random.seed(42); files = ['emotional_dataset.json', 'genz.json', 'aira_personality.json']; merged = []; [merged.extend(json.load(open(os.path.join('datasets', f), 'r', encoding='utf-8'))) for f in files if os.path.exists(os.path.join('datasets', f))]; random.shuffle(merged); out_path = os.path.join('datasets', 'main.json'); json.dump(merged, open(out_path, 'w', encoding='utf-8'), indent=2, ensure_ascii=False); print(f'Successfully merged {len(merged)} conversations into {out_path}')"
```

### 2. Inspect Total and Unique Conversations in `main.json`
Counts total records, unique conversation flows, and duplicate count:

```bash
python -c "import json, os; path = os.path.join('datasets', 'main.json'); data = json.load(open(path, 'r', encoding='utf-8')); total = len(data); unique = len({tuple((m['role'], m['content']) for m in c['messages']) for c in data}); print(f'Total conversations: {total}'); print(f'Unique conversations: {unique}'); print(f'Duplicates: {total - unique}')"
```

### 3. Deduplicate `main.json`
Filters out duplicates while preserving first-seen conversation ordering and updates `main.json`:

```bash
python -c "import json, os; path = os.path.join('datasets', 'main.json'); data = json.load(open(path, 'r', encoding='utf-8')); initial = len(data); seen = set(); deduped = [c for c in data if (k := tuple((m['role'], m['content']) for m in c['messages'])) not in seen and not seen.add(k)]; json.dump(deduped, open(path, 'w', encoding='utf-8'), indent=2, ensure_ascii=False); print(f'Removed {initial - len(deduped)} duplicates. Total unique remaining: {len(deduped)}')"
```