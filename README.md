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

---

## 🚀 Gemma 3 (4B) QLoRA Fine-Tuning Pipeline

This project includes a production-grade **QLoRA (Quantized Low-Rank Adaptation)** fine-tuning suite specifically optimized to train **Gemma 3 (4B)** on `datasets/main.json` using local consumer hardware (such as an **NVIDIA GeForce RTX 3050 6GB Laptop GPU**).

### 📁 Program & Folder Architecture

```text
c:\aira_dataset\
├── .venv/                         # Dedicated Python 3.12 virtual environment (CUDA enabled)
├── configs/
│   └── training_config.yaml       # Hyperparameters (batch size, lora rank, learning rate, etc.)
├── datasets/
│   ├── main.json                  # Master multi-turn conversation dataset
│   ├── emotional_dataset.json
│   ├── genz.json
│   └── aira_personality.json
├── models/
│   └── ollama/                    # Local copy of Ollama's gemma3:4b (GGUF + Modelfile)
│       ├── gemma3-4b-it-q4_k_m.gguf
│       ├── Modelfile
│       └── metadata.json
├── outputs/
│   ├── checkpoints/               # Periodic training checkpoints
│   └── aira_lora/                 # Final trained LoRA adapter weights & tokenizer
├── scripts/
│   ├── download_from_ollama.py    # Extracts/downloads gemma3:4b from Ollama to models/ollama/
│   ├── train.py                   # Main QLoRA training script with Rich Terminal UI
│   ├── export_to_ollama.py        # Packages fine-tuned model into Ollama as 'aira:latest'
│   └── chat.py                    # Interactive terminal chat with fine-tuned Aira
├── src/
│   ├── __init__.py
│   ├── dataset.py                 # Dataset loader with Gemma 3 chat template & loss masking
│   ├── model.py                   # 4-bit NormalFloat BitsAndBytesConfig & PEFT setup
│   └── ui.py                      # Rich terminal live dashboard (VRAM, progress, loss, ETA)
├── requirements.txt               # Pinned Python package dependencies
└── README.md                      # Complete project documentation
```

---

### 🖥️ Hardware & VRAM Optimization (RTX 3050 6GB)

Fine-tuning a 4-billion parameter model on a 6 GB VRAM GPU is made possible through key QLoRA techniques:
- **4-bit NormalFloat (`nf4`) Quantization**: Reduces model weight memory from ~8 GB down to ~2.8 GB.
- **Double Quantization (`bnb_4bit_use_double_quant`)**: Further compresses quantization constants.
- **Gradient Checkpointing**: Drastically reduces activation memory during backpropagation.
- **Paged 8-bit AdamW (`paged_adamw_8bit`)**: Allocates optimizer states in 8-bit precision and pages to system RAM during memory pressure.
- **Micro-batching**: `batch_size = 1` with `gradient_accumulation_steps = 8` (effective batch size of 8).
- **Target VRAM Footprint**: ~4.6 GB / 6.0 GB (leaves headroom for Windows OS and display buffers).

---

### 📋 Step-by-Step Instructions to Run Fine-Tuning

#### 1. Activate the Virtual Environment
Activate the isolated `.venv` containing CUDA-accelerated PyTorch:

```powershell
# In PowerShell:
.\.venv\Scripts\Activate.ps1
```

*(If PowerShell script execution is restricted, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` first).*

#### 2. Download / Extract `gemma3:4b` from Ollama
To extract and store Ollama's `gemma3:4b` model inside this repository:

```powershell
python scripts/download_from_ollama.py
```
This inspects Ollama, pulls the model if missing, copies the quantized GGUF weights into `models/ollama/gemma3-4b-it-q4_k_m.gguf`, and saves the associated Modelfile.

#### 3. Start QLoRA Fine-Tuning with Rich Terminal UI
Run the fine-tuning script:

```powershell
python fine_tune.py
```
*(You can also pass optional overrides like `python fine_tune.py --epochs 3 --batch-size 1` or test with `python fine_tune.py --dry-run`).*

While running, the terminal displays an interactive **Live Dashboard**:
- **Hardware Status**: Live GPU VRAM allocation bar (`[██████████░░░░] 4.6 / 6.0 GB`), device name, and temperature.
- **Training Metrics**: Current Epoch, Global Step progress bar, step loss, smoothed average loss, and learning rate.
- **Speed & Timers**: Elapsed time, estimated time of arrival (ETA), and steps/second.
- **Activity Log**: Checkpoint saves and milestone events.

#### 4. Export Trained Adapters to Ollama
Once training is finished, register your fine-tuned Aira companion into Ollama:

```powershell
python scripts/export_to_ollama.py
```
This generates the unified Modelfile and registers the model in Ollama under the tag `aira:latest`.

#### 5. Chat with Aira
Start chatting with your fine-tuned companion:

```powershell
# Option A: via the interactive script
python scripts/chat.py

# Option B: directly with Ollama
ollama run aira:latest
```