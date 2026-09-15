"""
QLoRA Fine-Tuning Execution Script for Gemma 3 (4B).
Trains Aira's conversational persona using multi-turn conversations from datasets/main.json.
Features a real-time Rich Terminal UI for monitoring GPU VRAM, loss, and training metrics.
"""

import os
import sys

# Suppress symlink warnings and parallel tokenizer warnings
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Ensure UTF-8 output on Windows consoles
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import yaml
import math
import time
from pathlib import Path
import torch
from torch.utils.data import DataLoader

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.model import load_qlora_model
from src.dataset import AiraConversationDataset, DataCollatorForGemma
from src.ui import TrainingDashboard

try:
    import bitsandbytes as bnb
    HAS_BNB = True
except ImportError:
    HAS_BNB = False

from transformers import get_cosine_schedule_with_warmup

def load_yaml_config(config_path: str = "configs/training_config.yaml"):
    path = PROJECT_ROOT / config_path
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def run_training():
    config = load_yaml_config()
    train_cfg = config.get("training", {})
    data_cfg = config.get("data", {})
    model_cfg = config.get("model", {})

    output_dir = PROJECT_ROOT / train_cfg.get("output_dir", "outputs/aira_lora")
    checkpoint_dir = PROJECT_ROOT / train_cfg.get("checkpoint_dir", "outputs/checkpoints")
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*60)
    print(" 🚀 INITIALIZING AIRA GEMMA 3 (4B) QLoRA FINE-TUNING")
    print("="*60 + "\n")

    # Check CUDA
    if not torch.cuda.is_available():
        print("[!] WARNING: CUDA is not available. Training on CPU will be extremely slow!")
        device = torch.device("cpu")
        vram_total_gb = 0.0
    else:
        device = torch.device("cuda:0")
        vram_total_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"[*] Detected GPU: {torch.cuda.get_device_name(0)} ({vram_total_gb:.2f} GB VRAM)")

    # 1. Load Model and Tokenizer
    model, tokenizer = load_qlora_model(config)

    # 2. Prepare Datasets
    print(f"[*] Preparing dataset from: {data_cfg.get('dataset_path')}")
    train_dataset = AiraConversationDataset(
        data_path=str(PROJECT_ROOT / data_cfg.get("dataset_path", "datasets/main.json")),
        tokenizer=tokenizer,
        max_seq_length=data_cfg.get("max_seq_length", 1024),
        split="train",
        val_ratio=data_cfg.get("validation_split", 0.05),
        seed=data_cfg.get("seed", 42)
    )

    val_dataset = AiraConversationDataset(
        data_path=str(PROJECT_ROOT / data_cfg.get("dataset_path", "datasets/main.json")),
        tokenizer=tokenizer,
        max_seq_length=data_cfg.get("max_seq_length", 1024),
        split="val",
        val_ratio=data_cfg.get("validation_split", 0.05),
        seed=data_cfg.get("seed", 42)
    )

    collator = DataCollatorForGemma(pad_token_id=tokenizer.pad_token_id)
    
    batch_size = train_cfg.get("per_device_train_batch_size", 1)
    grad_accum_steps = train_cfg.get("gradient_accumulation_steps", 8)
    num_epochs = train_cfg.get("num_train_epochs", 3)
    lr = float(train_cfg.get("learning_rate", 2e-4))
    weight_decay = float(train_cfg.get("weight_decay", 0.01))

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collator
    )

    steps_per_epoch = math.ceil(len(train_loader) / grad_accum_steps)
    total_training_steps = steps_per_epoch * num_epochs
    warmup_steps = int(total_training_steps * float(train_cfg.get("warmup_ratio", 0.03)))

    print(f"[*] Total training dialogues: {len(train_dataset)}")
    print(f"[*] Total validation dialogues: {len(val_dataset)}")
    print(f"[*] Effective batch size: {batch_size * grad_accum_steps}")
    print(f"[*] Total training optimization steps: {total_training_steps}\n")

    # 3. Setup Optimizer and LR Scheduler
    # Use 8-bit Paged AdamW for maximum VRAM conservation on 6GB RTX 3050
    if HAS_BNB:
        optimizer = bnb.optim.PagedAdamW8bit(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )

    scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_training_steps
    )

    # 4. Initialize Live UI Dashboard
    dashboard = TrainingDashboard(
        total_epochs=num_epochs,
        total_steps=total_training_steps,
        model_name="Gemma 3 (4B)",
        device_name=torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        total_vram_gb=vram_total_gb
    )

    dashboard.start()
    dashboard.update(
        epoch=1,
        step=0,
        loss=0.0,
        lr=lr,
        status="Training started...",
        log_entry=f"Loaded {len(train_dataset)} conversations. Warmup: {warmup_steps} steps."
    )

    # 5. Training Loop
    global_step = 0
    model.train()
    accumulated_loss = 0.0

    use_bf16 = train_cfg.get("bf16", True) and torch.cuda.is_bf16_supported()
    autocast_dtype = torch.bfloat16 if use_bf16 else torch.float16

    try:
        for epoch in range(1, num_epochs + 1):
            optimizer.zero_grad()

            for step_idx, batch in enumerate(train_loader):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                with torch.amp.autocast(device_type="cuda" if torch.cuda.is_available() else "cpu", dtype=autocast_dtype):
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels
                    )
                    loss = outputs.loss / grad_accum_steps

                loss.backward()
                accumulated_loss += loss.item()

                if (step_idx + 1) % grad_accum_steps == 0 or (step_idx + 1) == len(train_loader):
                    # Clip gradients
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

                    global_step += 1
                    current_lr = scheduler.get_last_lr()[0]
                    step_loss = accumulated_loss * grad_accum_steps
                    accumulated_loss = 0.0

                    # Update Dashboard
                    dashboard.update(
                        epoch=epoch,
                        step=global_step,
                        loss=step_loss,
                        lr=current_lr,
                        grad_norm=float(grad_norm.item()) if hasattr(grad_norm, 'item') else float(grad_norm),
                        status=f"Optimizing Step {global_step}/{total_training_steps}"
                    )

                    # Save intermediate checkpoint
                    if global_step % train_cfg.get("save_steps", 100) == 0:
                        ckpt_path = checkpoint_dir / f"checkpoint-{global_step}"
                        ckpt_path.mkdir(parents=True, exist_ok=True)
                        model.save_pretrained(str(ckpt_path))
                        tokenizer.save_pretrained(str(ckpt_path))
                        dashboard.update(
                            epoch=epoch,
                            step=global_step,
                            loss=step_loss,
                            lr=current_lr,
                            log_entry=f"💾 Saved checkpoint to {ckpt_path.name}"
                        )

            dashboard.update(
                epoch=epoch,
                step=global_step,
                loss=step_loss if 'step_loss' in locals() else 0.0,
                lr=current_lr if 'current_lr' in locals() else lr,
                log_entry=f"🎉 Completed Epoch {epoch}/{num_epochs}"
            )

    except KeyboardInterrupt:
        dashboard.update(
            epoch=epoch if 'epoch' in locals() else 1,
            step=global_step,
            loss=accumulated_loss,
            lr=lr,
            status="Training paused by user (Ctrl+C). Saving current state...",
            log_entry="[WARNING] Interrupted by user."
        )
    finally:
        # Save Final LoRA Weights
        print("\n[*] Saving final Aira LoRA adapter to:", output_dir)
        model.save_pretrained(str(output_dir))
        tokenizer.save_pretrained(str(output_dir))

        dashboard.update(
            epoch=num_epochs,
            step=global_step,
            loss=0.0,
            lr=0.0,
            status="Training complete!",
            log_entry=f"✅ Final LoRA adapter saved to {output_dir.name}"
        )
        dashboard.stop()

    print("\n" + "="*60)
    print(f" ✨ TRAINING FINISHED SUCCESSFULLY!")
    print(f" ✨ Trained LoRA Adapters saved to: {output_dir}")
    print("="*60 + "\n")

if __name__ == "__main__":
    run_training()
