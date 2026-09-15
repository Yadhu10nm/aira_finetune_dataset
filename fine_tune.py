#!/usr/bin/env python
"""
Aira Fine-Tuning Entrypoint Script.
Executes QLoRA (4-bit NormalFloat) fine-tuning for Gemma 3 (4B) on datasets/main.json.

Usage:
    python fine_tune.py
    python fine_tune.py --epochs 3 --batch-size 1
    python fine_tune.py --dry-run
"""

import os
import sys
import argparse
from pathlib import Path

# Set environment flags before importing HF/torch libraries
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Ensure UTF-8 output on Windows consoles
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

PROJECT_ROOT = Path(__file__).resolve().parent

# Auto-reexec inside the dedicated .venv if invoked with system Python
def ensure_virtualenv():
    venv_python = PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"
    current_executable = Path(sys.executable).resolve()
    
    if venv_python.exists() and current_executable != venv_python.resolve():
        print(f"[*] Detected system Python: {current_executable}")
        print(f"[*] Switching execution to project virtual environment: {venv_python}")
        import subprocess
        result = subprocess.run([str(venv_python)] + sys.argv, cwd=str(PROJECT_ROOT))
        sys.exit(result.returncode)

ensure_virtualenv()

# When inside the proper environment, proceed with imports
sys.path.insert(0, str(PROJECT_ROOT))

import math
import time
import yaml
import torch
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup

from src.model import load_qlora_model
from src.dataset import AiraConversationDataset, DataCollatorForGemma
from src.ui import TrainingDashboard

try:
    import bitsandbytes as bnb
    HAS_BNB = True
except ImportError:
    HAS_BNB = False

def load_config(config_path: str = "configs/training_config.yaml"):
    path = PROJECT_ROOT / config_path
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune Gemma 3 (4B) for Aira Persona")
    parser.add_argument("--config", type=str, default="configs/training_config.yaml", help="Path to config YAML")
    parser.add_argument("--epochs", type=int, default=None, help="Override number of training epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Override per-device train batch size")
    parser.add_argument("--grad-accum", type=int, default=None, help="Override gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    parser.add_argument("--dry-run", action="store_true", help="Quick sanity check on first batch only")
    return parser.parse_args()

def main():
    args = parse_args()
    config = load_config(args.config)

    train_cfg = config.get("training", {})
    data_cfg = config.get("data", {})
    model_cfg = config.get("model", {})

    # Apply CLI overrides if specified
    if args.epochs is not None:
        train_cfg["num_train_epochs"] = args.epochs
    if args.batch_size is not None:
        train_cfg["per_device_train_batch_size"] = args.batch_size
    if args.grad_accum is not None:
        train_cfg["gradient_accumulation_steps"] = args.grad_accum
    if args.lr is not None:
        train_cfg["learning_rate"] = args.lr

    output_dir = PROJECT_ROOT / train_cfg.get("output_dir", "outputs/aira_lora")
    checkpoint_dir = PROJECT_ROOT / train_cfg.get("checkpoint_dir", "outputs/checkpoints")
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*65)
    print(" 🌟 AIRA CONVERSATIONAL PERSONA — GEMMA 3 (4B) QLoRA FINE-TUNING")
    print("="*65 + "\n")

    # Hardware check
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram_total_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"[*] GPU Acceleration: ACTIVE")
        print(f"[*] Device: {gpu_name} ({vram_total_gb:.2f} GB VRAM)")
        device = torch.device("cuda:0")
    else:
        print("[!] WARNING: CUDA device not detected! Running on CPU will be extremely slow.")
        gpu_name = "CPU"
        vram_total_gb = 0.0
        device = torch.device("cpu")

    # 1. Load Model & Tokenizer
    model, tokenizer = load_qlora_model(config)

    # 2. Dataset Setup
    dataset_file = PROJECT_ROOT / data_cfg.get("dataset_path", "datasets/main.json")
    if not dataset_file.exists():
        raise FileNotFoundError(f"Master training dataset not found at {dataset_file}")

    print(f"[*] Loading training dataset from: {dataset_file}")
    max_seq_len = data_cfg.get("max_seq_length", 1024)
    val_split = data_cfg.get("validation_split", 0.05)
    seed = data_cfg.get("seed", 42)

    train_dataset = AiraConversationDataset(
        data_path=str(dataset_file),
        tokenizer=tokenizer,
        max_seq_length=max_seq_len,
        split="train",
        val_ratio=val_split,
        seed=seed
    )

    val_dataset = AiraConversationDataset(
        data_path=str(dataset_file),
        tokenizer=tokenizer,
        max_seq_length=max_seq_len,
        split="val",
        val_ratio=val_split,
        seed=seed
    )

    collator = DataCollatorForGemma(pad_token_id=tokenizer.pad_token_id or 0)
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

    print(f"[*] Total training dialogues: {len(train_dataset):,}")
    print(f"[*] Total validation dialogues: {len(val_dataset):,}")
    print(f"[*] Batch size: {batch_size} (effective batch size: {batch_size * grad_accum_steps})")
    print(f"[*] Total optimization steps: {total_training_steps:,}")
    print(f"[*] Warmup steps: {warmup_steps}\n")

    # 3. Optimizer & Scheduler
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
        device_name=gpu_name,
        total_vram_gb=vram_total_gb
    )

    dashboard.start()
    dashboard.update(
        epoch=1,
        step=0,
        loss=0.0,
        lr=lr,
        status="Starting training...",
        log_entry=f"Ready. {len(train_dataset)} conversations loaded."
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

                    dashboard.update(
                        epoch=epoch,
                        step=global_step,
                        loss=step_loss,
                        lr=current_lr,
                        grad_norm=float(grad_norm.item()) if hasattr(grad_norm, 'item') else float(grad_norm),
                        status=f"Optimizing Step {global_step}/{total_training_steps}"
                    )

                    # Save checkpoint
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
                            log_entry=f"💾 Checkpoint saved: {ckpt_path.name}"
                        )

                    if args.dry_run and global_step >= 2:
                        print("\n[!] Dry run test complete (2 steps executed successfully).")
                        return

            dashboard.update(
                epoch=epoch,
                step=global_step,
                loss=step_loss if 'step_loss' in locals() else 0.0,
                lr=current_lr if 'current_lr' in locals() else lr,
                log_entry=f"🎉 Completed Epoch {epoch}/{num_epochs}"
            )

    except KeyboardInterrupt:
        print("\n[!] Training paused by user (Ctrl+C). Saving current state...")
    finally:
        # Save final trained weights
        print(f"\n[*] Saving final Aira LoRA adapter to: {output_dir}")
        model.save_pretrained(str(output_dir))
        tokenizer.save_pretrained(str(output_dir))

        dashboard.update(
            epoch=num_epochs,
            step=global_step,
            loss=0.0,
            lr=0.0,
            status="Fine-tuning completed successfully!",
            log_entry=f"✅ Final LoRA adapter saved to {output_dir.name}"
        )
        dashboard.stop()

    print("\n" + "="*65)
    print(" 🎉 FINE-TUNING FINISHED SUCCESSFULLY!")
    print(f" 📂 Trained LoRA Adapters saved in: {output_dir}")
    print("="*65)
    print("\nNext Steps:")
    print("  1. Export fine-tuned model to Ollama:")
    print("     python scripts/export_to_ollama.py")
    print("  2. Chat with Aira:")
    print("     python scripts/chat.py")
    print("     ollama run aira:latest\n")

if __name__ == "__main__":
    main()
