"""
Dataset loader and formatting pipeline for Aira conversational training data.
Handles Gemma 3 turn formatting, tokenization, and assistant response loss masking.
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Tuple
import torch
from torch.utils.data import Dataset

class AiraConversationDataset(Dataset):
    """
    Loads multi-turn conversations from main.json, formats them using Gemma 3's chat template,
    and masks user tokens with -100 so loss is computed exclusively on Aira's responses.
    """

    def __init__(
        self,
        data_path: str,
        tokenizer: Any,
        max_seq_length: int = 1024,
        split: str = "train",
        val_ratio: float = 0.05,
        seed: int = 42
    ):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.samples = []

        data_file = Path(data_path)
        if not data_file.exists():
            raise FileNotFoundError(f"Dataset file not found: {data_file}")

        with open(data_file, "r", encoding="utf-8") as f:
            raw_data = json.load(f)

        # Ensure reproducibility for train/val split
        import random
        rng = random.Random(seed)
        shuffled = list(raw_data)
        rng.shuffle(shuffled)

        val_size = int(len(shuffled) * val_ratio)
        if split == "train":
            self.conversations = shuffled[val_size:]
        else:
            self.conversations = shuffled[:val_size]

    def __len__(self) -> int:
        return len(self.conversations)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.conversations[idx]
        messages = item.get("messages", [])

        # Process multi-turn messages into token IDs with loss masking for user turns
        input_ids = []
        labels = []

        # Gemma token markers
        # <start_of_turn>user\n...<end_of_turn>\n<start_of_turn>model\n...<end_of_turn>
        for message in messages:
            role = message.get("role")
            content = message.get("content", "")

            # Normalize role for Gemma template
            target_role = "model" if role == "assistant" else "user"

            # Format turn
            turn_text = f"<start_of_turn>{target_role}\n{content}<end_of_turn>\n"
            turn_tokens = self.tokenizer.encode(turn_text, add_special_tokens=False)

            input_ids.extend(turn_tokens)
            if target_role == "model":
                # Model turn: compute loss on model output
                # Optional refinement: mask the prefix "<start_of_turn>model\n" so loss only counts on actual speech
                prefix = f"<start_of_turn>{target_role}\n"
                prefix_len = len(self.tokenizer.encode(prefix, add_special_tokens=False))
                turn_labels = [-100] * prefix_len + turn_tokens[prefix_len:]
                labels.extend(turn_labels)
            else:
                # User turn: mask completely with -100
                labels.extend([-100] * len(turn_tokens))

        # Add BOS token if present
        if self.tokenizer.bos_token_id is not None:
            input_ids = [self.tokenizer.bos_token_id] + input_ids
            labels = [-100] + labels

        # Truncate to max_seq_length
        if len(input_ids) > self.max_seq_length:
            input_ids = input_ids[:self.max_seq_length]
            labels = labels[:self.max_seq_length]

        attention_mask = [1] * len(input_ids)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }

class DataCollatorForGemma:
    """
    Dynamic padding collator for variable-length conversation batches.
    Pads input_ids with pad_token_id and labels with -100.
    """
    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        max_len = max(item["input_ids"].size(0) for item in batch)

        batch_input_ids = []
        batch_attention_mask = []
        batch_labels = []

        for item in batch:
            input_len = item["input_ids"].size(0)
            pad_len = max_len - input_len

            # Pad right
            padded_input_ids = torch.cat([
                item["input_ids"],
                torch.full((pad_len,), self.pad_token_id, dtype=torch.long)
            ])
            padded_attention_mask = torch.cat([
                item["attention_mask"],
                torch.zeros((pad_len,), dtype=torch.long)
            ])
            padded_labels = torch.cat([
                item["labels"],
                torch.full((pad_len,), -100, dtype=torch.long)
            ])

            batch_input_ids.append(padded_input_ids)
            batch_attention_mask.append(padded_attention_mask)
            batch_labels.append(padded_labels)

        return {
            "input_ids": torch.stack(batch_input_ids),
            "attention_mask": torch.stack(batch_attention_mask),
            "labels": torch.stack(batch_labels),
        }
