"""
Model loading and QLoRA 4-bit configuration for Gemma 3 (4B).
"""

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training
)
from typing import Dict, Any, Tuple

def get_bnb_config(quant_cfg: Dict[str, Any]) -> BitsAndBytesConfig:
    """Configures 4-bit NormalFloat quantization with double quant."""
    compute_dtype = getattr(torch, quant_cfg.get("bnb_4bit_compute_dtype", "bfloat16"))
    return BitsAndBytesConfig(
        load_in_4bit=quant_cfg.get("load_in_4bit", True),
        bnb_4bit_quant_type=quant_cfg.get("bnb_4bit_quant_type", "nf4"),
        bnb_4bit_use_double_quant=quant_cfg.get("bnb_4bit_use_double_quant", True),
        bnb_4bit_compute_dtype=compute_dtype
    )

def load_qlora_model(config: Dict[str, Any]) -> Tuple[Any, Any]:
    """
    Loads base Gemma 3 model in 4-bit NormalFloat quantization and attaches LoRA adapters.
    Optimized for NVIDIA GPUs with 6 GB VRAM.
    """
    from transformers import AutoConfig

    model_cfg = config.get("model", {})
    quant_cfg = config.get("quantization", {})
    lora_cfg_dict = config.get("lora", {})
    trust_remote = model_cfg.get("trust_remote_code", True)

    model_id = model_cfg.get("base_model_id", "unsloth/gemma-3-4b-it-bnb-4bit")
    bnb_config = get_bnb_config(quant_cfg)

    # Check for gated repo or 403 fallback
    try:
        model_config = AutoConfig.from_pretrained(model_id, trust_remote_code=trust_remote)
    except Exception as e:
        if "google/gemma-3-4b-it" in model_id and ("gated" in str(e).lower() or "403" in str(e)):
            print(f"[!] Notice: {model_id} access is awaiting approval. Falling back to unsloth/gemma-3-4b-it-bnb-4bit.")
            model_id = "unsloth/gemma-3-4b-it-bnb-4bit"
            model_config = AutoConfig.from_pretrained(model_id, trust_remote_code=trust_remote)
        else:
            raise e

    print(f"[*] Loading tokenizer for: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=trust_remote
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    is_prequantized = hasattr(model_config, "quantization_config") and model_config.quantization_config is not None

    print(f"[*] Loading model in 4-bit NF4 precision (device_map='auto')...")
    load_kwargs = {
        "device_map": model_cfg.get("device_map", "auto"),
        "trust_remote_code": trust_remote,
        "torch_dtype": torch.bfloat16 if quant_cfg.get("bnb_4bit_compute_dtype") == "bfloat16" else torch.float16,
    }
    if not is_prequantized:
        load_kwargs["quantization_config"] = bnb_config

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        **load_kwargs
    )

    # Disable KV cache during gradient checkpointing training
    if hasattr(model, "config"):
        model.config.use_cache = False

    # Enable gradient checkpointing to save VRAM on RTX 3050 6GB
    use_grad_ckpt = config.get("training", {}).get("gradient_checkpointing", True)
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=use_grad_ckpt
    )

    # Attach LoRA adapters
    target_modules = lora_cfg_dict.get("target_modules", [
        "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"
    ])

    lora_config = LoraConfig(
        r=lora_cfg_dict.get("r", 16),
        lora_alpha=lora_cfg_dict.get("lora_alpha", 32),
        target_modules=target_modules,
        lora_dropout=lora_cfg_dict.get("lora_dropout", 0.05),
        bias=lora_cfg_dict.get("bias", "none"),
        task_type="CAUSAL_LM"
    )

    peft_model = get_peft_model(model, lora_config)
    peft_model.print_trainable_parameters()

    return peft_model, tokenizer

