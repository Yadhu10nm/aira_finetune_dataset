"""
Interactive Terminal Chat with Fine-Tuned Aira.
Allows chatting with Aira via Ollama or directly via PyTorch/PEFT weights.
"""

import os
import sys

# Ensure UTF-8 output on Windows consoles
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import subprocess
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt

PROJECT_ROOT = Path(__file__).resolve().parent.parent

def chat_via_ollama(model_name="aira:latest"):
    console = Console(force_terminal=True, highlight=False)
    console.print(Panel.fit(
        f"[bold pink1]💬 CHATTING WITH AIRA (Ollama: {model_name})[/bold pink1]\n"
        f"[dim white]Type 'exit' or 'quit' to leave the conversation.[/dim white]",
        border_style="magenta"
    ))

    # Check if ollama has the model
    proc = subprocess.run("ollama list", shell=True, capture_output=True, text=True)
    if "aira" not in proc.stdout:
        console.print(f"[yellow]⚠️ Model '{model_name}' not found in Ollama list.[/yellow]")
        console.print(f"[cyan]Tip: Run `python scripts/export_to_ollama.py` first, or falling back to ollama run gemma3:4b[/cyan]")
        model_name = "gemma3:4b"

    # Launch ollama run interactive session
    subprocess.run(f"ollama run {model_name}", shell=True)

if __name__ == "__main__":
    chat_via_ollama()
