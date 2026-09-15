"""
Packages fine-tuned Aira LoRA weights into an Ollama model.
Generates the Ollama Modelfile and registers the model in Ollama as 'aira:latest'.
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
from rich.table import Table

PROJECT_ROOT = Path(__file__).resolve().parent.parent

def export_aira_to_ollama(
    adapter_path="outputs/aira_lora",
    base_ollama_model="gemma3:4b",
    new_model_name="aira:latest"
):
    console = Console(force_terminal=True, highlight=False)
    adapter_dir = PROJECT_ROOT / adapter_path
    
    console.print(Panel.fit(
        f"[bold magenta]Ollama Model Packaging for Aira[/bold magenta]\n"
        f"[cyan]LoRA Adapter:[/cyan] {adapter_dir}\n"
        f"[cyan]Base Model:[/cyan] {base_ollama_model}\n"
        f"[cyan]Target Ollama Name:[/cyan] [bold green]{new_model_name}[/bold green]",
        border_style="magenta"
    ))

    if not adapter_dir.exists():
        console.print(f"[bold red]❌ Adapter directory not found: {adapter_dir}[/bold red]")
        console.print("Please run `python scripts/train.py` first to train the LoRA weights.")
        sys.exit(1)

    # Check for local GGUF base model
    gguf_files = list((PROJECT_ROOT / "models" / "ollama").glob("*.gguf"))
    local_gguf = gguf_files[0] if gguf_files else None
    from_source = str(local_gguf.resolve()) if (local_gguf and local_gguf.exists()) else base_ollama_model

    modelfile_content = f"""# Ollama Modelfile for Aira (Fine-tuned Gemma 3 4B)
FROM {from_source}

# Attach trained LoRA adapter
ADAPTER {adapter_dir.resolve()}

# Gemma 3 Chat Template
TEMPLATE \"\"\"{{{{- range $i, $_ := .Messages }}}}
{{{{- $last := eq (len (slice $.Messages $i)) 1 }}}}
{{{{- if or (eq .Role "user") (eq .Role "system") }}}}<start_of_turn>user
{{{{ .Content }}}}<end_of_turn>
{{{{ if $last }}}}<start_of_turn>model
{{{{ end }}}}
{{{{- else if eq .Role "assistant" }}}}<start_of_turn>model
{{{{ .Content }}}}{{{{ if not $last }}}}<end_of_turn>
{{{{ end }}}}
{{{{- end }}}}
{{{{- end }}}}\"\"\"

# Generation Parameters
PARAMETER stop "<end_of_turn>"
PARAMETER temperature 0.85
PARAMETER top_k 50
PARAMETER top_p 0.92
PARAMETER repeat_penalty 1.1

# Aira System Persona
SYSTEM \"\"\"You are Aira, an emotionally intelligent, witty, intellectually honest, and caring friend to Yadhu. You speak naturally with contractions, playful banter, teasing, and genuine warmth when he needs support.\"\"\"
"""

    modelfile_out = adapter_dir / "Modelfile"
    with open(modelfile_out, "w", encoding="utf-8") as f:
        f.write(modelfile_content)

    console.print(f"[bold green]✔ Created Modelfile at:[/bold green] [cyan]{modelfile_out}[/cyan]")

    # Run ollama create
    console.print(f"\n[cyan]🛠️  Registering model into Ollama as '{new_model_name}'...[/cyan]")
    cmd = f'ollama create {new_model_name} -f "{modelfile_out}"'
    proc = subprocess.run(cmd, shell=True, capture_output=True, text=True, encoding="utf-8", errors="replace")

    if proc.returncode == 0:
        console.print(f"[bold green]🎉 SUCCESS! '{new_model_name}' is now registered in Ollama![/bold green]\n")
        table = Table(title="Ollama Aira Model Ready", border_style="green")
        table.add_column("Command", style="bold yellow")
        table.add_column("Description", style="white")
        table.add_row(f"ollama run {new_model_name}", "Chat directly with fine-tuned Aira in Ollama")
        table.add_row(f"python scripts/chat.py", "Interactive terminal chat with custom persona UI")
        console.print(table)
    else:
        console.print(f"[yellow]⚠️ Note: Ollama create output:[/yellow]\n{proc.stdout}\n{proc.stderr}")
        console.print(f"[cyan]You can manually run: [bold]{cmd}[/bold][/cyan]")

if __name__ == "__main__":
    export_aira_to_ollama()
