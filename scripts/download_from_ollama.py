"""
Extracts and downloads gemma3:4b from Ollama and stores it locally inside this repository.
Path: models/ollama/gemma3-4b-it-q4_K_M.gguf
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

import json
import shutil
import subprocess
from pathlib import Path

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TransferSpeedColumn, TimeRemainingColumn
except ImportError:
    # Fallback if rich is not yet installed
    Console = None

def get_console():
    return Console(force_terminal=True, highlight=False) if Console else None

def run_command(cmd, desc="Running command"):
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True, encoding="utf-8", errors="replace")
    return result.returncode == 0, result.stdout, result.stderr

def pull_ollama_model(model_name="gemma3:4b", console=None):
    if console:
        console.print(f"[bold cyan]🔍 Checking Ollama model:[/bold cyan] [yellow]{model_name}[/yellow]")
    
    # Check if model exists in ollama
    success, stdout, _ = run_command("ollama list")
    if not success:
        if console:
            console.print("[bold red]❌ Error: Ollama is not installed or the Ollama service is not running.[/bold red]")
            console.print("Please make sure Ollama is running (`ollama serve` or Ollama app).")
        sys.exit(1)

    if model_name not in stdout:
        if console:
            console.print(f"[yellow]⚡ Pulling {model_name} from Ollama registry...[/yellow]")
        proc = subprocess.Popen(f"ollama pull {model_name}", shell=True)
        proc.wait()
        if proc.returncode != 0:
            if console:
                console.print(f"[bold red]❌ Failed to pull {model_name} from Ollama.[/bold red]")
            sys.exit(1)
    else:
        if console:
            console.print(f"[bold green]✔ Model '{model_name}' found in Ollama repository![/bold green]")

def find_ollama_storage():
    """Locates Ollama models directory on Windows/macOS/Linux."""
    home = Path.home()
    ollama_dir = os.environ.get("OLLAMA_MODELS")
    if ollama_dir and os.path.isdir(ollama_dir):
        return Path(ollama_dir)
    default_dir = home / ".ollama" / "models"
    if default_dir.exists():
        return default_dir
    return None

def copy_with_progress(src_file: Path, dst_file: Path, console=None):
    total_size = src_file.stat().st_size
    chunk_size = 1024 * 1024 * 16  # 16MB buffer

    if console and Progress:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.1f}%"),
            TransferSpeedColumn(),
            TimeRemainingColumn(),
            console=console
        ) as progress:
            task = progress.add_task(f"Copying {src_file.name[:20]}...", total=total_size)
            with open(src_file, "rb") as fsrc, open(dst_file, "wb") as fdst:
                while True:
                    buf = fsrc.read(chunk_size)
                    if not buf:
                        break
                    fdst.write(buf)
                    progress.update(task, advance=len(buf))
    else:
        print(f"Copying {src_file} -> {dst_file} ({total_size / (1024**3):.2f} GB)...")
        shutil.copy2(src_file, dst_file)

def export_gemma_from_ollama(target_dir="models/ollama", model_name="gemma3:4b"):
    console = get_console()
    target_path = Path(target_dir).resolve()
    target_path.mkdir(parents=True, exist_ok=True)

    if console:
        console.print(Panel.fit(
            f"[bold magenta]Ollama Model Exporter[/bold magenta]\n"
            f"[cyan]Target Model:[/cyan] {model_name}\n"
            f"[cyan]Destination Directory:[/cyan] {target_path}",
            border_style="magenta"
        ))

    pull_ollama_model(model_name, console)

    models_dir = find_ollama_storage()
    if not models_dir:
        if console:
            console.print("[bold red]❌ Could not locate Ollama models directory (~/.ollama/models).[/bold red]")
        sys.exit(1)

    # Search for the manifest
    tag = "4b" if ":" not in model_name else model_name.split(":")[1]
    name = model_name.split(":")[0]

    manifest_candidates = list((models_dir / "manifests").glob(f"**/{name}/{tag}"))
    if not manifest_candidates:
        manifest_candidates = list((models_dir / "manifests").glob(f"**/{name}/*"))

    if not manifest_candidates:
        if console:
            console.print(f"[bold red]❌ Manifest for {model_name} not found in {models_dir / 'manifests'}[/bold red]")
        sys.exit(1)

    manifest_path = manifest_candidates[0]
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    # Find the model layer blob
    model_blob_digest = None
    for layer in manifest.get("layers", []):
        if layer.get("mediaType") == "application/vnd.ollama.image.model":
            model_blob_digest = layer.get("digest", "")
            break

    if not model_blob_digest:
        if console:
            console.print("[bold red]❌ Could not find model weight layer in manifest.[/bold red]")
        sys.exit(1)

    blob_filename = model_blob_digest.replace(":", "-")
    blob_path = models_dir / "blobs" / blob_filename

    if not blob_path.exists():
        if console:
            console.print(f"[bold red]❌ Model blob does not exist: {blob_path}[/bold red]")
        sys.exit(1)

    # Destination GGUF file
    dest_gguf = target_path / f"{name}-{tag}-q4_k_m.gguf"
    if dest_gguf.exists() and dest_gguf.stat().st_size == blob_path.stat().st_size:
        if console:
            console.print(f"[bold green]✔ Model already exists in project folder:[/bold green] [cyan]{dest_gguf}[/cyan]")
    else:
        if console:
            console.print(f"[cyan]📦 Copying GGUF weights to project folder...[/cyan]")
        copy_with_progress(blob_path, dest_gguf, console)

    # Export Modelfile
    modelfile_path = target_path / "Modelfile"
    success, stdout, _ = run_command(f"ollama show --modelfile {model_name}")
    if success and stdout:
        with open(modelfile_path, "w", encoding="utf-8") as f:
            f.write(stdout)
        if console:
            console.print(f"[bold green]✔ Exported Modelfile to:[/bold green] [cyan]{modelfile_path}[/cyan]")

    # Save summary metadata
    meta_path = target_path / "metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump({
            "model_name": model_name,
            "gguf_filename": dest_gguf.name,
            "gguf_size_bytes": dest_gguf.stat().st_size,
            "gguf_size_gb": round(dest_gguf.stat().st_size / (1024**3), 2),
            "manifest_digest": model_blob_digest,
            "local_path": str(dest_gguf)
        }, f, indent=2)

    if console:
        table = Table(title="Ollama Local Model Details", border_style="cyan")
        table.add_column("Property", style="bold yellow")
        table.add_column("Value", style="green")
        table.add_row("Model Name", model_name)
        table.add_row("Format", "GGUF (Q4_K_M quantized)")
        table.add_row("Size", f"{round(dest_gguf.stat().st_size / (1024**3), 2)} GB")
        table.add_row("Location", str(dest_gguf))
        table.add_row("Modelfile", str(modelfile_path))
        console.print(table)
        console.print(f"\n[bold green]✅ Model successfully kept locally inside {target_dir}![/bold green]\n")

if __name__ == "__main__":
    export_gemma_from_ollama()
