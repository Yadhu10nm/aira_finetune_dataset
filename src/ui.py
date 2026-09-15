"""
Rich Live Terminal UI for Aira QLoRA Fine-Tuning.
Renders an interactive, real-time dashboard displaying GPU VRAM, loss, learning rate, ETA, and progress.
"""

import os
import sys
import time

# Ensure UTF-8 output on Windows consoles
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import torch
from typing import Optional, Dict, Any

from rich.console import Console, Group
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn
from rich.live import Live
from rich.text import Text
from rich import box

class TrainingDashboard:
    """
    Real-time terminal dashboard for monitoring QLoRA fine-tuning.
    """

    def __init__(
        self,
        total_epochs: int,
        total_steps: int,
        model_name: str = "Gemma 3 (4B)",
        device_name: Optional[str] = None,
        total_vram_gb: float = 6.0
    ):
        self.console = Console(force_terminal=True, highlight=False)
        self.total_epochs = total_epochs
        self.total_steps = total_steps
        self.model_name = model_name
        self.device_name = device_name or (torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
        self.total_vram_gb = total_vram_gb
        
        self.current_epoch = 1
        self.current_step = 0
        self.current_loss = 0.0
        self.avg_loss = 0.0
        self.loss_history = []
        self.learning_rate = 0.0
        self.grad_norm = 0.0
        self.start_time = time.time()
        self.last_step_time = time.time()
        self.step_durations = []
        self.status_message = "Initializing training environment..."
        self.recent_logs = []

        self.live = None

    def start(self):
        """Starts the Rich Live context."""
        self.live = Live(self.render(), console=self.console, refresh_per_second=4, screen=False)
        self.live.start()

    def stop(self):
        """Stops the Rich Live context."""
        if self.live:
            self.live.stop()

    def update(
        self,
        epoch: int,
        step: int,
        loss: float,
        lr: float,
        grad_norm: float = 0.0,
        status: Optional[str] = None,
        log_entry: Optional[str] = None
    ):
        now = time.time()
        duration = now - self.last_step_time
        self.last_step_time = now
        
        # Keep last 20 step durations for moving average
        self.step_durations.append(duration)
        if len(self.step_durations) > 20:
            self.step_durations.pop(0)

        self.current_epoch = epoch
        self.current_step = step
        self.current_loss = loss
        self.learning_rate = lr
        self.grad_norm = grad_norm

        self.loss_history.append(loss)
        # Exponential moving average
        if len(self.loss_history) == 1:
            self.avg_loss = loss
        else:
            self.avg_loss = 0.9 * self.avg_loss + 0.1 * loss

        if status:
            self.status_message = status
        if log_entry:
            self.recent_logs.append(log_entry)
            if len(self.recent_logs) > 4:
                self.recent_logs.pop(0)

        if self.live:
            self.live.update(self.render())

    def get_gpu_memory(self) -> Dict[str, float]:
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated(0) / (1024 ** 3)
            reserved = torch.cuda.memory_reserved(0) / (1024 ** 3)
            percent = (reserved / self.total_vram_gb) * 100 if self.total_vram_gb > 0 else 0
            return {
                "allocated_gb": allocated,
                "reserved_gb": reserved,
                "percent": percent
            }
        return {"allocated_gb": 0.0, "reserved_gb": 0.0, "percent": 0.0}

    def _render_bar(self, percent: float, width: int = 24) -> str:
        filled = int((percent / 100.0) * width)
        filled = max(0, min(width, filled))
        bar = "█" * filled + "░" * (width - filled)
        return bar

    def render(self) -> Group:
        # Header Banner
        header = Panel(
            Text.from_markup(
                f"[bold magenta]⚡ AIRA QLoRA FINE-TUNING DASHBOARD ⚡[/bold magenta]\n"
                f"[cyan]Base Model:[/cyan] [bold white]{self.model_name}[/bold white]  |  "
                f"[cyan]Architecture:[/cyan] 4-bit NormalFloat QLoRA  |  "
                f"[cyan]Persona:[/cyan] [bold pink1]Aira (Yadhu's Friend)[/bold pink1]"
            ),
            border_style="magenta",
            box=box.ROUNDED
        )

        # GPU / Hardware Stats
        gpu_info = self.get_gpu_memory()
        vram_bar = self._render_bar(gpu_info["percent"], width=20)
        vram_color = "green" if gpu_info["percent"] < 80 else ("yellow" if gpu_info["percent"] < 90 else "red")

        gpu_table = Table(box=box.SIMPLE, show_header=False, pad_edge=False)
        gpu_table.add_column("Key", style="bold cyan")
        gpu_table.add_column("Value")
        gpu_table.add_row("Device:", f"[bold]{self.device_name}[/bold]")
        gpu_table.add_row(
            "VRAM Usage:",
            f"[{vram_color}]{vram_bar}[/{vram_color}] "
            f"[{vram_color}]{gpu_info['reserved_gb']:.2f} / {self.total_vram_gb:.1f} GB ({gpu_info['percent']:.1f}%)[/{vram_color}]"
        )
        gpu_table.add_row("Active Alloc:", f"{gpu_info['allocated_gb']:.2f} GB")

        # Training Progress & Metrics
        progress_pct = (self.current_step / max(1, self.total_steps)) * 100
        step_bar = self._render_bar(progress_pct, width=28)

        elapsed = time.time() - self.start_time
        elapsed_str = time.strftime("%H:%M:%S", time.gmtime(elapsed))

        avg_step_sec = sum(self.step_durations) / max(1, len(self.step_durations)) if self.step_durations else 0
        remaining_steps = max(0, self.total_steps - self.current_step)
        eta_sec = remaining_steps * avg_step_sec
        eta_str = time.strftime("%H:%M:%S", time.gmtime(eta_sec)) if eta_sec > 0 else "--:--:--"

        speed_str = f"{1.0 / avg_step_sec:.2f} steps/s" if avg_step_sec > 0 else "calculating..."

        metrics_table = Table(box=box.SIMPLE, show_header=False, pad_edge=False)
        metrics_table.add_column("Metric", style="bold cyan")
        metrics_table.add_column("Value")
        metrics_table.add_row("Epoch:", f"[bold yellow]{self.current_epoch}[/bold yellow] / {self.total_epochs}")
        metrics_table.add_row(
            "Global Step:",
            f"[bold green]{step_bar}[/bold green] [bold white]{self.current_step}/{self.total_steps}[/bold white] ({progress_pct:.1f}%)"
        )
        metrics_table.add_row(
            "Step Loss:",
            f"[bold green]{self.current_loss:.4f}[/bold green] (Avg: [yellow]{self.avg_loss:.4f}[/yellow])"
        )
        metrics_table.add_row("Learning Rate:", f"{self.learning_rate:.2e}")
        metrics_table.add_row("Grad Norm:", f"{self.grad_norm:.3f}")
        metrics_table.add_row("Elapsed / ETA:", f"[white]{elapsed_str}[/white] / [bold yellow]{eta_str}[/bold yellow] ({speed_str})")

        # Layout side-by-side or combined
        grid = Table.grid(expand=True)
        grid.add_column(ratio=1)
        grid.add_column(ratio=1)
        grid.add_row(
            Panel(gpu_table, title="[bold]🖥️  Hardware Status[/bold]", border_style="blue", box=box.ROUNDED),
            Panel(metrics_table, title="[bold]📊 Training Progress[/bold]", border_style="green", box=box.ROUNDED)
        )

        # Status & Recent Logs Panel
        status_text = Text()
        status_text.append("Status: ", style="bold cyan")
        status_text.append(f"{self.status_message}\n", style="italic white")
        if self.recent_logs:
            status_text.append("Recent Events:\n", style="bold yellow")
            for log in self.recent_logs:
                status_text.append(f" • {log}\n", style="dim white")

        logs_panel = Panel(status_text, title="[bold]📝 Activity Log[/bold]", border_style="cyan", box=box.ROUNDED)

        return Group(header, grid, logs_panel)
