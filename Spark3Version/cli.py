import typer
from rich.console import Console
from rich.table import Table
from registry.manager import RegistryManager
from core.trainer import train_model
import scripts.init_defaults

app = typer.Typer()
console = Console()

@app.command()
def init():
    """Initialize default models and datasets."""
    scripts.init_defaults.init()
    console.print("[green]Defaults initialized.[/green]")

@app.command()
def list():
    """List available assets."""
    table = Table(title="MFT Registry")
    table.add_column("Category", style="cyan")
    table.add_column("Name", style="green")
    
    for m in RegistryManager.list_models():
        table.add_row("Model", m)
    for d in RegistryManager.list_datasets():
        table.add_row("Dataset", d)
    console.print(table)

@app.command()
def train(model: str, dataset: str, name: str = "experiment"):
    """Start training run."""
    train_model(model, dataset, name)

if __name__ == "__main__":
    app()
