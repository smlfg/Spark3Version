#!/usr/bin/env python3
"""
ModularFineTune CLI - Professional CLI for model fine-tuning.

A beautiful, feature-rich command-line interface built with Typer and Rich.
"""
import typer
from typing import Optional
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich import box
from rich.text import Text
import time

from Spark3Version.registry.manager import RegistryManager
from Spark3Version.core.trainer import train_model

# Initialize Typer app and Rich console
app = typer.Typer(
    name="mft",
    help="🚀 ModularFineTune - Professional model fine-tuning made simple",
    add_completion=False,
    rich_markup_mode="rich"
)
console = Console()


@app.command(name="list")
def list_resources():
    """
    📋 List all available models and datasets.

    Scans the registry to display all registered models and datasets
    in a beautiful side-by-side table format.
    """
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True
    ) as progress:
        task = progress.add_task("🔍 Scanning registry...", total=None)

        # Initialize registry manager
        registry = RegistryManager()

        # Fetch models and datasets
        models = registry.get_available_models()
        datasets = registry.get_available_datasets()

        progress.update(task, completed=True)

    # Create a stunning table
    table = Table(
        title="📦 Available Resources",
        box=box.ROUNDED,
        title_style="bold cyan",
        header_style="bold magenta",
        border_style="bright_blue",
        show_lines=True
    )

    table.add_column("🤖 Available Models", style="cyan", no_wrap=False)
    table.add_column("📊 Available Datasets", style="green", no_wrap=False)

    # Pad to equal length
    max_rows = max(len(models), len(datasets))
    models_padded = models + [""] * (max_rows - len(models))
    datasets_padded = datasets + [""] * (max_rows - len(datasets))

    for model, dataset in zip(models_padded, datasets_padded):
        table.add_row(model, dataset)

    console.print()
    console.print(table)
    console.print()

    # Summary panel
    summary = Panel(
        f"[bold]Total:[/bold] {len(models)} models • {len(datasets)} datasets",
        border_style="dim",
        box=box.ROUNDED
    )
    console.print(summary)


@app.command(name="train")
def train(
    model: str = typer.Option(
        ...,
        "--model", "-m",
        help="🤖 Model name to fine-tune"
    ),
    dataset: str = typer.Option(
        ...,
        "--dataset", "-d",
        help="📊 Dataset name to use for training"
    ),
    epochs: int = typer.Option(
        1,
        "--epochs", "-e",
        help="🔄 Number of training epochs",
        min=1
    ),
    name: Optional[str] = typer.Option(
        None,
        "--name", "-n",
        help="📝 Experiment name (auto-generated if not provided)"
    ),
    learning_rate: float = typer.Option(
        2e-4,
        "--lr",
        help="📈 Learning rate"
    ),
    batch_size: int = typer.Option(
        4,
        "--batch-size", "-b",
        help="📦 Training batch size",
        min=1
    )
):
    """
    🚀 Train a model on a dataset.

    Initiates a fine-tuning training run with the specified configuration.
    Experiment results will be saved for later evaluation.

    Example:
        mft train --model phi-2 --dataset gsm8k --epochs 3 --name my_experiment
    """
    # Display training header
    console.print()
    console.print(Panel.fit(
        "🚀 [bold cyan]ModularFineTune Training[/bold cyan]",
        border_style="cyan"
    ))
    console.print()

    # Show configuration
    config_table = Table(
        box=box.SIMPLE,
        show_header=False,
        border_style="dim"
    )
    config_table.add_column("Parameter", style="bold yellow")
    config_table.add_column("Value", style="white")

    config_table.add_row("🤖 Model", model)
    config_table.add_row("📊 Dataset", dataset)
    config_table.add_row("🔄 Epochs", str(epochs))
    config_table.add_row("📈 Learning Rate", f"{learning_rate:.0e}")
    config_table.add_row("📦 Batch Size", str(batch_size))
    if name:
        config_table.add_row("📝 Experiment Name", name)

    console.print(config_table)
    console.print()

    # Training initialization with status
    with console.status(
        "[bold green]⚙️  Initializing training pipeline...",
        spinner="dots"
    ) as status:
        time.sleep(0.5)  # Brief pause for visual effect

        try:
            # Start training
            results = train_model(
                model=model,
                dataset=dataset,
                epochs=epochs,
                name=name,
                learning_rate=learning_rate,
                batch_size=batch_size
            )

            status.update("[bold green]✅ Training initialized successfully!")
            time.sleep(0.3)

        except Exception as e:
            console.print(f"\n[bold red]❌ Error:[/bold red] {str(e)}")
            raise typer.Exit(code=1)

    # Display results
    console.print()
    result_panel = Panel(
        f"[bold green]✅ Training Configuration Complete[/bold green]\n\n"
        f"[cyan]Experiment:[/cyan] {results['experiment_name']}\n"
        f"[cyan]Output Directory:[/cyan] {results['experiment_dir']}\n"
        f"[cyan]Status:[/cyan] {results['status']}\n\n"
        f"[dim]{results.get('message', '')}[/dim]",
        title="📊 Training Results",
        border_style="green",
        box=box.ROUNDED
    )
    console.print(result_panel)
    console.print()


@app.command(name="test")
def test(
    experiment_name: str = typer.Option(
        ...,
        "--experiment-name", "-e",
        help="🧪 Name of the experiment to test"
    ),
    test_dataset: Optional[str] = typer.Option(
        None,
        "--test-dataset", "-d",
        help="📊 Optional test dataset (uses training dataset if not specified)"
    ),
    output_file: Optional[str] = typer.Option(
        None,
        "--output", "-o",
        help="💾 Output file for test results"
    )
):
    """
    🧪 Test a trained model (PLACEHOLDER).

    Evaluate a trained model on a test dataset. This functionality
    will call core.tester.run_test when implemented.

    Example:
        mft test --experiment-name my_experiment --test-dataset test_data
    """
    console.print()
    console.print(Panel.fit(
        "🧪 [bold yellow]Model Testing[/bold yellow]",
        border_style="yellow"
    ))
    console.print()

    # Configuration display
    info_table = Table(box=box.SIMPLE, show_header=False)
    info_table.add_column("Parameter", style="bold yellow")
    info_table.add_column("Value", style="white")

    info_table.add_row("🧪 Experiment", experiment_name)
    if test_dataset:
        info_table.add_row("📊 Test Dataset", test_dataset)
    if output_file:
        info_table.add_row("💾 Output File", output_file)

    console.print(info_table)
    console.print()

    # Placeholder message
    placeholder_panel = Panel(
        "[bold yellow]⚠️  Testing Functionality Coming Soon[/bold yellow]\n\n"
        f"[white]This command will evaluate the experiment:[/white] [cyan]{experiment_name}[/cyan]\n\n"
        "[dim]The testing module (core.tester.run_test) will be implemented soon.\n"
        "This will include:\n"
        "  • Model loading and inference\n"
        "  • Metric calculation (accuracy, perplexity, etc.)\n"
        "  • Result visualization and export[/dim]",
        title="🚧 Under Construction",
        border_style="yellow",
        box=box.ROUNDED
    )
    console.print(placeholder_panel)
    console.print()

    console.print("[dim]💡 Tip: Use 'mft list' to see available resources[/dim]")
    console.print()


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    version: bool = typer.Option(
        False,
        "--version", "-v",
        help="Show version information"
    )
):
    """
    🚀 ModularFineTune CLI

    A professional, feature-rich CLI for model fine-tuning and evaluation.
    Built with ❤️ using Typer and Rich.
    """
    if version:
        console.print()
        console.print(Panel(
            "[bold cyan]ModularFineTune CLI[/bold cyan]\n"
            "Version: [yellow]1.0.0[/yellow]\n"
            "Built with: [green]Typer[/green] • [magenta]Rich[/magenta]",
            border_style="cyan"
        ))
        console.print()
        raise typer.Exit()

    if ctx.invoked_subcommand is None:
        # Show welcome message when no command is provided
        console.print()
        console.print(Panel.fit(
            "🚀 [bold cyan]Welcome to ModularFineTune![/bold cyan]\n\n"
            "[white]Available Commands:[/white]\n"
            "  • [green]list[/green]  - View available models and datasets\n"
            "  • [green]train[/green] - Start a fine-tuning training run\n"
            "  • [green]test[/green]  - Test a trained model\n\n"
            "[dim]Run [bold]mft --help[/bold] for more information[/dim]",
            border_style="cyan"
        ))
        console.print()


if __name__ == "__main__":
    app()
