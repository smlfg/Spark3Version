#!/usr/bin/env python3
"""ModularFineTune CLI - High-End CLI for model fine-tuning."""
import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box

from Spark3Version.registry.manager import RegistryManager
from Spark3Version.core.trainer import train_model
# from Spark3Version.core.tester import run_test  # TODO: Implement next

app = typer.Typer(
    name="mft",
    help="ModularFineTune - Professional model fine-tuning CLI",
    add_completion=False
)
console = Console()


@app.command()
def list():
    """List all available models and datasets."""
    registry = RegistryManager()

    # Get detailed model and dataset information
    models = registry.get_models_detailed()
    datasets = registry.get_datasets_detailed()

    console.print()

    # Table 1: Models
    models_table = Table(
        title="Models",
        box=box.ROUNDED,
        title_style="bold cyan",
        header_style="bold magenta",
        border_style="cyan"
    )
    models_table.add_column("Name", style="cyan", no_wrap=False)
    models_table.add_column("Path", style="white", no_wrap=False)
    models_table.add_column("Type", style="yellow", no_wrap=False)

    if models:
        for model in models:
            models_table.add_row(
                model['name'],
                model['path'],
                model['type']
            )
    else:
        models_table.add_row("No models registered", "-", "-")

    console.print(models_table)
    console.print()

    # Table 2: Datasets
    datasets_table = Table(
        title="Datasets",
        box=box.ROUNDED,
        title_style="bold green",
        header_style="bold magenta",
        border_style="green"
    )
    datasets_table.add_column("Name", style="green", no_wrap=False)
    datasets_table.add_column("Path", style="white", no_wrap=False)

    if datasets:
        for dataset in datasets:
            datasets_table.add_row(
                dataset['name'],
                dataset['path']
            )
    else:
        datasets_table.add_row("No datasets registered", "-")

    console.print(datasets_table)
    console.print()


@app.command()
def train(
    model: str = typer.Option(..., "--model", "-m", help="Model name"),
    dataset: str = typer.Option(..., "--dataset", "-d", help="Dataset name"),
    epochs: int = typer.Option(1, "--epochs", "-e", help="Number of epochs"),
    name: str = typer.Option("demo", "--name", "-n", help="Experiment name")
):
    """Train a model on a dataset."""
    console.print()
    console.print(Panel.fit(
        f"[bold cyan]Training Configuration[/bold cyan]",
        border_style="cyan"
    ))
    console.print()

    # Display configuration
    config_table = Table(box=box.SIMPLE, show_header=False)
    config_table.add_column("Parameter", style="bold yellow")
    config_table.add_column("Value", style="white")

    config_table.add_row("Model", model)
    config_table.add_row("Dataset", dataset)
    config_table.add_row("Epochs", str(epochs))
    config_table.add_row("Experiment", name)

    console.print(config_table)
    console.print()

    # Training with status spinner
    try:
        with console.status(
            "[bold green]Initializing Unsloth Engine...",
            spinner="dots"
        ):
            results = train_model(
                model=model,
                dataset=dataset,
                epochs=epochs,
                name=name
            )

        # Display success
        console.print()
        console.print(Panel(
            f"[bold green]Training initialized successfully![/bold green]\n\n"
            f"Experiment: [cyan]{results['experiment_name']}[/cyan]\n"
            f"Output: [dim]{results['experiment_dir']}[/dim]",
            title="Success",
            border_style="green",
            box=box.ROUNDED
        ))
        console.print()

    except Exception as e:
        console.print()
        console.print(f"[bold red]Error: {str(e)}[/bold red]")
        console.print()
        raise typer.Exit(code=1)


@app.command()
def test(
    experiment: str = typer.Option(..., "--experiment", "-e", help="Experiment name")
):
    """Test a trained model."""
    console.print()
    console.print(Panel.fit(
        f"[bold yellow]Testing Experiment: {experiment}[/bold yellow]",
        border_style="yellow"
    ))
    console.print()

    # TODO: Uncomment when core.tester is implemented
    # try:
    #     with console.status("[bold green]Running tests...", spinner="dots"):
    #         results = run_test(experiment_name=experiment)
    #
    #     console.print(Panel(
    #         f"[bold green]Testing complete![/bold green]\n\n"
    #         f"Results: {results}",
    #         title="Test Results",
    #         border_style="green"
    #     ))
    # except Exception as e:
    #     console.print(f"[bold red]Error: {str(e)}[/bold red]")
    #     raise typer.Exit(code=1)

    # Placeholder
    console.print("[yellow]Testing functionality will be implemented soon.[/yellow]")
    console.print(f"[dim]This will call: core.tester.run_test(experiment_name='{experiment}')[/dim]")
    console.print()


if __name__ == "__main__":
    app()
