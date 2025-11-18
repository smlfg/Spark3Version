# ModularFineTune (MFT)

**ModularFineTune** ist ein modulares Framework zum Finetuning von Large Language Models (LLMs). Es bietet eine flexible und erweiterbare Architektur, die es ermöglicht, verschiedene Modelle, Datasets und Prompt-Strategien einfach zu kombinieren und zu verwalten.

## Features

- **Modulare Architektur**: Einfache Integration neuer Modelle, Datasets und Prompts
- **CLI-basiert**: Intuitive Kommandozeilen-Schnittstelle für alle Operationen
- **Konfigurierbar**: Zentrale Konfiguration über YAML-Dateien
- **Erweiterbar**: Plugin-System für benutzerdefinierte Komponenten

## Installation

```bash
pip install -r requirements.txt
```

## Projektstruktur

```
Spark3Version/
├── models/         # Modell-Definitionen und Adapter
├── datasets/       # Dataset-Loader und Preprocessing-Pipelines
├── prompts/        # Prompt-Templates und Strategien
├── core/           # Kern-Funktionalität und Basis-Klassen
├── registry/       # Komponenten-Registry für Plugins
├── cli.py          # Kommandozeilen-Interface
└── config.yaml     # Zentrale Konfigurationsdatei
```

### Ordner-Beschreibungen

#### `models/`
Enthält die Modell-Definitionen und Adapter für verschiedene LLM-Architekturen. Hier werden die zu trainierenden Modelle registriert und konfiguriert (z.B. Qwen, LLaMA, Mistral).

#### `datasets/`
Beinhaltet Dataset-Loader und Preprocessing-Pipelines. Datasets werden hier definiert und für das Training vorbereitet (z.B. StackOverflow, CustomDatasets).

#### `prompts/`
Verwaltet Prompt-Templates und -Strategien. Hier können verschiedene Prompt-Engineering-Ansätze für spezifische Anwendungsfälle definiert werden.

## Quickstart

### Verfügbare Modelle auflisten

```bash
python cli.py list models
```

### Training starten

```bash
python cli.py train --model qwen-0.5b --dataset stackoverflow
```

### Weitere Befehle

```bash
# Verfügbare Datasets anzeigen
python cli.py list datasets

# Verfügbare Prompts anzeigen
python cli.py list prompts

# Hilfe anzeigen
python cli.py --help
```

## Konfiguration

Die zentrale Konfiguration erfolgt über `Spark3Version/config.yaml`. Hier können globale Einstellungen wie Modell-Parameter, Training-Hyperparameter und Pfade definiert werden.

## Eigene Komponenten hinzufügen

### Neues Modell registrieren

Erstellen Sie eine neue Datei in `Spark3Version/models/` und registrieren Sie Ihr Modell über die Registry.

### Neues Dataset hinzufügen

Implementieren Sie einen Dataset-Loader in `Spark3Version/datasets/` und registrieren Sie ihn im System.

### Neue Prompt-Strategie

Definieren Sie Ihre Prompt-Templates in `Spark3Version/prompts/` und machen Sie sie über die CLI verfügbar.

## Entwicklung

```bash
# Repository klonen
git clone <repository-url>
cd Spark3Version

# Entwicklungs-Abhängigkeiten installieren
pip install -r requirements.txt

# Tests ausführen (wenn vorhanden)
pytest tests/
```

## Lizenz

[Lizenz hier angeben]

## Kontakt & Beiträge

Beiträge sind willkommen! Bitte erstellen Sie einen Pull Request oder öffnen Sie ein Issue für Vorschläge und Bugfixes.
