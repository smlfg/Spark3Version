# ModularFineTune (MFT)

Ein modulares Framework zum Finetuning von Large Language Models (LLMs) mit Fokus auf Flexibilität und Erweiterbarkeit.

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Projektstruktur

Das Projekt ist in drei Hauptkomponenten organisiert:

### `models/`
Enthält Modell-Definitionen und Adapter für verschiedene LLM-Architekturen. Hier werden die zu trainierenden Modelle registriert und konfiguriert (z.B. Qwen, LLaMA, Mistral).

### `datasets/`
Beinhaltet Dataset-Loader und Preprocessing-Pipelines. Datasets werden hier definiert und für das Training vorbereitet (z.B. StackOverflow, Custom Datasets).

### `prompts/`
Verwaltet Prompt-Templates und -Strategien. Hier können verschiedene Prompt-Engineering-Ansätze für spezifische Anwendungsfälle definiert werden.

---

## Quickstart

### Verfügbare Modelle auflisten

```bash
python cli.py list models
```

### Training starten

```bash
python cli.py train --model qwen-0.5b --dataset stackoverflow
```

---

## Weitere Informationen

```bash
# Hilfe anzeigen
python cli.py --help

# Training-Hilfe
python cli.py train --help
```

---

## Lizenz

[Lizenz hier angeben]
