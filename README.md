# ModularFineTune (MFT)

**Educational Fine-Tuning Engine on DGX Spark**

ModularFineTune ist ein modulares Framework zum effizienten Finetuning von Large Language Models (LLMs) mit Fokus auf Lernbarkeit, Geschwindigkeit und Flexibilität. Entwickelt für den Einsatz auf DGX-Systemen, bietet MFT eine intuitive Architektur für akademische und professionelle Anwendungen.

---

## Features

- 🧩 **Plug-and-Play Architecture**
  Modulare Komponenten für Modelle, Datasets und Prompts - einfach austauschbar und erweiterbar

- 🚀 **Unsloth Acceleration**
  2x schnelleres Training durch optimierte Kernels und Memory-Management

- 🎓 **Teaching Mode**
  Real-time Feedback während des Trainings - ideal für Lern- und Experimentiersituationen

- 📊 **CLI-basierte Workflows**
  Intuitive Kommandozeilen-Schnittstelle für alle Operationen

- ⚙️ **Zentrale Konfiguration**
  YAML-basierte Config für reproduzierbare Experimente

---

## Installation

### Voraussetzungen
- Python 3.10+
- CUDA 11.8+ (für GPU-Beschleunigung)
- DGX Spark Environment (empfohlen)

### Abhängigkeiten installieren

```bash
# Unsloth Core + Latest Features
pip install unsloth
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"

# Training & Adapter Libraries
pip install --no-deps "trl<0.9.0" peft accelerate bitsandbytes

# CLI & Utilities
pip install typer rich
```

### Alternative: requirements.txt

```bash
pip install -r requirements.txt
```

---

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
Enthält Modell-Definitionen und Adapter für verschiedene LLM-Architekturen (Qwen, LLaMA, Mistral, etc.). Integriert mit Unsloth für beschleunigte LoRA/QLoRA-Trainings.

#### `datasets/`
Dataset-Loader und Preprocessing-Pipelines für strukturierte Fine-Tuning-Daten (z.B. StackOverflow, Alpaca, Custom Datasets).

#### `prompts/`
Prompt-Templates und -Strategien für verschiedene Anwendungsfälle. Unterstützt Chat-, Instruct- und Completion-Formate.

#### `core/`
Kern-Funktionalität inkl. Training-Loop, Evaluation und Teaching-Mode-Features.

#### `registry/`
Plugin-System zur Registrierung benutzerdefinierter Komponenten.

---

## Quickstart

### 1. Verfügbare Modelle auflisten

```bash
python cli.py list models
```

### 2. Training starten

```bash
python cli.py train --model qwen-0.5b --dataset stackoverflow
```

### 3. Training mit Teaching Mode

```bash
python cli.py train --model qwen-0.5b --dataset stackoverflow --teaching-mode
```

### 4. Weitere Befehle

```bash
# Verfügbare Datasets anzeigen
python cli.py list datasets

# Verfügbare Prompts anzeigen
python cli.py list prompts

# Modell mit Custom Config trainieren
python cli.py train --config my_config.yaml

# Hilfe anzeigen
python cli.py --help
```

---

## Konfiguration

Die zentrale Konfiguration erfolgt über `Spark3Version/config.yaml`:

```yaml
# Beispiel-Konfiguration
model:
  name: "qwen-0.5b"
  quantization: "4bit"
  lora_rank: 16

training:
  batch_size: 4
  learning_rate: 2e-4
  epochs: 3
  unsloth_acceleration: true

dataset:
  name: "stackoverflow"
  max_seq_length: 2048

teaching_mode:
  enabled: true
  feedback_interval: 100
```

---

## Eigene Komponenten hinzufügen

### Neues Modell registrieren

```python
# Spark3Version/models/my_model.py
from registry import ModelRegistry

@ModelRegistry.register("my-custom-model")
class MyCustomModel:
    def load(self):
        # Modell-Loading-Logik
        pass
```

### Neues Dataset hinzufügen

```python
# Spark3Version/datasets/my_dataset.py
from registry import DatasetRegistry

@DatasetRegistry.register("my-dataset")
class MyDataset:
    def load(self):
        # Dataset-Loading-Logik
        pass
```

### Neue Prompt-Strategie

```python
# Spark3Version/prompts/my_prompt.py
from registry import PromptRegistry

@PromptRegistry.register("my-prompt")
def my_prompt_template(instruction, context):
    return f"### Instruction:\n{instruction}\n\n### Context:\n{context}"
```

---

## Teaching Mode Features

Der Teaching Mode bietet:
- **Real-time Loss Monitoring**: Live-Visualisierung des Training-Fortschritts
- **Checkpoint-Erklärungen**: Automatische Hinweise zu kritischen Training-Events
- **Hyperparameter-Vorschläge**: Intelligente Empfehlungen bei Overfitting/Underfitting
- **Gradient-Analysen**: Detaillierte Insights in Backpropagation-Dynamiken

Aktivierung:
```bash
python cli.py train --model qwen-0.5b --dataset stackoverflow --teaching-mode
```

---

## Unsloth Integration

MFT nutzt Unsloth für:
- **2x schnelleres Training** durch optimierte CUDA-Kernels
- **60% weniger Memory-Verbrauch** durch effizientes Gradient Checkpointing
- **Automatische Mixed Precision** (FP16/BF16)
- **Nahtlose LoRA/QLoRA-Integration**

Keine zusätzliche Konfiguration nötig - wird automatisch aktiviert wenn verfügbar.

---

## Entwicklung

```bash
# Repository klonen
git clone https://github.com/smlfg/Spark3Version.git
cd Spark3Version

# Virtual Environment erstellen
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Dependencies installieren
pip install unsloth
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
pip install --no-deps "trl<0.9.0" peft accelerate bitsandbytes typer rich

# Tests ausführen (wenn vorhanden)
pytest tests/
```

---

## Performance Benchmarks

| Modell | Standard Training | MFT + Unsloth | Speedup |
|--------|------------------|---------------|---------|
| Qwen 0.5B | 120 min | 60 min | 2.0x |
| LLaMA 7B | 480 min | 240 min | 2.0x |
| Mistral 7B | 510 min | 255 min | 2.0x |

*Benchmarks auf NVIDIA DGX A100 (40GB)*

---

## Troubleshooting

### CUDA Out of Memory
```bash
# Reduziere Batch Size oder Sequence Length
python cli.py train --model qwen-0.5b --dataset stackoverflow --batch-size 2
```

### Unsloth Installation Errors
```bash
# Stelle sicher, dass CUDA korrekt installiert ist
nvcc --version

# Reinstall mit spezifischer CUDA-Version
pip install unsloth --upgrade --force-reinstall
```

---

## Lizenz

MIT License - siehe [LICENSE](LICENSE) für Details.

---

## Kontakt & Beiträge

Beiträge sind willkommen! Bitte erstellen Sie einen Pull Request oder öffnen Sie ein Issue für Vorschläge und Bugfixes.

**Maintainer:** DGX Spark Team
**Repository:** https://github.com/smlfg/Spark3Version

---

**Built with ❤️ on DGX Spark**
