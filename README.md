# Spark3Version (ModularFineTune)

Earlier prototype of a small LLM fine-tuning engine for NVIDIA GPUs: a YAML registry of models and
datasets plus a LoRA training run built on Unsloth, Hugging Face TRL and PyTorch. Kept as a
historical version.

## Status

Historical prototype, last code change November 2025. Only one model (`qwen-0.5b`) and a
two-example dummy dataset are set up. Training needs a CUDA GPU and was not re-run for this README.

## Quickstart

The project lives in the `Spark3Version/` subfolder.

```bash
git clone https://github.com/smlfg/Spark3Version.git
cd Spark3Version/Spark3Version
python3 -m venv venv
source venv/bin/activate
bash setup.sh                      # installs Unsloth, TRL, PEFT and requirements.txt
python scripts/init_defaults.py    # writes the qwen-0.5b config and a dummy dataset
python test_system.py              # checks that the registry finds model and dataset
```

`init_defaults.py` and `test_system.py` only need `pyyaml` and `pydantic` and run without a GPU.

## Usage

| Command | What it does |
|---|---|
| `python cli.py init` | Same as `scripts/init_defaults.py` |
| `python cli.py list` | Lists registered models and datasets |
| `python cli.py train <model> <dataset> --name <run>` | Starts a training run |
| `python training.py --name <run> [--model qwen-0.5b] [--dataset stackoverflow]` | Training run with GPU pre-flight check; writes `metadata.json` |

`cli.py` and `training.py` import Unsloth at startup, so both need the full GPU install from
`setup.sh`.

## How it works

- `registry/`: reads model configs from `models/<name>/config.yaml` and dataset metadata from
  `data/<name>/metadata.yaml` (Pydantic schemas).
- `core/model_loader.py`: loads the model with Unsloth, optionally in 4-bit.
- `core/trainer.py`: supervised fine-tuning with TRL `SFTTrainer`; saves the adapter and tokenizer
  to `experiments/<run>/`.
- `core/callbacks.py`: prints loss per step with short explanations of what the value means.
- `utils/paths.py`: project paths (`models/`, `data/`, `prompts/`, `experiments/`).

## Limitations

- Training uses fixed settings in `core/trainer.py` (60 steps, batch size 2, learning rate 2e-4).
  `training.py` accepts `--epochs`, `--max-steps`, `--batch-size` and `--save`, but these are not
  passed to the trainer.
- The only dataset is a generated dummy file with two examples.
- `data/` is git-ignored; run `init_defaults.py` before anything else.
- There are no automated tests besides `test_system.py`.

More notes in [Spark3Version/README.md](Spark3Version/README.md).

## License

MIT, see [LICENSE](LICENSE).
