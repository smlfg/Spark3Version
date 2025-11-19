# 🧠 ModularFineTune (MFT)

**An Educational LLM Fine-Tuning Engine specialized for NVIDIA GPUs.**
*Powered by Unsloth, PyTorch & HuggingFace TRL.*

## 🚀 Quick Start on DGX/Uni-GPU

### 1. Setup
Clone the repo and run the installer script:
```bash
git clone https://github.com/smlfg/Spark3Version.git
cd Spark3Version

# Create venv (Optional but recommended)
python -m venv venv
source venv/bin/activate

# Install everything automatically
bash setup.sh
```

### 2. Usage
```bash
# List options
python cli.py list

# Start Training
python core/trainer.py
```