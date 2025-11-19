#!/bin/bash
set -e  # Stoppt das Skript bei Fehlern sofort

echo "========================================"
echo "🛠️  MFT SETUP: Installing Dependencies"
echo "========================================"

# 0. Check Python
python3 --version
echo "   -> Python detected."

# 1. Upgrade pip (Critical for wheel building)
echo -e "\n[1/4] Upgrading pip..."
pip install --upgrade pip

# 2. Install Unsloth (The tricky part)
echo -e "\n[2/4] Installing Unsloth (GPU Optimized)..."
# Wir nutzen die 'colab-new' Version, da sie am stabilsten für Torch 2.2+ ist
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"

# 3. Install TRL & PEFT (No deps to avoid conflict with Unsloth)
echo -e "\n[3/4] Installing TRL, PEFT, Accelerate..."
pip install --no-deps "trl<0.9.0" peft accelerate bitsandbytes

# 4. Install Rest from requirements.txt
echo -e "\n[4/4] Installing Project Requirements..."
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
else
    echo "⚠️ requirements.txt not found! Skipping."
fi

echo "========================================"
echo "✅ SETUP COMPLETE! Ready to train."
echo "   Run: python core/trainer.py"
echo "========================================"
