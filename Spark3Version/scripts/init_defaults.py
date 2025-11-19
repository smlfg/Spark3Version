import sys
import json
from pathlib import Path
import yaml

# Add root to path
sys.path.append(str(Path(__file__).parent.parent))
from utils.paths import MODELS_DIR, DATASETS_DIR

def init():
    print("⚙️  Initializing Defaults...")
    
    # 1. Qwen Model Config
    qwen_dir = MODELS_DIR / "qwen-0.5b"
    qwen_dir.mkdir(parents=True, exist_ok=True)
    
    qwen_config = {
        "name": "qwen-0.5b",
        "hf_path": "unsloth/Qwen1.5-0.5B-bnb-4bit",
        "model_type": "causal",
        "chat_template": "chatml",
        "load_in_4bit": True,
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"]
    }
    
    with open(qwen_dir / "config.yaml", "w") as f:
        yaml.dump(qwen_config, f)
        
    # 2. Dataset Structure & Dummy File
    ds_dir = DATASETS_DIR / "stackoverflow" / "slices"
    ds_dir.mkdir(parents=True, exist_ok=True)
    
    # Create Dummy JSONL if missing
    dummy_file = ds_dir / "slice_0000.json"
    if not dummy_file.exists():
        print("   -> Creating dummy dataset file...")
        data = [
            {"text": "User: Hello\nAssistant: Hi there! How can I help with Python?"},
            {"text": "User: Print hello\nAssistant: print('Hello')"}
        ]
        with open(dummy_file, "w") as f:
            json.dump(data, f, indent=2)

    ds_meta = {
        "name": "stackoverflow",
        "format": "jsonl",
        "train_path": str(dummy_file),
        "split_size": 1000
    }
    
    with open(DATASETS_DIR / "stackoverflow" / "metadata.yaml", "w") as f:
        yaml.dump(ds_meta, f)

    print("✅ Defaults initialized successfully.")

if __name__ == "__main__":
    init()