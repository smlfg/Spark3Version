import sys
from pathlib import Path
import yaml

# Add root to path
sys.path.append(str(Path(__file__).parent.parent))
from utils.paths import MODELS_DIR, DATASETS_DIR, PROMPTS_DIR

def init():
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
        
    # 2. Dataset Structure
    ds_dir = DATASETS_DIR / "stackoverflow" / "slices"
    ds_dir.mkdir(parents=True, exist_ok=True)
    
    ds_meta = {
        "name": "stackoverflow",
        "format": "jsonl",
        "train_path": str(ds_dir / "slice_0000.json"),
        "split_size": 1000
    }
    
    with open(DATASETS_DIR / "stackoverflow" / "metadata.yaml", "w") as f:
        yaml.dump(ds_meta, f)

    print("✅ Defaults initialized.")

if __name__ == "__main__":
    init()
