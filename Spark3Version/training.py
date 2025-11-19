import argparse
import json
import time
import sys
import torch
from pathlib import Path
from datetime import datetime

# Internal Imports
from registry.manager import RegistryManager
from core.trainer import train_model
from utils.paths import EXPERIMENTS_DIR

def parse_args():
    parser = argparse.ArgumentParser(description="Spark3 Training Orchestrator")
    parser.add_argument("--model", type=str, default="qwen-0.5b", help="Model key from registry")
    parser.add_argument("--dataset", type=str, default="stackoverflow", help="Dataset key from registry")
    parser.add_argument("--name", type=str, required=True, help="Unique run name")
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--max-steps", type=int, default=-1, help="Max steps override (default: -1 uses epochs)")
    parser.add_argument("--batch-size", type=int, default=2, help="Per device batch size")
    parser.add_argument("--save", choices=["no", "last", "epoch"], default="last", help="Checkpoint strategy")
    return parser.parse_args()

def main():
    args = parse_args()
    print(f"🚀 INITIALIZING ORCHESTRATOR for Run: {args.name}")

    # 1. Pre-flight Checks
    if not torch.cuda.is_available():
        print("❌ CRITICAL: No CUDA device found. Training aborted.")
        sys.exit(1)
    
    print(f"✅ GPU Detected: {torch.cuda.get_device_name(0)} (BF16: {torch.cuda.is_bf16_supported()})")

    try:
        m_meta = RegistryManager.get_model(args.model)
        d_meta = RegistryManager.get_dataset(args.dataset)
    except FileNotFoundError as e:
        print(f"❌ Registry Error: {e}")
        sys.exit(1)

    if not Path(d_meta.train_path).exists():
        print(f"❌ Dataset File Missing: {d_meta.train_path}")
        sys.exit(1)

    # 2. Run Training
    start_time = time.time()
    
    print(f"⚙️  Starting Engine [Model: {m_meta.hf_path} | Data: {d_meta.name}]")
    
    # Pass args to core trainer (we need to update trainer signature next)
    try:
        train_model(
            model_name=args.model, 
            dataset_name=args.dataset, 
            run_name=args.name, 
            epochs=args.epochs
        )
    except Exception as e:
        print(f"❌ TRAINING FAILED: {e}")
        sys.exit(1)

    end_time = time.time()
    duration = end_time - start_time

    # 3. Metadata & Reporting
    out_dir = EXPERIMENTS_DIR / args.name
    out_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "timestamp": datetime.now().isoformat(),
        "experiment_name": args.name,
        "duration_seconds": round(duration, 2),
        "config": {
            "model": m_meta.dict(),
            "dataset": d_meta.dict(),
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "gpu": torch.cuda.get_device_name(0)
        }
    }

    with open(out_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n✅ RUN COMPLETE. Metadata saved to {out_dir}/metadata.json")
    print(f"   Adapter Path: {out_dir}")

if __name__ == "__main__":
    main()
