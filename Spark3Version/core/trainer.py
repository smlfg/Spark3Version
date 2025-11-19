import torch
from unsloth import FastLanguageModel  # <--- MOVED TO TOP (Critical speedup)
import sys
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments
from registry.manager import RegistryManager
from utils.paths import EXPERIMENTS_DIR
from core.model_loader import load_model_for_training
from core.callbacks import TeachingCallback

def train_model(model_name: str, dataset_name: str, run_name: str, epochs: int = 1):
    print(f"\n🚀 STARTING EXPERIMENT: {run_name}")
    
    # 1. Config
    try:
        model_meta = RegistryManager.get_model(model_name)
        ds_meta = RegistryManager.get_dataset(dataset_name)
    except Exception as e:
        print(f"❌ Config Error: {e}")
        return

    # 2. Data
    print(f"📥 Loading Dataset: {ds_meta.name}")
    try:
        dataset = load_dataset("json", data_files=str(ds_meta.train_path), split="train")
    except Exception as e:
        print(f"❌ Data Error: {e}")
        return

    # 3. Model
    try:
        model, tokenizer = load_model_for_training(model_meta.hf_path, model_meta.load_in_4bit)
    except Exception as e:
        print(f"❌ Model Load Error: {e}")
        return

    # 4. Trainer
    output_dir = EXPERIMENTS_DIR / run_name
    print(f"⚙️  Configuring Trainer (Output: {output_dir})...")
    
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        max_steps=60, 
        learning_rate=2e-4,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=1,
        optim="adamw_8bit",
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=2048,
        args=training_args,
        callbacks=[TeachingCallback()]
    )

    print("\n🔥 ENGINE IGNITION...")
    trainer.train()
    
    print(f"💾 Saving adapter to {output_dir}")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("✅ TRAINING COMPLETE.")

if __name__ == "__main__":
    import scripts.init_defaults
    scripts.init_defaults.init()
    train_model("qwen-0.5b", "stackoverflow", "manual_run_v1")