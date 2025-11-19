import torch
import sys
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import FastLanguageModel
from registry.manager import RegistryManager
from utils.paths import EXPERIMENTS_DIR
from core.model_loader import load_model_for_training
from core.callbacks import TeachingCallback

def train_model(model_name: str, dataset_name: str, run_name: str, epochs: int = 1):
    print(f"\n🚀 INITIALIZING TRAINING RUN: {run_name}")
    
    # 1. Load Metadata
    try:
        model_meta = RegistryManager.get_model(model_name)
        ds_meta = RegistryManager.get_dataset(dataset_name)
        print(f"📂 Configuration Loaded:\n   - Model: {model_meta.name}\n   - Dataset: {ds_meta.name}")
    except Exception as e:
        print(f"❌ Registry Error: {e}")
        return None

    # 2. Load Dataset
    print(f"📥 Loading Data from: {ds_meta.train_path}")
    try:
        # Load local JSONL file
        dataset = load_dataset("json", data_files=str(ds_meta.train_path), split="train")
        print(f"   -> Loaded {len(dataset)} samples.")
    except Exception as e:
        print(f"❌ Data Load Error: {e}")
        return None
    
    # 3. Load Model (Unsloth Engine)
    try:
        model, tokenizer = load_model_for_training(
            model_name=model_meta.hf_path,
            load_in_4bit=model_meta.load_in_4bit
        )
    except Exception as e:
        print(f"❌ Model Load Error (CUDA/Unsloth): {e}")
        return None

    # 4. Prepare Output Path
    output_path = EXPERIMENTS_DIR / run_name
    print(f"💾 Checkpoints will be saved to: {output_path}")
    
    # 5. Configure Trainer
    print("⚙️  Configuring SFTTrainer parameters...")
    training_args = TrainingArguments(
        output_dir=str(output_path),
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        # Demo Settings: Short run
        max_steps=60 if epochs < 100 else epochs * len(dataset), 
        learning_rate=2e-4,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        report_to="none", # Disable WandB for local demo
    )

    # 6. Initialize Trainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text", # Expects 'text' column in JSONL
        max_seq_length=2048,
        dataset_num_proc=2,
        packing=False,
        args=training_args,
        callbacks=[TeachingCallback()]
    )

    # 7. Start Training
    print("\n🔥 ENGINE START. TEACHING IN PROGRESS...\n")
    try:
        trainer_stats = trainer.train()
    except Exception as e:
        print(f"\n❌ TRAINING CRASHED: {e}")
        return None

    # 8. Save Artifacts
    print(f"\n💾 Saving Adapter Model to {output_path}...")
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    
    print("✅ TRAINING COMPLETE. SYSTEM SHUTDOWN.")
    return trainer_stats

if __name__ == "__main__":
    # Debug Run
    print("🧪 TEST MODE: Running training loop check...")
    train_model("qwen-0.5b", "stackoverflow", "debug_run_001")