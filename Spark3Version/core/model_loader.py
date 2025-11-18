import torch
try:
    from unsloth import FastLanguageModel
except ImportError:
    FastLanguageModel = None
    print("⚠️ WARNING: Unsloth not installed. Running in Mock Mode creates errors later.")

def load_model_for_training(model_name: str, max_seq_length: int = 2048, load_in_4bit: bool = True):
    if FastLanguageModel is None:
        raise ImportError("Unsloth is not installed!")

    print(f"⏳ Loading model: {model_name}...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=None,
        load_in_4bit=load_in_4bit,
    )

    print("🔧 Attaching LoRA adapters...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )
    
    return model, tokenizer
