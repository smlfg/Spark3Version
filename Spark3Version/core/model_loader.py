from unsloth import FastLanguageModel # MUST BE FIRST IMPORT
import torch

def load_model_for_training(model_name: str, max_seq_length: int = 2048, load_in_4bit: bool = True):
    print(f"⏳ Loading Unsloth Model: {model_name}...")
    try:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=max_seq_length,
            dtype=None,
            load_in_4bit=load_in_4bit,
        )
    except Exception as e:
        raise ImportError(f"Failed to load model via Unsloth: {e}")

    print("🔧 Attaching LoRA Adapters...")
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