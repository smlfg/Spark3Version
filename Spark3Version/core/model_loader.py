"""
Model Loader for Unsloth-based LLM Training.
"""

try:
    from unsloth import FastLanguageModel
except ImportError:
    raise ImportError("Unsloth not installed")


def load_model_for_training(model_name: str, max_seq_length: int = 2048, load_in_4bit: bool = True):
    """
    Load model and tokenizer for training with LoRA configuration.

    Args:
        model_name: HuggingFace model identifier
        max_seq_length: Maximum sequence length (default: 2048)
        load_in_4bit: Enable 4-bit quantization (default: True)

    Returns:
        tuple: (model, tokenizer)
    """
    # Load pretrained model and tokenizer
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=None,
        load_in_4bit=load_in_4bit
    )

    # Apply PEFT/LoRA configuration
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
        use_rslora=False,
        loftq_config=None
    )

    return model, tokenizer
