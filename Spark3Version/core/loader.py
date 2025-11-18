"""
Model and Tokenizer Loader for Unsloth-based LLM Fine-tuning.
"""

try:
    from unsloth import FastLanguageModel
except ImportError:
    raise ImportError("Unsloth not installed")


def load_model_and_tokenizer(model_name: str, max_seq_length: int = 2048):
    """
    Load a pre-trained model and tokenizer with 4-bit quantization and LoRA configuration.

    Args:
        model_name: HuggingFace model identifier (e.g., "unsloth/llama-3-8b-bnb-4bit")
        max_seq_length: Maximum sequence length for the model (default: 2048)

    Returns:
        tuple: (model, tokenizer) - The PEFT-configured model and its tokenizer

    Raises:
        ImportError: If unsloth is not installed
    """
    # Load model and tokenizer with 4-bit quantization
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=None,  # Auto-detect optimal dtype
        load_in_4bit=True  # Required for our hardware constraints
    )

    # Configure PEFT (LoRA) for efficient fine-tuning
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,  # LoRA rank
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # Attention projection layers
        lora_alpha=16,  # LoRA scaling factor
        lora_dropout=0,  # No dropout for stability
        bias="none",  # No bias training
        use_gradient_checkpointing="unsloth",  # Memory-efficient training
        random_state=3407,  # Reproducibility
        use_rslora=False,  # Standard LoRA
        loftq_config=None  # No LoftQ quantization
    )

    return model, tokenizer
