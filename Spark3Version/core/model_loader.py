"""
Model Loader for efficient LLM loading with Unsloth.

This module provides utilities for loading language models optimized
for training with Unsloth's FastLanguageModel.
"""

try:
    from unsloth import FastLanguageModel
except ImportError:
    # Fallback if unsloth is not installed
    class FastLanguageModel:
        """Dummy FastLanguageModel class."""
        @staticmethod
        def from_pretrained(*args, **kwargs):
            raise ImportError(
                "unsloth is not installed. Please install it with: "
                "pip install unsloth"
            )


def load_model_for_training(
    model_name: str,
    max_seq_length: int = 2048,
    load_in_4bit: bool = True
):
    """
    Load a language model and tokenizer for training using Unsloth.

    This function uses Unsloth's FastLanguageModel to efficiently load
    models with optional 4-bit quantization for memory efficiency.

    Args:
        model_name: Name or path of the model to load.
                   Example: "unsloth/Qwen1.5-0.5B-bnb-4bit"
        max_seq_length: Maximum sequence length for the model.
                       Default: 2048
        load_in_4bit: Whether to load the model in 4-bit quantization.
                     This significantly reduces memory usage.
                     Default: True

    Returns:
        tuple: (model, tokenizer) - The loaded model and tokenizer objects

    Raises:
        ImportError: If unsloth is not installed
        Exception: If model loading fails

    Example:
        >>> model, tokenizer = load_model_for_training(
        ...     model_name="unsloth/Qwen1.5-0.5B-bnb-4bit",
        ...     max_seq_length=2048,
        ...     load_in_4bit=True
        ... )
    """
    print(f"🔄 Loading model: {model_name}")
    print(f"   Max sequence length: {max_seq_length}")
    print(f"   4-bit quantization: {'Enabled' if load_in_4bit else 'Disabled'}")

    try:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=max_seq_length,
            dtype=None,  # Auto-detect optimal dtype
            load_in_4bit=load_in_4bit,
        )

        print(f"✅ Model loaded successfully!")
        print(f"   Model type: {type(model).__name__}")
        print(f"   Tokenizer vocab size: {len(tokenizer)}")

        return model, tokenizer

    except Exception as e:
        print(f"❌ Error loading model: {str(e)}")
        raise
