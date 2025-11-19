import torch
from pathlib import Path
from typing import List, Dict
from unsloth import FastLanguageModel
from utils.paths import EXPERIMENTS_DIR

# Hardcoded Test Prompts
TEST_PROMPTS = [
    "Write a python function to add two numbers.",
    "Explain what a class is in Python.",
    "What is a list comprehension?"
]

# Default Base Model
DEFAULT_BASE_MODEL = "unsloth/Qwen2.5-0.5B-bnb-4bit"


def run_test(experiment_name: str, base_model: str = DEFAULT_BASE_MODEL) -> None:
    """
    Inference Testing Engine: Vergleicht Base Model vs. Fine-Tuned Model.

    Args:
        experiment_name: Name des Experiments (z.B. "exp-001")
        base_model: HuggingFace Model Name (default: Qwen2.5-0.5B-bnb-4bit)

    Output:
        Speichert Vergleich in experiments/{experiment_name}/comparison.md
    """
    experiment_dir = EXPERIMENTS_DIR / experiment_name
    output_path = experiment_dir / "comparison.md"

    if not experiment_dir.exists():
        raise FileNotFoundError(f"Experiment directory not found: {experiment_dir}")

    print("\n" + "="*70)
    print(f"🧪 INFERENCE TEST: {experiment_name}")
    print("="*70)

    # === PHASE 1: Base Model (Reference) ===
    print("\n🤖 PHASE 1: BEFORE TRAINING (Base Model)")
    print("-" * 70)

    base_responses = _test_base_model(base_model, TEST_PROMPTS)

    # VRAM Cleanup
    torch.cuda.empty_cache()

    # === PHASE 2: Fine-Tuned Model ===
    print("\n🚀 PHASE 2: AFTER TRAINING (Fine-Tuned Model)")
    print("-" * 70)

    finetuned_responses = _test_finetuned_model(base_model, experiment_dir, TEST_PROMPTS)

    # === PHASE 3: Save Comparison ===
    print("\n📝 PHASE 3: Saving Comparison")
    print("-" * 70)

    _save_comparison(output_path, experiment_name, TEST_PROMPTS, base_responses, finetuned_responses)

    print(f"\n✅ Test completed!")
    print(f"📄 Results saved to: {output_path}")


def _test_base_model(model_name: str, prompts: List[str]) -> List[str]:
    """Testet das Base Model und gibt Antworten zurück."""
    print(f"📥 Loading base model: {model_name}")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=True,
    )

    FastLanguageModel.for_inference(model)

    responses = []
    for i, prompt in enumerate(prompts, 1):
        print(f"  [{i}/{len(prompts)}] Generating response...")
        response = _generate_response(model, tokenizer, prompt)
        responses.append(response)

    # Cleanup
    print("🧹 Cleaning up base model...")
    del model, tokenizer
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    return responses


def _test_finetuned_model(base_model: str, experiment_dir: Path, prompts: List[str]) -> List[str]:
    """Testet das Fine-Tuned Model (Base + Adapter) und gibt Antworten zurück."""
    # Finde Adapter Directory
    adapter_dir = experiment_dir / "checkpoints"
    if not adapter_dir.exists():
        adapter_dir = experiment_dir / "output"
    if not adapter_dir.exists():
        raise FileNotFoundError(f"Adapter directory not found in {experiment_dir}")

    print(f"📥 Loading fine-tuned model from: {adapter_dir}")

    # Lade Base Model
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base_model,
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=True,
    )

    # Lade Adapter (PEFT)
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, str(adapter_dir))

    FastLanguageModel.for_inference(model)

    responses = []
    for i, prompt in enumerate(prompts, 1):
        print(f"  [{i}/{len(prompts)}] Generating response...")
        response = _generate_response(model, tokenizer, prompt)
        responses.append(response)

    # Cleanup
    print("🧹 Cleaning up fine-tuned model...")
    del model, tokenizer
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    return responses


def _generate_response(model, tokenizer, prompt: str, max_new_tokens: int = 256) -> str:
    """Generiert Antwort für einen Prompt."""
    # Format als Chat Message
    messages = [{"role": "user", "content": prompt}]

    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to("cuda")

    # Generate
    outputs = model.generate(
        input_ids=inputs,
        max_new_tokens=max_new_tokens,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
        use_cache=True,
        pad_token_id=tokenizer.eos_token_id
    )

    # Dekodiere nur neue Tokens (ohne Input)
    response = tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
    return response.strip()


def _save_comparison(output_path: Path, experiment_name: str, prompts: List[str],
                     base_responses: List[str], finetuned_responses: List[str]) -> None:
    """Speichert Vergleich in Markdown-Format."""
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(f"# Model Comparison: {experiment_name}\n\n")
        f.write(f"**Device:** {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}\n\n")
        f.write("---\n\n")

        for i, (prompt, base_resp, ft_resp) in enumerate(zip(prompts, base_responses, finetuned_responses), 1):
            f.write(f"## Test {i}\n\n")
            f.write(f"### 📋 Prompt\n```\n{prompt}\n```\n\n")

            f.write(f"### 🤖 BEFORE TRAINING\n")
            f.write(f"```\n{base_resp}\n```\n\n")

            f.write(f"### 🚀 AFTER TRAINING\n")
            f.write(f"```\n{ft_resp}\n```\n\n")

            f.write("---\n\n")

    print(f"✅ Comparison saved to: {output_path}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        exp_name = sys.argv[1]
        model = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_BASE_MODEL
        run_test(exp_name, base_model=model)
    else:
        print("Usage: python tester.py <experiment_name> [base_model]")
        print(f"Default base model: {DEFAULT_BASE_MODEL}")
