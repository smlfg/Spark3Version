import torch
from pathlib import Path
from typing import List, Dict
from unsloth import FastLanguageModel
from utils.paths import EXPERIMENTS_DIR

# Prompt Batches für verschiedene Test-Szenarien
PROMPT_BATCHES = {
    "coding-basic": [
        "Write a Python function to calculate the sum of two numbers.",
        "Create a Python function that returns the factorial of a number.",
        "Write a Python function to check if a string is a palindrome."
    ],
    "coding-advanced": [
        "Implement a binary search algorithm in Python.",
        "Write a Python class for a doubly linked list with insert and delete methods.",
        "Create a Python function to find the longest common subsequence of two strings."
    ],
    "debugging": [
        "Fix this code: def add(a b): return a + b",
        "Debug this function: def divide(x, y): return x / y",
        "Correct this loop: for i in range(10) print(i)"
    ]
}


class ModelTester:
    """Test-Framework für Fine-Tuned Models."""

    def __init__(self, experiment_name: str):
        self.experiment_name = experiment_name
        self.experiment_dir = EXPERIMENTS_DIR / experiment_name
        self.results_path = self.experiment_dir / "results.md"

        if not self.experiment_dir.exists():
            raise FileNotFoundError(f"Experiment directory not found: {self.experiment_dir}")

    def _load_base_model(self, model_name: str, max_seq_length: int = 2048):
        """Lädt das Base Model mit Unsloth."""
        print(f"📥 Loading base model: {model_name}")

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=max_seq_length,
            dtype=None,  # Auto-detect
            load_in_4bit=True,
        )

        FastLanguageModel.for_inference(model)
        return model, tokenizer

    def _load_finetuned_model(self, model_name: str, adapter_dir: Path, max_seq_length: int = 2048):
        """Lädt das Fine-Tuned Model (Base + Adapter)."""
        print(f"📥 Loading fine-tuned model from: {adapter_dir}")

        # Lade Base Model mit Adapter
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=max_seq_length,
            dtype=None,
            load_in_4bit=True,
        )

        # Lade Adapter
        model = FastLanguageModel.get_peft_model(
            model,
            r=16,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj"],
            lora_alpha=16,
            lora_dropout=0,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=3407,
        )

        # Lade gespeicherte Adapter-Weights
        adapter_path = adapter_dir / "adapter_model.safetensors"
        if adapter_path.exists():
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, str(adapter_dir))

        FastLanguageModel.for_inference(model)
        return model, tokenizer

    def _generate_response(self, model, tokenizer, prompt: str, max_new_tokens: int = 256) -> str:
        """Generiert Antwort für einen Prompt."""
        # Format prompt für Chat-Template
        messages = [{"role": "user", "content": prompt}]
        inputs = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to("cuda")

        # Generiere Antwort
        outputs = model.generate(
            input_ids=inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id
        )

        # Dekodiere nur die generierte Antwort (ohne Input)
        response = tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
        return response.strip()

    def _cleanup_model(self, model):
        """Gibt VRAM frei."""
        print("🧹 Cleaning up VRAM...")
        del model
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    def _save_results(self, results: List[Dict]):
        """Speichert Test-Ergebnisse in Markdown."""
        with open(self.results_path, 'w', encoding='utf-8') as f:
            f.write(f"# Test Results: {self.experiment_name}\n\n")
            f.write(f"**Date:** {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}\n\n")
            f.write("---\n\n")

            for i, result in enumerate(results, 1):
                f.write(f"## Test {i}: {result['prompt'][:60]}...\n\n")
                f.write(f"### Prompt\n```\n{result['prompt']}\n```\n\n")
                f.write(f"### Base Model Response\n```\n{result['base_response']}\n```\n\n")
                f.write(f"### Fine-Tuned Model Response\n```\n{result['finetuned_response']}\n```\n\n")
                f.write("---\n\n")

        print(f"✅ Results saved to: {self.results_path}")

    def run_test(self,
                 model_name: str,
                 prompt_batch: str = "coding-basic",
                 max_prompts: int = 3) -> None:
        """
        Führt Test aus: Vergleicht Base Model vs. Fine-Tuned Model.

        Args:
            model_name: HuggingFace Model Name (z.B. "unsloth/Meta-Llama-3.1-8B-Instruct")
            prompt_batch: Name des Prompt-Batches
            max_prompts: Anzahl der Prompts zum Testen
        """
        if prompt_batch not in PROMPT_BATCHES:
            raise ValueError(f"Unknown prompt batch: {prompt_batch}. Available: {list(PROMPT_BATCHES.keys())}")

        prompts = PROMPT_BATCHES[prompt_batch][:max_prompts]
        results = []

        # === PHASE 1: Base Model Testing ===
        print("\n" + "="*60)
        print("PHASE 1: Testing Base Model")
        print("="*60)

        base_model, base_tokenizer = self._load_base_model(model_name)
        base_responses = []

        for i, prompt in enumerate(prompts, 1):
            print(f"\n[{i}/{len(prompts)}] Generating base model response...")
            response = self._generate_response(base_model, base_tokenizer, prompt)
            base_responses.append(response)
            print(f"✓ Done")

        # Cleanup nach Base Model
        self._cleanup_model(base_model)
        del base_tokenizer

        # === PHASE 2: Fine-Tuned Model Testing ===
        print("\n" + "="*60)
        print("PHASE 2: Testing Fine-Tuned Model")
        print("="*60)

        adapter_dir = self.experiment_dir / "checkpoints"
        if not adapter_dir.exists():
            # Fallback: Suche nach finalem Output
            adapter_dir = self.experiment_dir / "output"

        if not adapter_dir.exists():
            raise FileNotFoundError(f"Adapter directory not found: {adapter_dir}")

        ft_model, ft_tokenizer = self._load_finetuned_model(model_name, adapter_dir)
        ft_responses = []

        for i, prompt in enumerate(prompts, 1):
            print(f"\n[{i}/{len(prompts)}] Generating fine-tuned model response...")
            response = self._generate_response(ft_model, ft_tokenizer, prompt)
            ft_responses.append(response)
            print(f"✓ Done")

        # Cleanup nach Fine-Tuned Model
        self._cleanup_model(ft_model)
        del ft_tokenizer

        # === PHASE 3: Ergebnisse zusammenstellen ===
        print("\n" + "="*60)
        print("PHASE 3: Saving Results")
        print("="*60)

        for prompt, base_resp, ft_resp in zip(prompts, base_responses, ft_responses):
            results.append({
                'prompt': prompt,
                'base_response': base_resp,
                'finetuned_response': ft_resp
            })

        self._save_results(results)

        print("\n✅ Test completed successfully!")
        print(f"📄 Results: {self.results_path}")


def run_test(experiment_name: str,
             model_name: str = "unsloth/Meta-Llama-3.1-8B-Instruct",
             prompt_batch: str = "coding-basic",
             max_prompts: int = 3) -> None:
    """
    Convenience-Funktion zum Testen eines Experiments.

    Args:
        experiment_name: Name des Experiments
        model_name: HuggingFace Model Name
        prompt_batch: Prompt-Batch Name ("coding-basic", "coding-advanced", "debugging")
        max_prompts: Anzahl der zu testenden Prompts

    Example:
        >>> run_test("exp-001", prompt_batch="coding-basic", max_prompts=3)
    """
    tester = ModelTester(experiment_name)
    tester.run_test(model_name, prompt_batch, max_prompts)


if __name__ == "__main__":
    # Test-Beispiel
    import sys
    if len(sys.argv) > 1:
        exp_name = sys.argv[1]
        batch = sys.argv[2] if len(sys.argv) > 2 else "coding-basic"
        run_test(exp_name, prompt_batch=batch)
    else:
        print("Usage: python tester.py <experiment_name> [prompt_batch]")
