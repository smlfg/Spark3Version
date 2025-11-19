from transformers import TrainerCallback
import time

class TeachingCallback(TrainerCallback):
    def __init__(self):
        self.start_time = 0
        self.last_loss = 0

    def on_train_begin(self, args, state, control, **kwargs):
        self.start_time = time.time()
        print("\n" + "="*60)
        print("🎓 MFT CLASSROOM: Training Session Started")
        print("="*60)
        print("ℹ️  Objective: Fine-tune the model using LoRA (Low-Rank Adaptation).")
        print("ℹ️  Optimization: 4-bit quantization enabled for VRAM efficiency.")
        print("-" * 60 + "\n")

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and 'loss' in logs:
            loss = logs['loss']
            step = state.global_step
            print(f"📊 Step {step}: Loss = {loss:.4f}")
            
            if loss > 2.0:
                print("   🤔 Model Status: High error. Still guessing randomly.")
            elif loss < 1.0:
                print("   💡 Model Status: Patterns recognized! Learning is effective.")
            elif loss < 0.5:
                print("   🚀 Model Status: Mastering the dataset details.")