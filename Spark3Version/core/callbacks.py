"""
Teaching Callback for educational fine-tuning.
"""

try:
    from transformers import TrainerCallback
except ImportError:
    class TrainerCallback:
        pass


class TeachingCallback(TrainerCallback):
    """A callback that provides educational explanations during training."""

    def on_log(self, args, state, control, logs=None, **kwargs):
        """Explains the loss value when logs come in."""
        if logs is None:
            return

        if 'loss' in logs:
            loss_value = logs['loss']
            print(f"📚 Loss: {loss_value:.4f} → Model is learning patterns. Lower loss = better performance!")
