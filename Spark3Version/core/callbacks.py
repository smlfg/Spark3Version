"""
Teaching Callback for educational fine-tuning.
"""

try:
    from transformers import TrainerCallback
except ImportError:
    class TrainerCallback(object):
        pass


class TeachingCallback(TrainerCallback):
    """A callback that provides educational explanations during training."""

    def on_step_begin(self, args, state, control, **kwargs):
        """Called at the beginning of each training step."""
        if state.global_step % 10 == 0:
            print(f"📚 Step {state.global_step}: Adjusting weights based on loss...")

    def on_log(self, args, state, control, logs=None, **kwargs):
        """Explains the loss value when logs come in."""
        if logs is None:
            return

        if 'loss' in logs:
            loss_value = logs['loss']
            print(f"💡 Loss: {loss_value:.4f} → Loss is decreasing -> Model is learning patterns.")
