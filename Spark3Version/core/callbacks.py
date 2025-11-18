"""
Teaching Callback for educational fine-tuning.

This module provides a custom callback that explains what's happening
during the training process to help users learn about ML training.
"""

try:
    from transformers import TrainerCallback
except ImportError:
    # Fallback if transformers is not installed
    class TrainerCallback:
        """Dummy TrainerCallback base class."""
        pass


class TeachingCallback(TrainerCallback):
    """
    A callback that provides educational explanations during training.

    This callback helps users understand what's happening at each step
    of the training process by printing informative messages.
    """

    def on_step_begin(self, args, state, control, **kwargs):
        """
        Called at the beginning of each training step.

        Prints educational messages every 10 steps to explain
        what's happening during training.

        Args:
            args: Training arguments
            state: Trainer state containing step information
            control: Trainer control object
            **kwargs: Additional arguments
        """
        if state.global_step % 10 == 0:
            print(f"📚 Step {state.global_step}: Adjusting weights based on loss...")
            print(f"   The model is performing forward pass, calculating loss, ")
            print(f"   and updating parameters through backpropagation.")

    def on_log(self, args, state, control, logs=None, **kwargs):
        """
        Called when the trainer logs metrics.

        Explains the loss value and what it means for model learning.

        Args:
            args: Training arguments
            state: Trainer state
            control: Trainer control object
            logs: Dictionary containing logged metrics
            **kwargs: Additional arguments
        """
        if logs is None:
            return

        if 'loss' in logs:
            loss_value = logs['loss']
            print(f"\n💡 Training Update (Step {state.global_step}):")
            print(f"   Loss: {loss_value:.4f}")

            # Provide context about what the loss means
            if hasattr(state, 'best_metric') and state.best_metric is not None:
                if loss_value < state.best_metric:
                    print(f"   ✓ Loss is decreasing -> Model is learning patterns!")
                else:
                    print(f"   ⚠ Loss increased slightly -> This is normal, training fluctuates")
            else:
                print(f"   → The model is calculating how far predictions are from targets")
                print(f"   → Lower loss = better performance")

        if 'learning_rate' in logs:
            lr = logs['learning_rate']
            print(f"   Learning Rate: {lr:.2e}")
            print(f"   → Controls how much we adjust weights each step")
