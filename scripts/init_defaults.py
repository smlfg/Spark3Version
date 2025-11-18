#!/usr/bin/env python3
"""Initialize default configuration files for Spark3Version.

This script creates default model configuration files.
"""

import sys
from pathlib import Path

# Add parent directory to path to import Spark3Version modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from Spark3Version.utils.paths import ensure_dir, get_model_dir


def create_qwen_05b_config():
    """Create default configuration for Qwen-0.5B model."""
    model_dir = get_model_dir("qwen-0.5b")
    ensure_dir(model_dir)

    config_path = model_dir / "config.yaml"

    config_content = """name: qwen-0.5b
hf_path: unsloth/Qwen1.5-0.5B-bnb-4bit
model_type: causal
chat_template: chatml
load_in_4bit: true
"""

    config_path.write_text(config_content)
    print(f"✓ Created {config_path}")


def main():
    """Initialize all default configurations."""
    print("Initializing default configurations...")
    print()

    # Create model configurations
    create_qwen_05b_config()

    print()
    print("✓ Default configurations initialized successfully!")


if __name__ == "__main__":
    main()
