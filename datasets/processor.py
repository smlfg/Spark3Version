"""
Dataset Slicer Module

This module provides functionality to slice large JSONL datasets into smaller chunks
for distributed processing and fine-tuning.
"""

import json
import random
from pathlib import Path
from typing import List


class DatasetSlicer:
    """
    High-performance dataset slicer for processing large JSONL files.

    This class provides static methods to split large datasets into manageable chunks
    with deterministic shuffling for reproducibility.
    """

    @staticmethod
    def slice_file(
        input_path: str,
        output_dir: str,
        chunk_size: int = 1000,
        seed: int = 42
    ) -> List[str]:
        """
        Slice a JSONL file into multiple JSON array chunks.

        Args:
            input_path: Path to the input JSONL file
            output_dir: Directory where chunk files will be saved
            chunk_size: Number of lines per chunk (default: 1000)
            seed: Random seed for deterministic shuffling (default: 42)

        Returns:
            List of created file paths

        Raises:
            FileNotFoundError: If input_path does not exist
            ValueError: If chunk_size is less than 1
        """
        # Validate input path
        input_file = Path(input_path)
        if not input_file.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")

        # Validate chunk size
        if chunk_size < 1:
            raise ValueError(f"chunk_size must be >= 1, got {chunk_size}")

        print(f"[DatasetSlicer] Loading data from: {input_path}")

        # Load all lines from JSONL file
        lines = []
        with open(input_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if line:  # Skip empty lines
                    try:
                        data = json.loads(line)
                        lines.append(data)
                    except json.JSONDecodeError as e:
                        print(f"[DatasetSlicer] WARNING: Skipping invalid JSON at line {line_num}: {e}")

        total_lines = len(lines)
        print(f"[DatasetSlicer] Loaded {total_lines} valid lines")

        # Deterministic shuffle
        print(f"[DatasetSlicer] Shuffling with seed={seed}")
        rng = random.Random(seed)
        rng.shuffle(lines)

        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        print(f"[DatasetSlicer] Output directory: {output_dir}")

        # Split into chunks and save
        created_files = []
        num_chunks = (total_lines + chunk_size - 1) // chunk_size  # Ceiling division

        for chunk_index in range(num_chunks):
            start_idx = chunk_index * chunk_size
            end_idx = min(start_idx + chunk_size, total_lines)
            chunk_data = lines[start_idx:end_idx]

            # Create filename with zero-padded index
            filename = f"slice_{chunk_index:04d}.json"
            file_path = output_path / filename

            # Write chunk as JSON array
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(chunk_data, f, ensure_ascii=False, indent=2)

            created_files.append(str(file_path))
            print(f"[DatasetSlicer] Created {filename} with {len(chunk_data)} entries")

        print(f"[DatasetSlicer] Slicing complete! Created {len(created_files)} chunks")
        return created_files
