import json
import random
from pathlib import Path
from typing import List


class DatasetSlicer:

    @staticmethod
    def slice_file(input_path: str, output_dir: str, chunk_size: int = 1000) -> None:
        lines = []
        with open(input_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    lines.append(json.loads(line))

        random.seed(42)
        random.shuffle(lines)

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        num_chunks = (len(lines) + chunk_size - 1) // chunk_size

        for i in range(num_chunks):
            start = i * chunk_size
            end = min(start + chunk_size, len(lines))
            chunk = lines[start:end]

            filename = output_path / f"slice_{i:04d}.json"
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(chunk, f, ensure_ascii=False, indent=2)
