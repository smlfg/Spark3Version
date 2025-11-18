import json
import random
from pathlib import Path
from typing import List

class DatasetSlicer:
    @staticmethod
    def slice_file(input_path: str, output_dir: str, chunk_size: int = 1000, seed: int = 42) -> List[str]:
        path = Path(input_path)
        if not path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")
        
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f if line.strip()]
            
        random.seed(seed)
        random.shuffle(lines)
        
        created_files = []
        total_chunks = (len(lines) + chunk_size - 1) // chunk_size
        
        for i in range(total_chunks):
            chunk = lines[i*chunk_size : (i+1)*chunk_size]
            # Parse JSONL line to dict
            data_chunk = [json.loads(line) for line in chunk]
            
            filename = f"slice_{i:04d}.json"
            file_path = out_path / filename
            
            with open(file_path, 'w', encoding='utf-8') as out_f:
                json.dump(data_chunk, out_f, indent=2)
                
            created_files.append(str(file_path))
            
        return created_files
