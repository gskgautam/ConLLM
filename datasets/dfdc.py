import os
import json
from typing import List, Dict

class DFDCdataset:
    def __init__(self, root_dir: str):
        self.root_dir = root_dir

    def get_all(self) -> List[Dict]:
        entries = []
        for part in os.listdir(self.root_dir):
            part_dir = os.path.join(self.root_dir, part)
            if not os.path.isdir(part_dir) or not part.startswith('dfdc_train_part_'):
                continue
            metadata_path = os.path.join(part_dir, 'metadata.json')
            if not os.path.exists(metadata_path):
                continue
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            for fname, meta in metadata.items():
                file_path = os.path.join(part_dir, fname)
                label = meta.get('label', None)
                original = meta.get('original', None)
                entries.append({
                    'file_path': file_path,
                    'label': label,  # 'REAL' or 'FAKE'
                    'original': original
                })
        return entries 