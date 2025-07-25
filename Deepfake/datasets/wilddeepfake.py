import os
from typing import List, Dict

class WildDeepfakeDataset:
    def __init__(self, root_dir: str):
        self.root_dir = root_dir
        self.splits = {
            'real': os.path.join(root_dir, 'real'),
            'fake': os.path.join(root_dir, 'fake'),
        }
        self.source_type = 'Internet scraped (unconstrained)'

    def get_split(self, split: str) -> List[Dict]:
        assert split in self.splits, f"Unknown split: {split}"
        data_dir = self.splits[split]
        entries = []
        if not os.path.exists(data_dir):
            return entries
        for fname in os.listdir(data_dir):
            if fname.endswith('.mp4'):
                entries.append({
                    'file_path': os.path.join(data_dir, fname),
                    'label': split,  # 'real' or 'fake'
                    'source_type': self.source_type
                })
        return entries

    def get_all(self) -> List[Dict]:
        all_entries = []
        for split in self.splits:
            all_entries.extend(self.get_split(split))
        return all_entries 