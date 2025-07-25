import os
from typing import List, Dict

class CelebDFDataset:
    def __init__(self, root_dir: str):
        self.root_dir = root_dir
        self.splits = {
            'real': os.path.join(root_dir, 'RealVideos'),
            'fake': os.path.join(root_dir, 'FakeVideos'),
        }
        self.error_log = os.path.join(self.root_dir, 'celebdf_errors.log')

    def log_error(self, msg):
        with open(self.error_log, 'a') as f:
            f.write(msg + '\n')

    def get_split(self, split: str) -> List[Dict]:
        assert split in self.splits, f"Unknown split: {split}"
        data_dir = self.splits[split]
        entries = []
        if not os.path.exists(data_dir):
            return entries
        try:
            files = os.listdir(data_dir)
        except Exception as e:
            self.log_error(f"[ERROR] Listing {data_dir}: {e}")
            return entries
        for fname in files:
            try:
                if fname.endswith('.mp4'):
                    parts = fname.split('_')
                    actor_id = parts[1].split('.')[0] if len(parts) > 1 else None
                    entries.append({
                        'file_path': os.path.join(data_dir, fname),
                        'label': split,  # 'real' or 'fake'
                        'actor_id': actor_id
                    })
            except Exception as e:
                self.log_error(f"[ERROR] File {fname} in {data_dir}: {e}")
        return entries

    def get_all(self) -> List[Dict]:
        all_entries = []
        for split in self.splits:
            all_entries.extend(self.get_split(split))
        return all_entries 