import os
from typing import List, Dict

class DECRODataset:
    def __init__(self, root_dir: str):
        self.root_dir = root_dir
        self.languages = ['English', 'Chinese']
        self.splits = ['real', 'fake']
        self.error_log = os.path.join(self.root_dir, 'decro_errors.log')

    def log_error(self, msg):
        with open(self.error_log, 'a') as f:
            f.write(msg + '\n')

    def get_split(self, language: str, split: str) -> List[Dict]:
        assert language in self.languages, f"Unknown language: {language}"
        assert split in self.splits, f"Unknown split: {split}"
        data_dir = os.path.join(self.root_dir, language, split)
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
                if fname.endswith('.wav'):
                    entries.append({
                        'file_path': os.path.join(data_dir, fname),
                        'label': split,  # 'real' or 'fake'
                        'language': language
                    })
            except Exception as e:
                self.log_error(f"[ERROR] File {fname} in {data_dir}: {e}")
        return entries

    def get_all(self) -> List[Dict]:
        all_entries = []
        for language in self.languages:
            for split in self.splits:
                all_entries.extend(self.get_split(language, split))
        return all_entries 