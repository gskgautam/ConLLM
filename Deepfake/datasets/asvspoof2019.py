import os
from typing import List, Dict

class ASVSpoof2019LADataset:
    def __init__(self, root_dir: str):
        self.root_dir = root_dir
        self.protocol_dir = os.path.join(root_dir, 'ASVspoof2019_LA_protocols')
        self.splits = {
            'train': os.path.join(root_dir, 'ASVspoof2019_LA_train'),
            'dev': os.path.join(root_dir, 'ASVspoof2019_LA_dev'),
            'eval': os.path.join(root_dir, 'ASVspoof2019_LA_eval'),
        }
        self.protocol_files = {
            'train': os.path.join(self.protocol_dir, 'ASVspoof2019.LA.cm.train.trl.txt'),
            'dev': os.path.join(self.protocol_dir, 'ASVspoof2019.LA.cm.dev.trl.txt'),
            'eval': os.path.join(self.protocol_dir, 'ASVspoof2019.LA.cm.eval.trl.txt'),
        }
        self.error_log = os.path.join(self.root_dir, 'asvspoof2019_errors.log')

    def log_error(self, msg):
        with open(self.error_log, 'a') as f:
            f.write(msg + '\n')

    def parse_protocol(self, split: str) -> List[Dict]:
        protocol_path = self.protocol_files[split]
        data_dir = self.splits[split]
        entries = []
        try:
            with open(protocol_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 4:
                        utt_id, speaker_id, label, system_id = parts
                    else:
                        continue
                    file_path = os.path.join(data_dir, utt_id + '.flac')
                    entries.append({
                        'utt_id': utt_id,
                        'speaker_id': speaker_id,
                        'label': label,  # 'bonafide' or 'spoofed'
                        'system_id': system_id,
                        'file_path': file_path
                    })
        except Exception as e:
            self.log_error(f"[ERROR] Reading protocol {protocol_path}: {e}")
        return entries

    def get_split(self, split: str) -> List[Dict]:
        assert split in self.splits
        return self.parse_protocol(split)

if __name__ == '__main__':
    # Example usage/demo
    root = os.path.join('..', 'ASVSpoof 2019 (LA)')
    dataset = ASVSpoof2019LADataset(root)
    train = dataset.get_split('train')
    print(f"Loaded {len(train)} train samples. Example:")
    print(train[0] if train else 'No data found.') 