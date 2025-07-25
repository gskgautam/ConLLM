import os
import numpy as np
import torch
import torchaudio
from tqdm import tqdm
from transformers import Wav2Vec2Processor, Wav2Vec2Model
from datasets.asvspoof2019 import ASVSpoof2019LADataset
from datasets.decro import DECRODataset

AUDIO_DATASETS = {
    'asvspoof2019': ASVSpoof2019LADataset(os.path.join('..', 'ASVSpoof 2019 (LA)')),
    'decro': DECRODataset(os.path.join('..', 'DECRO (D-E and D-C)')),
}

SAVE_DIR = 'embeddings/audio/'
os.makedirs(SAVE_DIR, exist_ok=True)

# Load XLS-R model and processor
processor = Wav2Vec2Processor.from_pretrained('facebook/wav2vec2-xls-r-1b')
model = Wav2Vec2Model.from_pretrained('facebook/wav2vec2-xls-r-1b')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
model.eval()

def load_audio(file_path, target_sr=16000):
    wav, sr = torchaudio.load(file_path)
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
    return wav.squeeze(0).numpy()

def extract_and_save_embeddings(dataset_name, dataset_loader):
    print(f"[INFO] Extracting embeddings for {dataset_name}")
    if dataset_name == 'asvspoof2019':
        splits = ['train', 'dev', 'eval']
        for split in splits:
            entries = dataset_loader.get_split(split)
            for entry in tqdm(entries, desc=f"{dataset_name}-{split}"):
                file_path = entry['file_path']
                utt_id = entry['utt_id']
                try:
                    audio = load_audio(file_path)
                    inputs = processor(audio, sampling_rate=16000, return_tensors="pt", padding=True)
                    with torch.no_grad():
                        input_values = inputs.input_values.to(device)
                        attention_mask = inputs.attention_mask.to(device)
                        outputs = model(input_values, attention_mask=attention_mask)
                        emb = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
                    save_path = os.path.join(SAVE_DIR, f"{dataset_name}_{split}_{utt_id}.npy")
                    np.save(save_path, emb)
                except Exception as e:
                    print(f"[ERROR] {file_path}: {e}")
    elif dataset_name == 'decro':
        for language in ['English', 'Chinese']:
            for split in ['real', 'fake']:
                entries = dataset_loader.get_split(language, split)
                for entry in tqdm(entries, desc=f"{dataset_name}-{language}-{split}"):
                    file_path = entry['file_path']
                    fname = os.path.splitext(os.path.basename(file_path))[0]
                    try:
                        audio = load_audio(file_path)
                        inputs = processor(audio, sampling_rate=16000, return_tensors="pt", padding=True)
                        with torch.no_grad():
                            input_values = inputs.input_values.to(device)
                            attention_mask = inputs.attention_mask.to(device)
                            outputs = model(input_values, attention_mask=attention_mask)
                            emb = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
                        save_path = os.path.join(SAVE_DIR, f"{dataset_name}_{language}_{split}_{fname}.npy")
                        np.save(save_path, emb)
                    except Exception as e:
                        print(f"[ERROR] {file_path}: {e}")

if __name__ == '__main__':
    for name, loader in AUDIO_DATASETS.items():
        extract_and_save_embeddings(name, loader) 