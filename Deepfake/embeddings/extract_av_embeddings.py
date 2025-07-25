import os
import numpy as np
import torch
from tqdm import tqdm
from datasets.fakeavceleb import FakeAVCelebDataset
from datasets.dfdc import DFDCdataset
import torchaudio
import decord
from decord import VideoReader, cpu

AV_DATASETS = {
    'fakeavceleb': FakeAVCelebDataset(os.path.join('..', 'FakeAVCeleb (FAFC)')),
    'dfdc': DFDCdataset(os.path.join('..', 'DFDC (DeepFake Detection Challenge)')),
}

SAVE_DIR = 'embeddings/av/'
os.makedirs(SAVE_DIR, exist_ok=True)

def load_audio(file_path, target_sr=16000):
    wav, sr = torchaudio.load(file_path)
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
    return wav.squeeze(0).numpy()

def load_video_frames(file_path, num_frames=16):
    try:
        vr = VideoReader(file_path, ctx=cpu(0))
        total_frames = len(vr)
        if total_frames < num_frames:
            indices = np.linspace(0, total_frames - 1, total_frames).astype(int)
        else:
            indices = np.linspace(0, total_frames - 1, num_frames).astype(int)
        frames = vr.get_batch(indices).asnumpy()  # (num_frames, H, W, 3)
        return frames
    except Exception as e:
        print(f"[ERROR] Loading video {file_path}: {e}")
        return None

# Placeholder for VATLM model integration
class VATLMModel(torch.nn.Module):
    def __init__(self, audio_dim=16000, video_shape=(16, 224, 224, 3), output_dim=768):
        super().__init__()
        # TODO: Replace with real VATLM model loading
        self.output_dim = output_dim
    def forward(self, audio, video):
        # TODO: Replace with real VATLM inference
        # audio: (batch, audio_dim), video: (batch, num_frames, 224, 224, 3)
        # For now, concatenate mean of audio and mean of video as dummy embedding
        audio_feat = audio.mean(dim=1, keepdim=True)
        video_feat = video.mean(dim=(1,2,3,4), keepdim=True)
        return torch.cat([audio_feat, video_feat], dim=1)

def extract_and_save_embeddings(dataset_name, dataset_loader):
    print(f"[INFO] Extracting AV embeddings for {dataset_name}")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = VATLMModel().to(device)
    model.eval()
    entries = dataset_loader.get_all()
    for entry in tqdm(entries, desc=f"{dataset_name}"):
        file_path = entry['file_path']
        fname = os.path.splitext(os.path.basename(file_path))[0]
        try:
            frames = load_video_frames(file_path)
            if frames is None:
                continue
            # Extract audio from video file
            audio, _ = torchaudio.load(file_path)
            audio = audio.squeeze(0)
            # Preprocess video frames to (num_frames, 224, 224, 3)
            if frames.shape[1:3] != (224, 224):
                # Resize frames if needed
                import cv2
                frames = np.stack([cv2.resize(f, (224, 224)) for f in frames])
            video_tensor = torch.tensor(frames).float().unsqueeze(0).to(device)  # (1, num_frames, 224, 224, 3)
            audio_tensor = audio.unsqueeze(0).to(device)  # (1, audio_dim)
            with torch.no_grad():
                emb = model(audio_tensor, video_tensor).cpu().numpy().squeeze(0)
            save_path = os.path.join(SAVE_DIR, f"{dataset_name}_{fname}.npy")
            np.save(save_path, emb)
        except Exception as e:
            print(f"[ERROR] {file_path}: {e}")

if __name__ == '__main__':
    for name, loader in AV_DATASETS.items():
        extract_and_save_embeddings(name, loader) 