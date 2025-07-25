import os
import numpy as np
import torch
from tqdm import tqdm
from datasets.celebdf import CelebDFDataset
from datasets.wilddeepfake import WildDeepfakeDataset
from torchvision import transforms
import decord
from decord import VideoReader
from decord import cpu
from transformers import VideoMAEImageProcessor, VideoMAEModel

VIDEO_DATASETS = {
    'celebdf': CelebDFDataset(os.path.join('..', 'Celeb-DF (CDF)')),
    'wilddeepfake': WildDeepfakeDataset(os.path.join('..', 'WildDeepfake (WD)')),
}

SAVE_DIR = 'embeddings/video/'
os.makedirs(SAVE_DIR, exist_ok=True)

# Load VideoMAE model and processor
processor = VideoMAEImageProcessor.from_pretrained('MCG-NJU/videomae-base')
model = VideoMAEModel.from_pretrained('MCG-NJU/videomae-base')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
model.eval()

# VideoMAE expects 16 frames of 224x224
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

def load_video_frames(file_path, num_frames=16):
    try:
        vr = VideoReader(file_path, ctx=cpu(0))
        total_frames = len(vr)
        if total_frames < num_frames:
            indices = np.linspace(0, total_frames - 1, total_frames).astype(int)
        else:
            indices = np.linspace(0, total_frames - 1, num_frames).astype(int)
        frames = vr.get_batch(indices).asnumpy()  # (num_frames, H, W, 3)
        # Apply transform to each frame
        frames = [transform(frame) for frame in frames]
        frames = torch.stack(frames)  # (num_frames, 3, 224, 224)
        return frames
    except Exception as e:
        print(f"[ERROR] Loading video {file_path}: {e}")
        return None

def extract_and_save_embeddings(dataset_name, dataset_loader):
    print(f"[INFO] Extracting embeddings for {dataset_name}")
    for split in ['real', 'fake']:
        entries = dataset_loader.get_split(split)
        for entry in tqdm(entries, desc=f"{dataset_name}-{split}"):
            file_path = entry['file_path']
            actor_id = entry.get('actor_id', 'unknown')
            fname = os.path.splitext(os.path.basename(file_path))[0]
            try:
                frames = load_video_frames(file_path)
                if frames is None or frames.shape[0] != 16:
                    continue
                # VideoMAE expects input as (batch, num_frames, 3, 224, 224)
                pixel_values = frames.unsqueeze(0).to(device)  # (1, 16, 3, 224, 224)
                # Rearrange to (batch, 3, num_frames, 224, 224)
                pixel_values = pixel_values.permute(0, 2, 1, 3, 4)
                with torch.no_grad():
                    outputs = model(pixel_values)
                    emb = outputs.last_hidden_state.mean(dim=1).cpu().numpy()  # (1, hidden_dim)
                save_path = os.path.join(SAVE_DIR, f"{dataset_name}_{split}_{actor_id}_{fname}.npy")
                np.save(save_path, emb.squeeze(0))
            except Exception as e:
                print(f"[ERROR] {file_path}: {e}")

if __name__ == '__main__':
    for name, loader in VIDEO_DATASETS.items():
        extract_and_save_embeddings(name, loader) 