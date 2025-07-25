import os
import numpy as np
import torch
import torch.nn as nn
from models.classification import train_classifier, evaluate_classifier
from embeddings.projection_head import ProjectionHead
from models.llm_refinement import LLMRefiner
from tqdm import tqdm

# 1. Replace PTMs with CNNs (dummy CNN for demo)
class DummyCNN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.conv = nn.Conv1d(1, 8, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(8, output_dim)
    def forward(self, x):
        # x: (batch, input_dim)
        x = x.unsqueeze(1)  # (batch, 1, input_dim)
        x = self.conv(x)
        x = self.pool(x).squeeze(-1)
        x = self.fc(x)
        return x

def ablation_no_ptm(embedding_dir, output_dim=256, device='cuda'):
    print("[Ablation] Replacing PTM with CNN...")
    # Use DummyCNN instead of PTM/projection head
    files = [f for f in os.listdir(embedding_dir) if f.endswith('.npy')]
    sample_emb = np.load(os.path.join(embedding_dir, files[0]))
    input_dim = sample_emb.shape[-1]
    model = DummyCNN(input_dim, output_dim).to(device)
    model.eval()
    save_dir = embedding_dir + '_cnn/'
    os.makedirs(save_dir, exist_ok=True)
    for f in tqdm(files, desc='CNN Embedding'):
        emb = np.load(os.path.join(embedding_dir, f))
        emb_tensor = torch.tensor(emb, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            out = model(emb_tensor).cpu().numpy().squeeze(0)
        np.save(os.path.join(save_dir, f), out)
    train_classifier(save_dir, device=device)
    evaluate_classifier(save_dir, os.path.join(save_dir, 'classifier_head.pt'), device=device)

def ablation_no_contrastive(embedding_dir, device='cuda'):
    print("[Ablation] Removing contrastive learning...")
    # Skip projection head, use raw embeddings
    train_classifier(embedding_dir, device=device)
    evaluate_classifier(embedding_dir, os.path.join(embedding_dir, 'classifier_head.pt'), device=device)

def ablation_no_llm(embedding_dir, output_dim=256, device='cuda'):
    print("[Ablation] Removing LLM refinement...")
    # Use projection head output directly
    files = [f for f in os.listdir(embedding_dir) if f.endswith('.npy')]
    sample_emb = np.load(os.path.join(embedding_dir, files[0]))
    input_dim = sample_emb.shape[-1]
    model = ProjectionHead(input_dim, output_dim, activation='relu').to(device)
    model.eval()
    save_dir = embedding_dir + '_proj/'
    os.makedirs(save_dir, exist_ok=True)
    for f in tqdm(files, desc='Projection Embedding'):
        emb = np.load(os.path.join(embedding_dir, f))
        emb_tensor = torch.tensor(emb, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            out = model(emb_tensor).cpu().numpy().squeeze(0)
        np.save(os.path.join(save_dir, f), out)
    train_classifier(save_dir, device=device)
    evaluate_classifier(save_dir, os.path.join(save_dir, 'classifier_head.pt'), device=device)

def ablation_non_concat_fusion(embedding_dir, device='cuda'):
    print("[Ablation] Trying non-concatenation fusion (mean)...")
    # For demo: fuse multiple embeddings by mean instead of concat
    files = [f for f in os.listdir(embedding_dir) if f.endswith('.npy')]
    save_dir = embedding_dir + '_meanfuse/'
    os.makedirs(save_dir, exist_ok=True)
    for f in tqdm(files, desc='Mean Fusion'):
        emb = np.load(os.path.join(embedding_dir, f))
        if emb.ndim > 1:
            emb = emb.mean(axis=0)
        np.save(os.path.join(save_dir, f), emb)
    train_classifier(save_dir, device=device)
    evaluate_classifier(save_dir, os.path.join(save_dir, 'classifier_head.pt'), device=device)

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    embedding_dir = '../embeddings/audio_refined/'
    ablation_no_ptm(embedding_dir, device=device)
    ablation_no_contrastive(embedding_dir, device=device)
    ablation_no_llm(embedding_dir, device=device)
    ablation_non_concat_fusion(embedding_dir, device=device) 