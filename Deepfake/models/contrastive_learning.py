import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
from embeddings.projection_head import ProjectionHead

class EmbeddingPairDataset(Dataset):
    """
    Loads embeddings and their labels from .npy files.
    Args:
        embedding_dir (str): Directory containing .npy embedding files
        label_map (dict): Mapping from file name to label (0=authentic, 1=fake)
    """
    def __init__(self, embedding_dir, label_map):
        self.embedding_dir = embedding_dir
        self.label_map = label_map
        self.files = [f for f in os.listdir(embedding_dir) if f.endswith('.npy')]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]
        emb = np.load(os.path.join(self.embedding_dir, fname))
        label = self.label_map.get(fname, 0)
        return torch.tensor(emb, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)

class NTXentLoss(nn.Module):
    """
    Normalized Temperature-scaled Cross Entropy Loss (NT-Xent) for contrastive learning.
    """
    def __init__(self, temperature=0.5):
        super().__init__()
        self.temperature = temperature

    def forward(self, z_i, z_j):
        z_i = F.normalize(z_i, dim=1)
        z_j = F.normalize(z_j, dim=1)
        representations = torch.cat([z_i, z_j], dim=0)
        similarity_matrix = torch.matmul(representations, representations.T)
        sim_ij = torch.diag(similarity_matrix, len(z_i))
        sim_ji = torch.diag(similarity_matrix, -len(z_i))
        positives = torch.cat([sim_ij, sim_ji], dim=0)
        nominator = torch.exp(positives / self.temperature)
        mask = ~torch.eye(len(representations), dtype=bool)
        denominator = mask.to(z_i.device) * torch.exp(similarity_matrix / self.temperature)
        denominator = denominator.sum(dim=1)
        loss = -torch.log(nominator / denominator)
        return loss.mean()

def build_label_map(embedding_dir):
    """
    Build a label map from embedding filenames.
    Assumes filenames contain 'bonafide', 'real', 'spoofed', or 'fake'.
    Returns: dict mapping filename to 0 (authentic) or 1 (fake)
    """
    label_map = {}
    for fname in os.listdir(embedding_dir):
        if not fname.endswith('.npy'):
            continue
        if any(x in fname.lower() for x in ['bonafide', 'real']):
            label_map[fname] = 0
        elif any(x in fname.lower() for x in ['spoofed', 'fake']):
            label_map[fname] = 1
        else:
            label_map[fname] = 0  # Default to authentic if unknown
    return label_map

def train_contrastive(embedding_dir, input_dim, output_dim=256, epochs=10, batch_size=64, lr=1e-3, device='cuda'):
    label_map = build_label_map(embedding_dir)
    dataset = EmbeddingPairDataset(embedding_dir, label_map)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model = ProjectionHead(input_dim, output_dim, activation='relu').to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = NTXentLoss()
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for batch in dataloader:
            emb, label = batch
            emb = emb.to(device)
            # For contrastive, split batch into two views (simulate augmentations)
            z_i = model(emb)
            z_j = model(emb[torch.randperm(emb.size(0))])  # Shuffle as negative
            loss = loss_fn(z_i, z_j)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss/len(dataloader):.4f}")
    # Save trained projection head
    save_path = os.path.join(embedding_dir, f'projection_head_{input_dim}to{output_dim}.pt')
    torch.save(model.state_dict(), save_path)
    print(f"[INFO] Saved projection head to {save_path}")

if __name__ == '__main__':
    # Example: train on audio embeddings
    embedding_dir = '../embeddings/audio/'
    # Infer input_dim from a sample embedding
    sample_file = next((f for f in os.listdir(embedding_dir) if f.endswith('.npy')), None)
    if sample_file is not None:
        sample_emb = np.load(os.path.join(embedding_dir, sample_file))
        input_dim = sample_emb.shape[-1]
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        train_contrastive(embedding_dir, input_dim, output_dim=256, epochs=10, batch_size=64, lr=1e-3, device=device)
    else:
        print("No embeddings found in directory.") 