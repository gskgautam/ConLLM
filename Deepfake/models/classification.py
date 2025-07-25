import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, roc_auc_score
from tqdm import tqdm

class RefinedEmbeddingDataset(Dataset):
    """
    Loads refined embeddings and their labels from .npy files.
    Args:
        embedding_dir (str): Directory containing .npy refined embedding files
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

class ClassifierHead(nn.Module):
    """
    Simple 1-layer MLP with Sigmoid for binary classification.
    """
    def __init__(self, input_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.linear(x)
        x = self.sigmoid(x)
        return x.squeeze(-1)

def build_label_map(embedding_dir):
    label_map = {}
    for fname in os.listdir(embedding_dir):
        if not fname.endswith('.npy'):
            continue
        if any(x in fname.lower() for x in ['bonafide', 'real']):
            label_map[fname] = 0
        elif any(x in fname.lower() for x in ['spoofed', 'fake']):
            label_map[fname] = 1
        else:
            label_map[fname] = 0
    return label_map

def train_classifier(embedding_dir, epochs=10, batch_size=64, lr=1e-3, device='cuda'):
    label_map = build_label_map(embedding_dir)
    dataset = RefinedEmbeddingDataset(embedding_dir, label_map)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    # Infer input_dim from a sample embedding
    sample_file = next((f for f in os.listdir(embedding_dir) if f.endswith('.npy')), None)
    if sample_file is None:
        print('No embeddings found.')
        return
    sample_emb = np.load(os.path.join(embedding_dir, sample_file))
    input_dim = sample_emb.shape[-1]
    model = ClassifierHead(input_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for emb, label in tqdm(dataloader, desc=f'Epoch {epoch+1}/{epochs}'):
            emb = emb.to(device)
            label = label.to(device)
            pred = model(emb)
            loss = criterion(pred, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss/len(dataloader):.4f}")
    # Evaluation
    model.eval()
    all_labels, all_preds = [], []
    with torch.no_grad():
        for emb, label in dataloader:
            emb = emb.to(device)
            pred = model(emb).cpu().numpy()
            all_preds.extend(pred)
            all_labels.extend(label.numpy())
    acc = accuracy_score(all_labels, np.round(all_preds))
    auc = roc_auc_score(all_labels, all_preds)
    print(f"[RESULT] Accuracy: {acc:.4f}, AUC: {auc:.4f}")
    # Save model
    save_path = os.path.join(embedding_dir, 'classifier_head.pt')
    torch.save(model.state_dict(), save_path)
    print(f"[INFO] Saved classifier head to {save_path}")

if __name__ == '__main__':
    embedding_dir = '../embeddings/audio_refined/'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    train_classifier(embedding_dir, epochs=10, batch_size=64, lr=1e-3, device=device) 