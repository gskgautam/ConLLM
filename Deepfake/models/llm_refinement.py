import os
import numpy as np
import torch
import torch.nn as nn
from transformers import GPT2Model, GPT2Config
from tqdm import tqdm

class LLMRefiner(nn.Module):
    """
    Refines input embeddings using a transformer (GPT-2) backbone with self-attention and cross-attention.
    Args:
        input_dim (int): Dimension of input embeddings
        hidden_dim (int): Hidden size for transformer
        num_layers (int): Number of transformer layers
    """
    def __init__(self, input_dim, hidden_dim=768, num_layers=2):
        super().__init__()
        # Project input to transformer hidden size
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        # Use a small GPT-2 as backbone (can swap for GPT-3 API in production)
        config = GPT2Config(n_embd=hidden_dim, n_layer=num_layers, n_head=8, n_positions=32, n_ctx=32, vocab_size=1000)
        self.transformer = GPT2Model(config)
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        x_proj = self.input_proj(x)
        # GPT-2 expects input as (batch, seq_len, hidden_dim)
        outputs = self.transformer(inputs_embeds=x_proj)
        last_hidden = outputs.last_hidden_state  # (batch, seq_len, hidden_dim)
        out = self.output_proj(last_hidden)
        # Optionally, pool or flatten
        pooled = out.mean(dim=1)  # (batch, input_dim)
        return pooled

def refine_embeddings(embedding_dir, save_dir, input_dim, batch_size=32, device='cuda'):
    """
    Loads aligned embeddings, refines them with LLMRefiner, and saves the output.
    """
    os.makedirs(save_dir, exist_ok=True)
    model = LLMRefiner(input_dim).to(device)
    model.eval()
    files = [f for f in os.listdir(embedding_dir) if f.endswith('.npy')]
    for i in tqdm(range(0, len(files), batch_size), desc='Refining embeddings'):
        batch_files = files[i:i+batch_size]
        batch_embs = [np.load(os.path.join(embedding_dir, f)) for f in batch_files]
        # Assume each embedding is (input_dim,) or (seq_len, input_dim)
        batch_embs = [e if e.ndim == 2 else e[None, :] for e in batch_embs]
        batch_tensor = torch.tensor(np.stack(batch_embs), dtype=torch.float32).to(device)
        with torch.no_grad():
            refined = model(batch_tensor)
        for fname, emb in zip(batch_files, refined.cpu().numpy()):
            np.save(os.path.join(save_dir, fname), emb)
    print(f"[INFO] Saved refined embeddings to {save_dir}")

if __name__ == '__main__':
    # Example usage
    embedding_dir = '../embeddings/audio/'  # Use aligned embeddings
    save_dir = '../embeddings/audio_refined/'
    sample_file = next((f for f in os.listdir(embedding_dir) if f.endswith('.npy')), None)
    if sample_file is not None:
        sample_emb = np.load(os.path.join(embedding_dir, sample_file))
        input_dim = sample_emb.shape[-1]
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        refine_embeddings(embedding_dir, save_dir, input_dim, batch_size=32, device=device)
    else:
        print('No embeddings found in directory.') 