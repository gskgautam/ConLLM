import torch
import torch.nn as nn

class ProjectionHead(nn.Module):
    """
    Simple projection head to map modality-specific embeddings into a shared latent space.
    Args:
        input_dim (int): Dimension of input embeddings
        output_dim (int): Dimension of projected latent space
        activation (str): Optional activation ('relu', 'gelu', or None)
    """
    def __init__(self, input_dim, output_dim, activation=None):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'gelu':
            self.activation = nn.GELU()
        else:
            self.activation = None

    def forward(self, x):
        x = self.linear(x)
        if self.activation:
            x = self.activation(x)
        return x 