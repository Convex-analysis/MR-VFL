"""
Transformer-based scheduler for MR-VFL.
"""

import torch
from .ml_scheduler import MLBasedScheduler
from models.transformer_model import TransformerModel

# Set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class TransformerScheduler(MLBasedScheduler):
    """Transformer-based scheduler for vehicle selection"""
    
    def __init__(self, input_dim=6, state_dim=6, d_model=128, nhead=4, num_layers=2):
        """
        Initialize the transformer scheduler.
        
        Args:
            input_dim: Dimension of vehicle features
            state_dim: Dimension of global state
            d_model: Model dimension
            nhead: Number of attention heads
            num_layers: Number of transformer layers
        """
        # Create transformer model
        model = TransformerModel(input_dim, state_dim, d_model, nhead, num_layers).to(device)
        super().__init__(model, "Transformer")
