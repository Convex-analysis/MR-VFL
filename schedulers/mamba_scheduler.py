"""
Mamba-based scheduler for MR-VFL.
"""

import os
import torch
from .ml_scheduler import MLBasedScheduler
from ..models.mamba_model import MambaModel

# Set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class MambaScheduler(MLBasedScheduler):
    """Mamba-based scheduler for vehicle selection"""
    
    def __init__(self, model_path=None, input_dim=6, state_dim=6, d_model=128, n_layers=2):
        """
        Initialize the Mamba scheduler.
        
        Args:
            model_path: Path to the pre-trained model checkpoint
            input_dim: Dimension of vehicle features
            state_dim: Dimension of global state
            d_model: Model dimension
            n_layers: Number of Mamba layers
        """
        # Create Mamba model
        model = MambaModel(input_dim, state_dim, d_model, n_layers).to(device)
        
        # Load pre-trained model if available
        if model_path and os.path.exists(model_path):
            try:
                checkpoint = torch.load(model_path, map_location=device)
                model.load_state_dict(checkpoint['actor_state_dict'])
                print(f"Loaded Mamba model from {model_path}")
            except Exception as e:
                print(f"Error loading Mamba model: {e}")
                print("Using untrained Mamba model")
        
        super().__init__(model, "Mamba")
