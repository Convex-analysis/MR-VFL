"""
Mamba-based scheduler for MR-VFL.
"""

import os
import torch
from .ml_scheduler import MLBasedScheduler
from models.mamba_model import MambaModel

class MambaScheduler(MLBasedScheduler):
    def __init__(self, model_path, input_dim=6, state_dim=6, d_model=128, n_layers=2):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        # Create Mamba model
        model = MambaModel(
            input_dim=input_dim,
            state_dim=state_dim,
            d_model=d_model,
            n_layers=n_layers
        ).to(self.device)

        if model_path and os.path.exists(model_path):
            try:
                checkpoint = torch.load(model_path, map_location=self.device)
                model.load_state_dict(checkpoint['model_state_dict'])  # Changed from actor_state_dict
                print(f"Loaded Mamba model from {model_path}")
            except Exception as e:
                print(f"Error loading Mamba model: {e}")
                print("Using untrained Mamba model")

        super().__init__(model, "Mamba")
