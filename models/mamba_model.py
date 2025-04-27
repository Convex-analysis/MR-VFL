"""
Mamba model for vehicle scheduling.

This file serves as a wrapper for the MR-VFL Mamba model.
"""

import torch
from .base_models import BaseModel, device

# Import the actual Mamba model from the MR-VFL package
from mr_vfl_mamba_scheduler import MRVFLMambaActor

class MambaModel(BaseModel):
    def __init__(self, input_dim, state_dim, d_model, n_layers=None):
        super().__init__()
        # n_layers parameter is kept for backward compatibility but not used

        # Create the Mamba actor
        self.actor = MRVFLMambaActor(
            input_dim=input_dim,
            state_dim=state_dim,
            d_model=d_model,
            d_state=16  # Default value for d_state
        )

    def forward(self, vehicles, global_state, mask=None):
        """
        Forward pass through the Mamba model.

        Args:
            vehicles: List of vehicle tensors
            global_state: Global state tensor
            mask: Optional mask for vehicles

        Returns:
            Selection probabilities for each vehicle
        """
        # Use the Mamba actor's forward method but only return the selection probabilities
        selection_probs, _, _, _ = self.actor(vehicles, global_state, mask)
        return selection_probs

    def load_state_dict(self, state_dict):
        """
        Load state dictionary into the model.

        Args:
            state_dict: State dictionary to load
        """
        self.actor.load_state_dict(state_dict)

    def state_dict(self):
        """
        Get the model's state dictionary.

        Returns:
            State dictionary of the model
        """
        return self.actor.state_dict()
