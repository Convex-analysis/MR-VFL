"""
LSTM model for vehicle scheduling.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_models import BaseModel, device

class LSTMModel(BaseModel):
    """LSTM-based model for vehicle scheduling"""

    def __init__(self, input_dim=6, state_dim=6, hidden_dim=128, num_layers=2):
        """
        Initialize the LSTM model.
        
        Args:
            input_dim: Dimension of vehicle features
            state_dim: Dimension of global state
            hidden_dim: Hidden dimension of LSTM
            num_layers: Number of LSTM layers
        """
        super(LSTMModel, self).__init__()

        # Embeddings
        self.vehicle_embedding = nn.Linear(input_dim, hidden_dim)
        self.state_embedding = nn.Linear(state_dim, hidden_dim)

        # LSTM
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )

        # Output head
        self.selection_head = nn.Linear(hidden_dim, 1)

    def forward(self, vehicles, global_state, mask=None):
        """
        Forward pass through the LSTM model.
        
        Args:
            vehicles: List of vehicle tensors
            global_state: Global state tensor
            mask: Optional mask for vehicles
            
        Returns:
            Selection probabilities for each vehicle
        """
        # Stack vehicle tensors
        vehicle_tensors = torch.stack(vehicles)

        # Embed vehicles
        vehicle_embeds = self.vehicle_embedding(vehicle_tensors)

        # Embed global state and expand
        state_embed = self.state_embedding(global_state)
        state_expanded = state_embed.unsqueeze(0).expand(len(vehicles), -1)

        # Combine embeddings
        combined = vehicle_embeds + state_expanded

        # Pass through LSTM
        lstm_out, _ = self.lstm(combined.unsqueeze(0))
        lstm_out = lstm_out.squeeze(0)

        # Get selection logits
        logits = self.selection_head(lstm_out).squeeze(-1)

        # Apply mask if provided
        if mask is not None:
            logits = logits.masked_fill(mask == 0, -1e9)

        # Return selection probabilities
        return F.softmax(logits, dim=0)
