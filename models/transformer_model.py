"""
Transformer model for vehicle scheduling.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_models import BaseModel, device

class TransformerModel(BaseModel):
    """Transformer-based model for vehicle scheduling"""

    def __init__(self, input_dim=6, state_dim=6, d_model=128, nhead=4, num_layers=2):
        """
        Initialize the transformer model.
        
        Args:
            input_dim: Dimension of vehicle features
            state_dim: Dimension of global state
            d_model: Model dimension
            nhead: Number of attention heads
            num_layers: Number of transformer layers
        """
        super(TransformerModel, self).__init__()

        # Embeddings
        self.vehicle_embedding = nn.Linear(input_dim, d_model)
        self.state_embedding = nn.Linear(state_dim, d_model)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output head
        self.selection_head = nn.Linear(d_model, 1)

    def forward(self, vehicles, global_state, mask=None):
        """
        Forward pass through the transformer model.
        
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

        # Pass through transformer
        transformer_out = self.transformer(combined.unsqueeze(0))
        transformer_out = transformer_out.squeeze(0)

        # Get selection logits
        logits = self.selection_head(transformer_out).squeeze(-1)

        # Apply mask if provided
        if mask is not None:
            logits = logits.masked_fill(mask == 0, -1e9)

        # Return selection probabilities
        return F.softmax(logits, dim=0)
