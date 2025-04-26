"""
Base model classes for MR-VFL scheduler comparison.
"""

import torch
import torch.nn as nn

# Set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class BaseModel(nn.Module):
    """Base class for all models used in the comparison."""
    
    def __init__(self):
        super(BaseModel, self).__init__()
    
    def forward(self, vehicles, global_state, mask=None):
        """
        Forward pass through the model.
        
        Args:
            vehicles: List of vehicle tensors
            global_state: Global state tensor
            mask: Optional mask for vehicles
            
        Returns:
            Selection probabilities for each vehicle
        """
        raise NotImplementedError("Subclasses must implement forward method")
