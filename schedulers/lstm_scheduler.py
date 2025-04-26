"""
LSTM-based scheduler for MR-VFL.
"""

import torch
from .ml_scheduler import MLBasedScheduler
from ..models.lstm_model import LSTMModel

# Set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class LSTMScheduler(MLBasedScheduler):
    """LSTM-based scheduler for vehicle selection"""
    
    def __init__(self, input_dim=6, state_dim=6, hidden_dim=128, num_layers=2):
        """
        Initialize the LSTM scheduler.
        
        Args:
            input_dim: Dimension of vehicle features
            state_dim: Dimension of global state
            hidden_dim: Hidden dimension of LSTM
            num_layers: Number of LSTM layers
        """
        # Create LSTM model
        model = LSTMModel(input_dim, state_dim, hidden_dim, num_layers).to(device)
        super().__init__(model, "LSTM")
