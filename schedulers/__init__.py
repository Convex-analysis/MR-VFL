"""
Schedulers for MR-VFL comparison.
"""

from .base_scheduler import BaseScheduler
from .ml_scheduler import MLBasedScheduler
from .heuristic_scheduler import HeuristicScheduler
from .transformer_scheduler import TransformerScheduler
from .lstm_scheduler import LSTMScheduler
from .mamba_scheduler import MambaScheduler
from .greedy_scheduler import GreedyQualityScheduler, GreedyComputeScheduler, RandomScheduler

def initialize_schedulers():
    """
    Initialize all schedulers for comparison.

    Returns:
        Dictionary of scheduler instances
    """
    # Initialize schedulers
    schedulers = {
        "Mamba": MambaScheduler(model_path="models/mamba_scheduler.pth"),
        "Transformer": TransformerScheduler(),
        "LSTM": LSTMScheduler(),
        "Greedy-Quality": GreedyQualityScheduler(),
        "Greedy-Compute": GreedyComputeScheduler(),
        "Random": RandomScheduler()
    }

    return schedulers
