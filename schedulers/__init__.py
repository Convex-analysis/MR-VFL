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
    # Import config
    from config import scheduler_config

    # Initialize schedulers
    schedulers = {
        "Mamba": MambaScheduler(
            model_path=scheduler_config["mamba"]["model_path"],
            input_dim=scheduler_config["mamba"]["input_dim"],
            state_dim=scheduler_config["mamba"]["state_dim"],
            d_model=scheduler_config["mamba"]["d_model"],
            n_layers=scheduler_config["mamba"]["n_layers"]
        ),
        "Transformer": TransformerScheduler(),
        "LSTM": LSTMScheduler(),
        "Greedy-Quality": GreedyQualityScheduler(),
        "Greedy-Compute": GreedyComputeScheduler(),
        "Random": RandomScheduler()
    }

    return schedulers
