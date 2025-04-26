"""
Configuration for MR-VFL scheduler comparison.
"""

import os
import torch

# Device configuration
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Output directory
output_dir = os.path.join("MR-VFL", "results")
os.makedirs(output_dir, exist_ok=True)

# Random seed
random_seed = 42

# Scheduler configuration
scheduler_config = {
    # Mamba scheduler
    "mamba": {
        "model_path": "models/mamba_scheduler.pth",
        "input_dim": 6,
        "state_dim": 6,
        "d_model": 128,
        "n_layers": 2
    },
    
    # Transformer scheduler
    "transformer": {
        "input_dim": 6,
        "state_dim": 6,
        "d_model": 128,
        "nhead": 4,
        "num_layers": 2
    },
    
    # LSTM scheduler
    "lstm": {
        "input_dim": 6,
        "state_dim": 6,
        "hidden_dim": 128,
        "num_layers": 2
    }
}

# Environment configuration
env_config = {
    # Standard environment
    "standard": {
        "vehicle_count": 50,
        "max_round": 100,
        "sync_limit": 1000,
        "traffic_density": 10,
        "data_categories": 10,
        "learning_rate": 0.01
    },
    
    # Streaming environment
    "streaming": {
        "vehicle_count": 100,
        "max_round": 50,
        "sync_limit": 1000,
        "traffic_density": 20
    },
    
    # Dynamic environment
    "dynamic": {
        "vehicle_count": 50,
        "max_round": 40,
        "sync_limit": 1000,
        "phase_length": 10,
        "traffic_density": 10
    },
    
    # Scheduling count environment
    "scheduling_count": {
        "vehicle_count": 100,
        "max_round": 20,
        "sync_limit": 1000,
        "traffic_density": 10
    }
}

# Experiment configuration
experiment_config = {
    # Standard comparison
    "standard": {
        "num_episodes": 5,
        "max_rounds": 100
    },
    
    # Streaming scenario
    "streaming": {
        "window_size": 50
    },
    
    # Dynamic scenario
    "dynamic": {
        "phase_names": [
            "Balanced Urban",
            "High Compute/Low Quality",
            "Low Compute/High Quality",
            "Poor Connectivity Rural"
        ]
    },
    
    # Scheduling count scenario
    "scheduling_count": {
        "num_episodes": 3
    }
}
