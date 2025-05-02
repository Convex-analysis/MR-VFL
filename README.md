# MR-VFL: Multi-Resolution Vehicular Federated Learning

This repository contains the implementation of the Multi-Resolution Vehicular Federated Learning (MR-VFL) framework with various scheduler implementations, including a Mamba-based scheduler. The framework focuses on efficient vehicle scheduling for federated learning in vehicular networks.

## Table of Contents

1. [Overview](#overview)
2. [Project Structure](#project-structure)
3. [Installation](#installation)
4. [Training the Mamba Scheduler](#training-the-mamba-scheduler)
5. [Evaluating the Scheduler](#evaluating-the-scheduler)
6. [Running Experiments](#running-experiments)
7. [Environment Configuration](#environment-configuration)
8. [Vehicle Attributes](#vehicle-attributes)
9. [Scheduler Types](#scheduler-types)
10. [Advanced Configuration](#advanced-configuration)

## Overview

The MR-VFL system implements a federated learning framework for vehicular networks, where vehicles contribute to a global model while maintaining their privacy. The key components are:

- **Mamba Scheduler**: Uses a state-space model to select vehicles for participation
- **Environment Simulation**: Realistic simulation of vehicular networks with dynamic mobility
- **Federated Learning**: Implementation of federated learning algorithms for distributed training
- **Evaluation Framework**: Comprehensive evaluation of different scheduling strategies

## Project Structure

```
MR-VFL/
├── models/                  # Neural network models
│   ├── base_models.py       # Base model classes
│   ├── transformer_model.py # Transformer model
│   ├── lstm_model.py        # LSTM model
│   └── mamba_model.py       # Mamba model
├── schedulers/              # Scheduler implementations
│   ├── base_scheduler.py    # Base scheduler class
│   ├── ml_scheduler.py      # ML-based scheduler
│   ├── heuristic_scheduler.py # Heuristic scheduler
│   ├── transformer_scheduler.py # Transformer scheduler
│   ├── lstm_scheduler.py    # LSTM scheduler
│   ├── mamba_scheduler.py   # Mamba scheduler
│   └── greedy_scheduler.py  # Greedy schedulers
├── environments/            # Environment implementations
│   ├── vehicular_fl_env.py  # Base environment
│   ├── streaming_env.py     # Streaming environment
│   ├── dynamic_env.py       # Dynamic environment
│   └── scheduling_count_env.py # Scheduling count environment
├── experiments/             # Experiment implementations
│   ├── standard_comparison.py # Standard comparison
│   ├── streaming_scenario.py # Streaming scenario
│   ├── dynamic_scenario.py  # Dynamic scenario
│   ├── scheduling_count_scenario.py # Scheduling count scenario
│   ├── simplified_enhanced_comparison.py # Enhanced comparison
│   ├── simplified_enhanced_streaming.py # Enhanced streaming
│   └── simplified_enhanced_dynamic.py # Enhanced dynamic
├── utils/                   # Utility functions
│   ├── visualization.py     # Visualization utilities
│   └── metrics.py           # Metrics utilities
├── train_mamba_scheduler.py # Script to train Mamba scheduler
├── evaluate_mamba_scheduler.py # Script to evaluate Mamba scheduler
├── run_experiments.py       # Script to run standard experiments
├── run_simplified_enhanced_experiments.py # Script to run enhanced experiments
└── README.md                # This file
```

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/MR-VFL.git
cd MR-VFL

# Install dependencies
python install_requirements.py
```

### Requirements

- Python 3.8+
- PyTorch 1.10+
- NumPy
- Matplotlib
- tqdm
- mamba_ssm (optional, will use GRU as fallback if not available)

## Training the Mamba Scheduler

The `train_mamba_scheduler.py` script implements Proximal Policy Optimization (PPO) to train the Mamba scheduler.

### Basic Usage

```bash
python train_mamba_scheduler.py
```

This will train the scheduler with default parameters in the standard scenario.

### Advanced Usage

```bash
python train_mamba_scheduler.py --scenario streaming --n_episodes 200 --vehicle_count 100 --d_model 256 --actor_lr 0.0001 --critic_lr 0.0001
```

### Key Parameters

- `--scenario`: Training scenario (standard, streaming, dynamic)
- `--vehicle_count`: Number of vehicles in the environment
- `--max_round`: Maximum number of rounds per episode
- `--traffic_density`: Traffic density (vehicles per km²)
- `--n_episodes`: Number of episodes to train
- `--d_model`: Model dimension
- `--d_state`: State dimension for Mamba
- `--actor_lr`: Learning rate for actor
- `--critic_lr`: Learning rate for critic
- `--load_model`: Path to load pretrained model (optional)
- `--batch_size`: Batch size for training
- `--n_epochs`: Number of epochs for each update
- `--entropy_coef`: Entropy coefficient for exploration

### Training Process

The training process uses PPO with the following steps:

1. Initialize the actor and critic networks
2. For each episode:
   - Reset the environment
   - For each step:
     - Select vehicles using the actor network
     - Take action in the environment
     - Store transition in memory
     - Update networks periodically
   - Update networks at the end of the episode
   - Save the best model based on performance

### Model Output

The trained model is saved to:
- `models/trained/{timestamp}/best_model.pth` - Best model based on performance
- `models/trained/{timestamp}/checkpoint_{episode}.pth` - Checkpoints during training
- `models/mamba_scheduler.pth` - Final model (used by default in experiments)

### Example Training Command

For best results, try:

```bash
python train_mamba_scheduler.py --scenario dynamic --n_episodes 500 --vehicle_count 100 --d_model 256 --d_state 32 --actor_lr 0.0001 --critic_lr 0.0001 --batch_size 64 --n_epochs 20 --entropy_coef 0.02
```

This configuration provides a good balance between exploration and exploitation while training on the challenging dynamic scenario.

## Evaluating the Scheduler

The `evaluate_mamba_scheduler.py` script evaluates the trained scheduler and compares it with other schedulers in different scenarios.

### Basic Usage

```bash
python evaluate_mamba_scheduler.py
```

This will evaluate the scheduler in all available scenarios.

### Advanced Usage

```bash
python evaluate_mamba_scheduler.py --scenarios standard streaming --num_episodes 10
```

### Key Parameters

- `--scenarios`: Scenarios to evaluate (standard, streaming, dynamic, scheduling_count)
- `--num_episodes`: Number of episodes for standard comparison
- `--max_rounds`: Maximum number of rounds per episode

### Evaluation Metrics

The evaluation script generates the following metrics:

- **Standard Comparison**: Average reward, performance, and decision time
- **Streaming Scenario**: Total reward, final performance, and average decision time
- **Dynamic Scenario**: Total reward, final performance, adaptability score, and phase performances
- **Scheduling Count**: Unique vehicles, high quality vehicles, low quality vehicles, and success rate

### Results

The evaluation script generates plots and summary files in the `results/evaluation/{timestamp}` directory. The plots include:

- Standard comparison results
- Streaming scenario results
- Dynamic scenario results (overall and by phase)
- Scheduling count results

The summary file (`results_summary.txt`) provides a detailed comparison of all schedulers across all evaluated scenarios.

## Experiments

The framework includes several experiments to compare different schedulers:

### Standard Experiments

1. **Standard Comparison**: Compares all schedulers in a standard environment.
2. **Streaming Scenario**: Tests schedulers in a high-arrival-rate environment.
3. **Dynamic Scenario**: Tests schedulers in a changing environment with different phases.
4. **Scheduling Count Scenario**: Counts and compares how many vehicles are successfully scheduled by each scheduler.

### Enhanced Experiments

Enhanced experiments feature more complex decision-making scenarios:

1. **Enhanced Comparison**: Adds interference, resource constraints, and channel variations.
2. **Enhanced Streaming**: Adds burst arrivals, data staleness, and deadline constraints.
3. **Enhanced Dynamic**: Adds phase-specific characteristics, vehicle specialization, and adaptability metrics.

## Running Experiments

### Standard Experiments

To run all standard experiments:

```bash
python run_experiments.py --experiment all
```

To run a specific standard experiment:

```bash
python run_experiments.py --experiment comparison
python run_experiments.py --experiment streaming
python run_experiments.py --experiment dynamic
python run_experiments.py --experiment scheduling_count
```

### Enhanced Experiments

Enhanced experiments feature more complex decision-making scenarios:

To run all enhanced experiments:

```bash
python run_simplified_enhanced_experiments.py --experiment all
```

To run a specific enhanced experiment:

```bash
python run_simplified_enhanced_experiments.py --experiment comparison
python run_simplified_enhanced_experiments.py --experiment streaming
python run_simplified_enhanced_experiments.py --experiment dynamic
```

### Using the Trained Model in Experiments

All scripts in the `experiments/` directory automatically use the model trained by `train_mamba_scheduler.py`. The integration works as follows:

1. The `train_mamba_scheduler.py` script saves the trained model to `models/mamba_scheduler.pth`
2. All experiment scripts use the `initialize_schedulers()` function from `schedulers/__init__.py`
3. This function creates a `MambaScheduler` instance with the model path from `config.py`
4. In `config.py`, the model path is set to `models/mamba_scheduler.pth`

You can run experiments with the trained model using the provided script:

```bash
python experiments/run_with_trained_model.py
```

This script allows you to override environment parameters:

```bash
python experiments/run_with_trained_model.py --vehicle_count 100 --traffic_density 15
```

You can also use a specific trained model with:

```bash
python experiments/use_trained_model.py --model_path models/trained/20250426_020937/best_model.pth
```

## Vehicle Attributes

Vehicles in the environment have the following attributes:

### Basic Attributes

- `id`: Unique identifier for the vehicle
- `computation_capacity`: Computational power (0.1-1.0)
- `data_quality`: Quality of the vehicle's data (0.0-1.0)
- `data_size`: Size of the vehicle's dataset (100-1000 units)
- `sojourn_time`: Time the vehicle stays in the area (100-2000 seconds)
- `arrival_time`: Time when the vehicle arrives (0-sync_limit/2)
- `departure_time`: Time when the vehicle departs (arrival_time + sojourn_time)
- `channel_gain`: Normalized channel gain (0.0-1.0)
- `scheduled`: Whether the vehicle has been scheduled (boolean)

### Communication Parameters

- `bandwidth_mhz`: Vehicle bandwidth (10-30 MHz)
- `tx_power_dbm`: Transmission power (23-30 dBm)
- `distance`: Distance from server (0-area_radius*1000 meters)
- `road_type`: Type of road (1-8)

### Status Attributes

- `selection_count`: Times selected by edge server
- `participation_count`: Times participated in training
- `comm_time`: Communication time (calculated based on bandwidth and channel gain)
- `training_time`: Training time (calculated based on computation capacity and data quality)
- `predicted_arrival`: Predicted arrival time

### Vehicle Types

In some environments (like the scheduling count environment), vehicles are categorized into types:

- **High Quality**: `data_quality > 0.7`
- **Medium Quality**: `0.3 < data_quality <= 0.7`
- **Low Quality**: `data_quality <= 0.3`

### Vehicle Attribute Ranges

The vehicle attributes are generated within the following ranges:

- **computation_capacity**: Uniformly distributed between 0.1 and 1.0
- **data_quality**: Depends on the environment and vehicle type:
  - Standard environment: Uniformly distributed between 0.1 and 0.9
  - Scheduling count environment:
    - High quality vehicles: 0.7 to 0.9
    - Medium quality vehicles: 0.3 to 0.7
    - Low quality vehicles: 0.1 to 0.3
- **data_size**: Uniformly distributed between 100 and 1000 units
- **sojourn_time**: Uniformly distributed between 100 and 2000 seconds
- **arrival_time**: Uniformly distributed between 0 and sync_limit/2
- **bandwidth_mhz**: Uniformly distributed between 10 and 30 MHz
- **tx_power_dbm**: Uniformly distributed between 23 and 30 dBm
- **distance**: Uniformly distributed between 0 and area_radius*1000 meters
- **road_type**: Integer between 1 and 8

## Scheduler Types

The system includes several scheduler types for comparison:

### Machine Learning-Based Schedulers

- **Mamba**: State-space model-based scheduler (our main contribution)
- **Transformer**: Transformer-based scheduler
- **LSTM**: LSTM-based scheduler

### Heuristic Schedulers

- **Greedy-Quality**: Selects vehicles with highest data quality
- **Greedy-Compute**: Selects vehicles with highest computation capacity
- **Random**: Selects vehicles randomly

## Environments

The framework includes the following environments:

1. **VehicularFLEnv**: Base environment for vehicular federated learning.
2. **StreamingVehicularFLEnv**: Environment with high arrival rate of vehicles.
3. **DynamicVehicularFLEnv**: Environment with changing vehicle characteristics.
4. **SchedulingCountEnv**: Environment for tracking scheduling statistics.

## Environment Configuration

The environment configuration is defined in `config.py`. You can modify these parameters to customize the experiments.

### Setting Environment Parameters

There are several ways to set environment parameters:

1. **Modify `config.py`**: Edit the environment configuration directly in the config file.

2. **Command-line Arguments**: Use the provided scripts with command-line arguments:
   ```bash
   python experiments/run_with_trained_model.py --vehicle_count 100 --traffic_density 15
   ```

3. **Training Script Parameters**: When training the model, specify environment parameters:
   ```bash
   python train_mamba_scheduler.py --vehicle_count 100 --traffic_density 15
   ```

4. **Evaluation Script Parameters**: When evaluating the model, specify environment parameters:
   ```bash
   python evaluate_mamba_scheduler.py --vehicle_count 100
   ```

### Key Environment Parameters

- **vehicle_count**: Number of vehicles in the environment
- **traffic_density**: Density of vehicles per square kilometer
- **max_round**: Maximum number of rounds per episode
- **sync_limit**: Synchronization time limit in seconds
- **data_categories**: Number of data categories (for standard environment)
- **learning_rate**: Learning rate for model updates (for standard environment)
- **phase_length**: Number of rounds per phase (for dynamic environment)

### Standard Environment

```python
"standard": {
    "vehicle_count": 50,      # Number of vehicles
    "max_round": 100,         # Maximum rounds
    "sync_limit": 1000,       # Synchronization time limit (seconds)
    "traffic_density": 10,    # Traffic density (vehicles per km²)
    "data_categories": 10,    # Number of data categories
    "learning_rate": 0.01     # Learning rate for model updates
}
```

### Streaming Environment

```python
"streaming": {
    "vehicle_count": 100,     # More vehicles for streaming
    "max_round": 50,          # Fewer rounds
    "sync_limit": 1000,       # Same sync limit
    "traffic_density": 20     # Higher traffic density
}
```

### Dynamic Environment

```python
"dynamic": {
    "vehicle_count": 50,      # Standard vehicle count
    "max_round": 40,          # 10 rounds per phase, 4 phases
    "sync_limit": 1000,       # Same sync limit
    "phase_length": 10,       # Rounds per phase
    "traffic_density": 10     # Standard traffic density
}
```

### Scheduling Count Environment

```python
"scheduling_count": {
    "vehicle_count": 100,     # More vehicles for analysis
    "max_round": 20,          # Fewer rounds
    "sync_limit": 1000,       # Same sync limit
    "traffic_density": 10     # Standard traffic density
}
```

### Relationship Between Parameters

- **vehicle_count and traffic_density**: The expected number of vehicles is calculated as `traffic_density * π * (area_radius)²`. If the expected number differs from `vehicle_count`, the environment will use `vehicle_count` but log a warning.

- **max_round and phase_length**: In the dynamic environment, each phase lasts for `phase_length` rounds, and there are 4 phases, so `max_round` should be at least `4 * phase_length`.

- **sync_limit**: This parameter sets the maximum time (in seconds) for synchronization in each round. It affects the arrival and departure times of vehicles.

## Advanced Configuration

### Scheduler Configuration

The scheduler configuration is defined in `config.py`:

```python
"mamba": {
    "model_path": "models/mamba_scheduler.pth",  # Path to the trained model
    "input_dim": 6,                             # Input dimension
    "state_dim": 6,                             # State dimension
    "d_model": 128,                             # Model dimension
    "n_layers": 2                               # Number of layers
}
```

### Experiment Configuration

The experiment configuration is also defined in `config.py`:

```python
"standard": {
    "num_episodes": 5,        # Number of episodes
    "max_rounds": 100         # Maximum rounds
}
```

### Environment Parameters

The environment has several parameters that affect the simulation:

- **Area Radius**: 2.5 km (default)
- **Road Types**: 8 different types
- **Server Bandwidth**: 50 MHz
- **Server Transmission Power**: 30 dBm
- **Path Loss Parameters**: Realistic path loss model
- **Reward Function Weights**: Configurable weights for different objectives

## Improving the Scheduler

To improve the scheduler's performance, consider:

1. **Increasing model capacity**: Use larger `d_model` and `d_state` values
2. **Longer training**: Increase `n_episodes` for more training time
3. **Hyperparameter tuning**: Adjust learning rates, batch size, and other parameters
4. **Curriculum learning**: Start with simpler scenarios and gradually increase difficulty
5. **Reward shaping**: Modify the environment's reward function to better guide learning

## Results

Results are saved to the `results/` directory, including:

- Performance metrics
- Decision time metrics
- Plots and visualizations
- Summary reports
- Phase-specific performance (for dynamic experiments)
- Adaptability scores (for dynamic experiments)
- Deadline miss rates (for streaming experiments)
- Data staleness metrics (for streaming experiments)

## License

This project is licensed under the MIT License - see the LICENSE file for details.
