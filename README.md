# MR-VFL: Multi-Round Vehicular Federated Learning

This repository contains the implementation of the MR-VFL framework with various scheduler implementations, including a Mamba-based scheduler.

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
│   └── scheduling_count_scenario.py # Scheduling count scenario
├── utils/                   # Utility functions
│   ├── visualization.py     # Visualization utilities
│   └── metrics.py           # Metrics utilities
├── config.py                # Configuration
├── main.py                  # Main script
└── README.md                # This file
```

## Experiments

The framework includes several experiments to compare different schedulers:

1. **Standard Comparison**: Compares all schedulers in a standard environment.
2. **Streaming Scenario**: Tests schedulers in a high-arrival-rate environment.
3. **Dynamic Scenario**: Tests schedulers in a changing environment with different phases.
4. **Scheduling Count Scenario**: Counts and compares how many vehicles are successfully scheduled by each scheduler.

## Running the Experiments

To run all experiments:

```bash
python main.py
```

To run a specific experiment:

```bash
python main.py --experiment standard
python main.py --experiment streaming
python main.py --experiment dynamic
python main.py --experiment scheduling_count
```

## Schedulers

The framework includes the following schedulers:

1. **Mamba**: A scheduler based on the Mamba SSM architecture.
2. **Transformer**: A scheduler based on the Transformer architecture.
3. **LSTM**: A scheduler based on the LSTM architecture.
4. **Greedy-Quality**: A greedy scheduler that selects vehicles based on data quality.
5. **Greedy-Compute**: A greedy scheduler that selects vehicles based on computation capacity.
6. **Random**: A random scheduler that selects vehicles randomly.

## Environments

The framework includes the following environments:

1. **VehicularFLEnv**: Base environment for vehicular federated learning.
2. **StreamingVehicularFLEnv**: Environment with high arrival rate of vehicles.
3. **DynamicVehicularFLEnv**: Environment with changing vehicle characteristics.
4. **SchedulingCountEnv**: Environment for tracking scheduling statistics.

## Results

Results are saved to the `MR-VFL/results` directory, including:

- Performance metrics
- Decision time metrics
- Plots and visualizations
- Summary reports
