# MR-VFL: Multi-Round Vehicular Federated Learning

This repository contains the implementation of the Multi-Round Vehicular Federated Learning (MR-VFL) framework with various scheduler implementations, including a Mamba-based scheduler. The framework focuses on efficient vehicle scheduling for federated learning in vehicular networks.

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
├── run_experiments.py       # Script to run standard experiments
├── run_simplified_enhanced_experiments.py # Script to run enhanced experiments
└── README.md                # This file
```

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

## Running the Experiments

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

## Training the Mamba Scheduler

To train the Mamba scheduler model:

```bash
python train_mamba_scheduler.py --epochs 100 --batch_size 64 --learning_rate 0.001
```

This will train the Mamba scheduler model and save it to `models/mamba_scheduler.pth`.

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

## Experiment Parameters

### Standard Experiments

- `--num_episodes`: Number of episodes to run (default: 5)
- `--max_rounds`: Maximum number of rounds per episode (default: 100)

### Enhanced Comparison Experiment

- `--num_episodes`: Number of episodes to run (default: 1)
- `--max_rounds`: Maximum number of rounds per episode (default: 10)
- `--interference_level`: Level of interference between vehicles (default: 0.3)
- `--resource_constraint`: Resource constraint factor (default: 0.7)
- `--channel_variation`: Channel variation factor (default: 0.2)

### Enhanced Streaming Experiment

- Vehicle count: 100
- Traffic density: 20 vehicles per km²
- `--burst_intensity`: Intensity of arrival bursts (default: 0.7)
- `--quality_variation`: Variation in data quality over time (default: 0.3)
- `--deadline_strictness`: Strictness of deadlines (default: 0.8)

### Enhanced Dynamic Experiment

- Vehicle count: 50
- Traffic density: 10 vehicles per km²
- Phase length: 10 rounds per phase
- `--phase_difficulty`: Difficulty increase per phase (default: 0.7)
- `--specialization_level`: Level of vehicle specialization (default: 0.8)
- Phase characteristics:
  - Phase 1: Balanced vehicle distribution
  - Phase 2: More high-quality vehicles
  - Phase 3: Vehicles with high mobility
  - Phase 4: Limited resources available

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
