# Mamba Scheduler Training for MR-VFL

This document provides instructions for training and evaluating the Mamba scheduler for Multi-Resolution Vehicular Federated Learning (MR-VFL).

## Overview

The Mamba scheduler uses a state-space model architecture to efficiently select vehicles for participation in federated learning. This implementation includes:

1. A training script using Proximal Policy Optimization (PPO)
2. An evaluation script to compare the trained scheduler with other schedulers
3. Support for different training scenarios (standard, streaming, dynamic)

## Requirements

- Python 3.8+
- PyTorch 1.10+
- NumPy
- Matplotlib
- tqdm
- mamba_ssm (optional, will use GRU as fallback if not available)

## Training the Mamba Scheduler

The `train_mamba_scheduler.py` script implements PPO to train the Mamba scheduler. It supports various training scenarios and hyperparameters.

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
- `--n_episodes`: Number of episodes to train
- `--d_model`: Model dimension
- `--actor_lr`: Learning rate for actor
- `--critic_lr`: Learning rate for critic
- `--load_model`: Path to load pretrained model (optional)

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

## Evaluating the Mamba Scheduler

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

## Results

The evaluation script generates plots and summary files in the `results/evaluation/{timestamp}` directory. The plots include:

- Standard comparison results
- Streaming scenario results
- Dynamic scenario results (overall and by phase)
- Scheduling count results

The summary file (`results_summary.txt`) provides a detailed comparison of all schedulers across all evaluated scenarios.

## Improving the Scheduler

To improve the scheduler's performance, consider:

1. **Increasing model capacity**: Use larger `d_model` and `d_state` values
2. **Longer training**: Increase `n_episodes` for more training time
3. **Hyperparameter tuning**: Adjust learning rates, batch size, and other parameters
4. **Curriculum learning**: Start with simpler scenarios and gradually increase difficulty
5. **Reward shaping**: Modify the environment's reward function to better guide learning

## Example Training Command

For best results, try:

```bash
python train_mamba_scheduler.py --scenario dynamic --n_episodes 500 --vehicle_count 100 --d_model 256 --d_state 32 --actor_lr 0.0001 --critic_lr 0.0001 --batch_size 64 --n_epochs 20 --entropy_coef 0.02
```

This configuration provides a good balance between exploration and exploitation while training on the challenging dynamic scenario.
