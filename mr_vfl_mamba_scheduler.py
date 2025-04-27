# -*- coding: utf-8 -*-
"""
Multi-Resolution Vehicular Federated Learning (MR-VFL) Scheduler with Mamba Architecture
This scheduler uses a Mamba-based Actor-Critic architecture to optimize vehicle selection
with fairness constraints and dual-timescale optimization.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import deque
import random
import time

# Try to import Mamba SSM, use GRU as fallback if not available
try:
    from mamba_ssm import Mamba
    MAMBA_AVAILABLE = True
    print("Using Mamba SSM for sequence modeling")
except ImportError:
    MAMBA_AVAILABLE = False
    print("Mamba SSM not available, using GRU as fallback")

# Set device to cpu or cuda
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    # Enable cuDNN benchmarking for better performance
    torch.backends.cudnn.benchmark = True
    print(f"Device set to: {torch.cuda.get_device_name(device)}")
else:
    print(f"Device set to: {device}")

################################## Define Data Structures ##################################

class Vehicle:
    """Vehicle class representing a connected vehicle in the system"""
    # Class-level tensor cache for common dtypes
    _tensor_cache = {}

    def __init__(self, vehicle_id, model_version, sojourn_time, compute_capacity,
                 data_quality, connectivity, vehicle_type):
        self.vehicle_id = vehicle_id
        self.model_version = model_version  # Current model version
        self.sojourn_time = sojourn_time    # Expected available time
        self.compute_capacity = compute_capacity  # Processing capability
        self.data_quality = data_quality    # Data quality metric
        self.connectivity = connectivity    # Network conditions
        self.vehicle_type = vehicle_type    # Vehicle type/category
        self.arrival_time = None
        self.scheduled = False
        self._tensor = None  # Cache for tensor representation

        # Pre-compute tensor values as a list for faster conversion
        self._tensor_values = [
            self.model_version,
            self.sojourn_time,
            self.compute_capacity,
            self.data_quality,
            self.connectivity,
            self.vehicle_type
        ]

    def to_tensor(self, dtype=torch.float32):
        """Convert vehicle attributes to tensor for model input - optimized version"""
        # Use cached tensor if available and dtype matches
        if self._tensor is not None and self._tensor.dtype == dtype:
            return self._tensor

        # Create new tensor with specified dtype - optimized for speed
        # Use pre-allocated tensor if possible
        tensor_key = (self.vehicle_id, dtype)
        if tensor_key in Vehicle._tensor_cache:
            # Reuse pre-allocated tensor
            self._tensor = Vehicle._tensor_cache[tensor_key]
            # Fast update using torch operations
            self._tensor.copy_(torch.tensor(self._tensor_values, dtype=dtype, device=device))
        else:
            # Create new tensor directly on device
            self._tensor = torch.tensor(self._tensor_values, dtype=dtype, device=device)
            # Cache for future use (limit cache size to prevent memory issues)
            if len(Vehicle._tensor_cache) < 1000:  # Limit cache size
                Vehicle._tensor_cache[tensor_key] = self._tensor

        return self._tensor

class GlobalState:
    """Global state of the federated learning system"""
    def __init__(self):
        self.current_model_performance = 0.0
        self.round_number = 0
        self.elapsed_time = 0.0
        self.scheduled_count = 0
        self.target_vehicle_count = 0
        self.performance_gap = 1.0  # Gap between current and target performance

    def to_tensor(self, dtype=torch.float32):
        """Convert global state to tensor for model input"""
        return torch.tensor([
            self.current_model_performance,
            self.round_number,
            self.elapsed_time,
            self.scheduled_count,
            self.target_vehicle_count,
            self.performance_gap
        ], dtype=dtype).to(device)

    def update(self, scheduled_vehicles):
        """Update global state based on scheduled vehicles"""
        self.scheduled_count += len(scheduled_vehicles)
        self.round_number += 1
        # Update model performance based on scheduled vehicles
        if scheduled_vehicles:
            # Simple model: performance increases based on data quality and compute capacity
            avg_quality = sum(v.data_quality for v in scheduled_vehicles) / len(scheduled_vehicles)
            avg_compute = sum(v.compute_capacity for v in scheduled_vehicles) / len(scheduled_vehicles)
            performance_increase = 0.01 * avg_quality * avg_compute
            self.current_model_performance += performance_increase
            self.performance_gap = max(0, 1.0 - self.current_model_performance)

################################## Define Mamba Actor Network ##################################

class MRVFLMambaActor(nn.Module):
    """
    Optimized Mamba-based actor for vehicle selection.
    Key features:
    - Linear time complexity O(n) with sequence length
    - Efficient processing of streaming vehicle arrivals
    - State-space model for maintaining context
    - Outputs both vehicle selection probabilities and continuous action parameters
    - Optimized for inference speed
    """
    def __init__(self, input_dim=6, state_dim=6, d_model=128, d_state=16):
        super(MRVFLMambaActor, self).__init__()

        # Vehicle feature embedding
        self.vehicle_embedding = nn.Linear(input_dim, d_model)

        # Global state embedding
        self.state_embedding = nn.Linear(state_dim, d_model)

        # Sequence modeling layers (Mamba or GRU)
        if MAMBA_AVAILABLE:
            # Use a single Mamba layer with optimized parameters for faster inference
            self.sequence_layer = Mamba(
                d_model=d_model,
                d_state=d_state,
                d_conv=2,  # Reduced convolution size
                expand=1    # No expansion for faster computation
            )
        else:
            # Fallback to GRU if Mamba is not available
            self.sequence_layer = nn.GRU(
                input_size=d_model,
                hidden_size=d_model,
                num_layers=1,  # Single layer for speed
                batch_first=True
            )

        # Vehicle selection head (discrete action)
        self.selection_head = nn.Linear(d_model, 1)

        # Combined continuous action head for efficiency
        self.continuous_head = nn.Sequential(
            nn.Linear(d_model, 32),  # Smaller hidden layer
            nn.ReLU(),
            nn.Linear(32, 3),        # Combined output for all continuous actions
            nn.Sigmoid()             # Normalized between 0 and 1, will be scaled later
        )

    def forward(self, vehicles, global_state, mask=None):
        """
        Optimized forward pass through the actor network

        Args:
            vehicles: List of vehicle tensors
            global_state: Global state tensor
            mask: Optional mask for unavailable vehicles

        Returns:
            selection_probs: Vehicle selection probabilities
            alpha: Amplification factor (scaled to [0.5, 2.0])
            scheduled_time: Scheduled time offset (scaled based on environment)
            bandwidth: Bandwidth allocation (scaled to [0.1, 1.0])
        """
        batch_size = len(vehicles)

        # Embed global state
        state_embed = self.state_embedding(global_state)

        # Batch process vehicle embeddings
        vehicles_tensor = torch.stack(vehicles)
        vehicle_embeds = self.vehicle_embedding(vehicles_tensor)

        # Concatenate state with each vehicle embedding
        state_expanded = state_embed.unsqueeze(0).expand(batch_size, -1)
        combined_embeds = vehicle_embeds + state_expanded

        # Process through sequence layer
        if MAMBA_AVAILABLE:
            # Add sequence dimension for Mamba: (batch, dim) -> (1, batch, dim)
            x = combined_embeds.unsqueeze(0)  # shape: (1, batch, d_model)
            x = self.sequence_layer(x)
            x = x.squeeze(0)  # shape: (batch, d_model)
        else:
            # Process through GRU
            x = combined_embeds.unsqueeze(0)  # Add batch dimension
            x, _ = self.sequence_layer(x)
            x = x.squeeze(0)  # Remove batch dimension

        # Vehicle selection logits
        selection_logits = self.selection_head(x).squeeze(-1)

        # Apply mask if provided
        if mask is not None:
            # Use a smaller value for masking that's compatible with half-precision
            mask_value = -65504.0 if selection_logits.dtype == torch.float16 else -1e9
            selection_logits = selection_logits.masked_fill(mask == 0, mask_value)

        # Selection probabilities
        selection_probs = F.softmax(selection_logits, dim=0)

        # Continuous action parameters (combined for efficiency)
        continuous_actions = self.continuous_head(x)

        # Extract and scale continuous actions
        alpha_raw = continuous_actions[:, 0]
        time_raw = continuous_actions[:, 1]
        bandwidth_raw = continuous_actions[:, 2]

        # Scale continuous actions to appropriate ranges
        alpha = 0.5 + 1.5 * alpha_raw  # Scale to [0.5, 2.0]
        scheduled_time = 10 + 90 * time_raw  # Scale to [10, 100] offset
        bandwidth = 0.1 + 0.9 * bandwidth_raw  # Scale to [0.1, 1.0]

        return selection_probs, alpha, scheduled_time, bandwidth

################################## Define Mamba Critic Network ##################################

class MRVFLMambaCritic(nn.Module):
    """
    Optimized Mamba-based critic for state-value estimation.
    Evaluates the quality of current state and scheduling decisions.
    """
    def __init__(self, input_dim=6, state_dim=6, d_model=128, d_state=16):
        super(MRVFLMambaCritic, self).__init__()

        # Vehicle feature embedding
        self.vehicle_embedding = nn.Linear(input_dim, d_model)

        # Global state embedding
        self.state_embedding = nn.Linear(state_dim, d_model)

        # Action embedding (for continuous actions)
        self.action_embedding = nn.Sequential(
            nn.Linear(3, 32),  # alpha, scheduled_time, bandwidth
            nn.ReLU(),
            nn.Linear(32, d_model)
        )

        # Sequence modeling layer (Mamba or GRU)
        if MAMBA_AVAILABLE:
            self.sequence_layer = Mamba(
                d_model=d_model,
                d_state=d_state,
                d_conv=2,
                expand=1
            )
        else:
            # Fallback to GRU if Mamba is not available
            self.sequence_layer = nn.GRU(
                input_size=d_model,
                hidden_size=d_model,
                num_layers=1,
                batch_first=True
            )

        # Value head
        self.value_head = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, vehicles, global_state, actions):
        """
        Forward pass through the critic network

        Args:
            vehicles: List of vehicle tensors
            global_state: Global state tensor
            actions: Tuple of (vehicle_idx, alpha, scheduled_time, bandwidth)

        Returns:
            value: Estimated state-action value
        """
        # Extract actions
        vehicle_idx, alpha, scheduled_time, bandwidth = actions

        # Get selected vehicle tensor
        if isinstance(vehicle_idx, int):
            # Single action case
            vehicle_tensor = vehicles[vehicle_idx]
        else:
            # Batch case - not implemented for simplicity
            raise NotImplementedError("Batch processing not implemented for critic")

        # Embed vehicle features
        vehicle_embed = self.vehicle_embedding(vehicle_tensor)

        # Embed global state
        state_embed = self.state_embedding(global_state)

        # Embed continuous actions
        action_tensor = torch.tensor([alpha, scheduled_time, bandwidth],
                                     dtype=torch.float32).to(device)
        action_embed = self.action_embedding(action_tensor)

        # Combine embeddings (vehicle + state + action)
        combined_embed = vehicle_embed + state_embed + action_embed

        # Process through sequence layer
        if MAMBA_AVAILABLE:
            # Add sequence dimension for Mamba: (dim) -> (1, 1, dim)
            x = combined_embed.unsqueeze(0).unsqueeze(0)  # shape: (1, 1, d_model)
            x = self.sequence_layer(x)
            x = x.squeeze(0).squeeze(0)  # shape: (d_model)
        else:
            # Process through GRU
            x = combined_embed.unsqueeze(0).unsqueeze(0)  # Add batch and sequence dimensions
            x, _ = self.sequence_layer(x)
            x = x.squeeze(0).squeeze(0)  # Remove batch and sequence dimensions

        # Output state-action value
        return self.value_head(x)

################################## Define Experience Replay Buffer ##################################

class EpisodicReplayMemory:
    """Experience replay buffer for training the Mamba scheduler"""
    def __init__(self, capacity=10000, max_episode_length=200):
        self.num_episodes = capacity // max_episode_length
        self.buffer = deque(maxlen=self.num_episodes)
        self.buffer.append([])
        self.position = 0

    def push(self, vehicles, global_state, actions, reward, next_vehicles, next_global_state, done):
        self.buffer[self.position].append((vehicles, global_state, actions, reward, next_vehicles, next_global_state, done))
        if done:
            self.buffer.append([])
            self.position = min(self.position + 1, self.num_episodes - 1)

    def sample(self, batch_size, max_len=None):
        min_len = 0
        while min_len == 0:
            rand_episodes = random.sample(self.buffer, batch_size)
            min_len = min(len(episode) for episode in rand_episodes)

        if max_len:
            max_len = min(max_len, min_len)
        else:
            max_len = min_len

        episodes = []
        for episode in rand_episodes:
            if len(episode) > max_len:
                rand_idx = random.randint(0, len(episode) - max_len)
            else:
                rand_idx = 0

            episodes.append(episode[rand_idx:rand_idx+max_len])

        return list(map(list, zip(*episodes)))

    def __len__(self):
        return len(self.buffer)

################################## Create Streaming Scheduler ##################################

class StreamingMRVFLScheduler:
    """
    Optimized scheduler that handles real-time vehicle arrivals with O(1) complexity per new vehicle.
    Maintains persistent state for efficient updates.
    """
    def __init__(self, actor, critic=None, max_vehicles=100):
        self.actor = actor
        self.critic = critic
        self.vehicle_cache = []
        self.global_state = GlobalState()
        self.max_vehicles = max_vehicles
        self.selection_history = {}  # Track selection history for fairness

        # Pre-allocate tensors for faster inference
        self.cached_mask = None

        # Determine dtype from actor parameters
        self.cached_dtype = torch.float32
        if hasattr(self.actor, 'parameters'):
            try:
                self.cached_dtype = next(self.actor.parameters()).dtype
            except (StopIteration, RuntimeError):
                pass  # Use default dtype if no parameters or other error

    def process_new_vehicle(self, vehicle):
        """Process single vehicle arrival efficiently"""
        vehicle.arrival_time = self.global_state.elapsed_time
        self.vehicle_cache.append(vehicle)

        # Initialize selection history for this vehicle
        if vehicle.vehicle_id not in self.selection_history:
            self.selection_history[vehicle.vehicle_id] = 0

        # Limit cache size by removing oldest vehicles if needed
        if len(self.vehicle_cache) > self.max_vehicles:
            self.vehicle_cache.pop(0)

        return vehicle

    def make_scheduling_decision(self, target_count=10):
        """Generate scheduling actions based on current state - highly optimized for speed"""
        if not self.vehicle_cache:
            return []

        # Use cached dtype if available, otherwise determine from actor parameters
        if self.cached_dtype is None:
            self.cached_dtype = torch.float32
            if hasattr(self.actor, 'parameters'):
                try:
                    self.cached_dtype = next(self.actor.parameters()).dtype
                except StopIteration:
                    pass  # Use default dtype if no parameters

        # Fast path: prepare mask and tensors in a single pass
        cache_size = len(self.vehicle_cache)

        # Prepare mask - reuse if possible
        if self.cached_mask is None or self.cached_mask.shape[0] != cache_size:
            self.cached_mask = torch.ones(cache_size, dtype=self.cached_dtype, device=device)
        else:
            self.cached_mask.fill_(1)  # Reset mask to all ones

        # Prepare vehicle tensors and update mask in a single pass
        vehicle_tensors = []
        available_indices = []

        for i, vehicle in enumerate(self.vehicle_cache):
            # Check if vehicle is available
            if vehicle.scheduled or vehicle.sojourn_time < 1.0:
                self.cached_mask[i] = 0
            else:
                available_indices.append(i)

            # Get tensor representation
            vehicle_tensors.append(vehicle.to_tensor(dtype=self.cached_dtype))

        # Fast return if no vehicles are available
        if not available_indices:
            return []

        # Get global state tensor
        global_state_tensor = self.global_state.to_tensor().to(self.cached_dtype)

        # Fast inference path
        with torch.no_grad():
            selection_probs, _, _, _ = self.actor(vehicle_tensors, global_state_tensor, self.cached_mask)

        # Fast selection using PyTorch operations
        # Get probabilities for available vehicles
        available_probs = selection_probs[available_indices].cpu()

        # Check for valid probabilities
        if torch.sum(available_probs) > 0 and not torch.isnan(available_probs).any():
            # Normalize probabilities
            available_probs = available_probs / torch.sum(available_probs)

            # Convert to numpy for random selection
            probs_np = available_probs.numpy()

            # Select vehicles without replacement
            num_to_select = min(target_count, len(available_indices))
            try:
                selected_positions = np.random.choice(
                    len(available_indices),
                    size=num_to_select,
                    replace=False,
                    p=probs_np
                )
                selected_indices = [available_indices[pos] for pos in selected_positions]
            except ValueError:
                # Fallback for numerical issues
                selected_indices = available_indices[:num_to_select]
        else:
            # Fallback to deterministic selection
            num_to_select = min(target_count, len(available_indices))
            selected_indices = available_indices[:num_to_select]

        # Prepare selected vehicles list efficiently
        selected_vehicles = []
        for idx in selected_indices:
            vehicle = self.vehicle_cache[idx]
            vehicle.scheduled = True
            self.selection_history[vehicle.vehicle_id] += 1
            selected_vehicles.append(vehicle)

        # Update global state
        self.global_state.update(selected_vehicles)

        return selected_vehicles

################################## Optimize for Production Deployment ##################################

class OptimizedMRVFLScheduler:
    """
    Production-ready scheduler with optimizations:
    - Model quantization (int8/fp16)
    - JIT compilation
    - Hardware-specific acceleration
    - Optimized for inference speed
    - Streamlined processing pipeline
    """
    def __init__(self, pretrained_model_path=None):
        # Initialize models
        vehicle_feature_dim = 6
        global_state_dim = 6

        # Create actor model with optimized parameters
        self.actor = MRVFLMambaActor(
            input_dim=vehicle_feature_dim,
            state_dim=global_state_dim,
            d_model=128,  # Reduced model dimension
            d_state=16
        ).to(device)

        # Load pretrained model if provided
        if pretrained_model_path and os.path.exists(pretrained_model_path):
            try:
                checkpoint = torch.load(pretrained_model_path, map_location=device)

                # Check if we need to convert from old model architecture to new one
                if 'sequence_layers.0.A_log' in checkpoint['actor_state_dict']:
                    print("Converting from old model architecture to new optimized architecture")

                    # Create a temporary model with the old architecture to load the weights
                    class LegacyMambaActor(nn.Module):
                        def __init__(self, input_dim=6, state_dim=6, d_model=256, n_layers=3, d_state=16):
                            super(LegacyMambaActor, self).__init__()

                            # Vehicle feature embedding
                            self.vehicle_embedding = nn.Linear(input_dim, d_model)

                            # Global state embedding
                            self.state_embedding = nn.Linear(state_dim, d_model)

                            # Sequence modeling layers
                            if MAMBA_AVAILABLE:
                                self.sequence_layers = nn.ModuleList([
                                    Mamba(
                                        d_model=d_model,
                                        d_state=d_state,
                                        d_conv=4,
                                        expand=2
                                    ) for _ in range(n_layers)
                                ])
                            else:
                                self.sequence_layers = nn.GRU(
                                    input_size=d_model,
                                    hidden_size=d_model,
                                    num_layers=n_layers,
                                    batch_first=True
                                )

                            # Vehicle selection head
                            self.selection_head = nn.Linear(d_model, 1)

                            # Continuous action heads
                            self.alpha_head = nn.Sequential(
                                nn.Linear(d_model, 64),
                                nn.ReLU(),
                                nn.Linear(64, 1),
                                nn.Sigmoid()
                            )

                            self.time_head = nn.Sequential(
                                nn.Linear(d_model, 64),
                                nn.ReLU(),
                                nn.Linear(64, 1),
                                nn.Sigmoid()
                            )

                            self.bandwidth_head = nn.Sequential(
                                nn.Linear(d_model, 64),
                                nn.ReLU(),
                                nn.Linear(64, 1),
                                nn.Sigmoid()
                            )

                    # Create legacy model and load weights
                    legacy_model = LegacyMambaActor().to(device)
                    legacy_model.load_state_dict(checkpoint['actor_state_dict'])

                    # Initialize new model with transferred weights
                    # Transfer embeddings with dimension adjustment
                    with torch.no_grad():
                        # Transfer embeddings with dimension reduction
                        self.actor.vehicle_embedding.weight.data = legacy_model.vehicle_embedding.weight.data[:128, :]
                        self.actor.vehicle_embedding.bias.data = legacy_model.vehicle_embedding.bias.data[:128]

                        self.actor.state_embedding.weight.data = legacy_model.state_embedding.weight.data[:128, :]
                        self.actor.state_embedding.bias.data = legacy_model.state_embedding.bias.data[:128]

                        # Transfer selection head with dimension adjustment
                        self.actor.selection_head.weight.data = legacy_model.selection_head.weight.data[:, :128]
                        self.actor.selection_head.bias.data = legacy_model.selection_head.bias.data

                        # Transfer Mamba layer parameters if available
                        if MAMBA_AVAILABLE:
                            # Use first layer parameters from legacy model
                            for name, param in legacy_model.sequence_layers[0].named_parameters():
                                if hasattr(self.actor.sequence_layer, name):
                                    target_param = getattr(self.actor.sequence_layer, name)
                                    if param.shape == target_param.shape:
                                        target_param.data.copy_(param.data)

                        # Initialize continuous head from legacy heads
                        # Input weights
                        self.actor.continuous_head[0].weight.data[:, :128] = torch.cat([
                            legacy_model.alpha_head[0].weight.data[:16, :128],
                            legacy_model.time_head[0].weight.data[:8, :128],
                            legacy_model.bandwidth_head[0].weight.data[:8, :128]
                        ], dim=0)

                        # Input biases
                        self.actor.continuous_head[0].bias.data = torch.cat([
                            legacy_model.alpha_head[0].bias.data[:16],
                            legacy_model.time_head[0].bias.data[:8],
                            legacy_model.bandwidth_head[0].bias.data[:8]
                        ], dim=0)

                        # Output weights
                        self.actor.continuous_head[2].weight.data[0, :] = legacy_model.alpha_head[2].weight.data.mean(dim=1)
                        self.actor.continuous_head[2].weight.data[1, :] = legacy_model.time_head[2].weight.data.mean(dim=1)
                        self.actor.continuous_head[2].weight.data[2, :] = legacy_model.bandwidth_head[2].weight.data.mean(dim=1)

                        # Output biases
                        self.actor.continuous_head[2].bias.data[0] = legacy_model.alpha_head[2].bias.data.mean()
                        self.actor.continuous_head[2].bias.data[1] = legacy_model.time_head[2].bias.data.mean()
                        self.actor.continuous_head[2].bias.data[2] = legacy_model.bandwidth_head[2].bias.data.mean()

                    print("Successfully transferred weights from legacy model to optimized model")
                else:
                    # Direct loading for compatible models
                    self.actor.load_state_dict(checkpoint['actor_state_dict'])
                    print(f"Loaded model from {pretrained_model_path}")
            except Exception as e:
                print(f"Error loading model: {e}")
                print("Using randomly initialized model")
        else:
            print("Using randomly initialized model")

        # Set to evaluation mode
        self.actor.eval()

        # Try to use half precision if available
        if device.type == 'cuda':
            try:
                self.actor = self.actor.half()
                print("Using half precision (FP16) for faster inference")
                self.use_half = True
            except Exception as e:
                print(f"Could not use half precision: {e}")
                self.use_half = False
        else:
            self.use_half = False

        # Skip TorchScript for now as it requires fixed input sizes
        # We'll use eager mode with optimizations instead
        print("Using eager mode with optimizations for inference")
        self.use_jit = False

        # Initialize scheduler with optimized actor
        self.scheduler = StreamingMRVFLScheduler(self.actor)

        # Cache for vehicle tensors to avoid repeated conversions
        self.tensor_cache = {}

    def inference_mode_scheduling(self, vehicles, global_state, target_count=10):
        """Fast inference path for production"""
        # Update scheduler state
        self.scheduler.global_state = global_state
        self.scheduler.vehicle_cache = []

        # Process vehicles with tensor caching
        for vehicle in vehicles:
            # Check if we already have a tensor for this vehicle
            if vehicle.vehicle_id in self.tensor_cache:
                vehicle._tensor = self.tensor_cache[vehicle.vehicle_id]
            self.scheduler.process_new_vehicle(vehicle)
            # Cache the tensor for future use
            if vehicle._tensor is not None:
                self.tensor_cache[vehicle.vehicle_id] = vehicle._tensor

        # Make scheduling decision
        with torch.no_grad():
            selected_vehicles = self.scheduler.make_scheduling_decision(target_count)

        return selected_vehicles

    def select_vehicles(self, env_vehicles, target_count=10):
        """
        Interface method for compatibility with comparison framework

        Args:
            env_vehicles: List of environment vehicles
            target_count: Number of vehicles to select

        Returns:
            selected_vehicles: List of selected vehicles
        """
        # Convert environment vehicles to scheduler vehicles efficiently
        scheduler_vehicles = []

        # Prepare all vehicles at once
        for v in env_vehicles:
            if not v['scheduled']:
                # Use fallback to support both 'model_version' and 'version'
                model_version = v.get('model_version', v.get('version', 0.0))
                sojourn_time = v.get('sojourn_time', v.get('sojourn', 0.0))
                compute_capacity = v.get('compute_capacity', v.get('compute', 0.0))
                data_quality = v.get('data_quality', v.get('quality', 0.0))
                connectivity = v.get('connectivity', v.get('conn', 0.0))
                vehicle_type = v.get('vehicle_type', v.get('type', 0))
                vehicle = Vehicle(
                    vehicle_id=v['id'],
                    model_version=model_version,
                    sojourn_time=sojourn_time,
                    compute_capacity=compute_capacity,
                    data_quality=data_quality,
                    connectivity=connectivity,
                    vehicle_type=vehicle_type
                )
                scheduler_vehicles.append(vehicle)

        # Create global state
        global_state = GlobalState()

        # Make scheduling decision
        selected_vehicles = self.inference_mode_scheduling(
            scheduler_vehicles, global_state, target_count
        )

        # Convert back to environment vehicle format efficiently
        selected_ids = {sv.vehicle_id for sv in selected_vehicles}
        return [v for v in env_vehicles if v['id'] in selected_ids]

# Example usage
if __name__ == "__main__":
    # Create a dummy model for testing
    vehicle_feature_dim = 6
    global_state_dim = 6

    # Create model with optimized parameters
    actor = MRVFLMambaActor(
        input_dim=vehicle_feature_dim,
        state_dim=global_state_dim,
        d_model=128,
        d_state=16
    ).to(device)

    # Save dummy model
    os.makedirs("mr_vfl_models", exist_ok=True)
    torch.save({
        'actor_state_dict': actor.state_dict(),
        'critic_state_dict': {},
    }, 'mr_vfl_models/mr_vfl_mamba_scheduler_optimized.pt')

    # Test optimized scheduler
    optimized_scheduler = OptimizedMRVFLScheduler('mr_vfl_models/mr_vfl_mamba_scheduler_optimized.pt')

    # Generate test vehicles
    test_vehicles = [
        Vehicle(1, 0.5, 5.0, 0.8, 0.9, 0.7, 1),
        Vehicle(2, 0.3, 3.0, 0.6, 0.7, 0.8, 2),
        Vehicle(3, 0.7, 7.0, 0.9, 0.8, 0.9, 0)
    ]

    # Test scheduling
    test_global_state = GlobalState()

    # Warm-up run
    _ = optimized_scheduler.inference_mode_scheduling(test_vehicles, test_global_state)

    # Reset vehicles
    for v in test_vehicles:
        v.scheduled = False

    # Measure inference time with multiple runs for better accuracy
    num_runs = 100
    total_time = 0

    for _ in range(num_runs):
        # Reset vehicles
        for v in test_vehicles:
            v.scheduled = False

        # Measure inference time
        start_time = time.time()
        selected = optimized_scheduler.inference_mode_scheduling(test_vehicles, test_global_state)
        inference_time = (time.time() - start_time) * 1000  # ms
        total_time += inference_time

    avg_inference_time = total_time / num_runs
    print(f"Selected {len(selected)} vehicles for scheduling")
    print(f"Average inference time over {num_runs} runs: {avg_inference_time:.2f} ms")

    # Test with larger vehicle set to measure scaling
    large_test_vehicles = []
    for i in range(100):
        large_test_vehicles.append(
            Vehicle(
                i,
                np.random.uniform(0, 1),
                np.random.uniform(1, 10),
                np.random.uniform(0.1, 1.0),
                np.random.uniform(0.1, 1.0),
                np.random.uniform(0.1, 1.0),
                np.random.randint(0, 3)
            )
        )

    # Warm-up run
    _ = optimized_scheduler.inference_mode_scheduling(large_test_vehicles, test_global_state)

    # Reset vehicles
    for v in large_test_vehicles:
        v.scheduled = False

    # Measure inference time with multiple runs
    total_time = 0
    for _ in range(10):  # Fewer runs for the larger set
        # Reset vehicles
        for v in large_test_vehicles:
            v.scheduled = False

        # Measure inference time
        start_time = time.time()
        selected = optimized_scheduler.inference_mode_scheduling(large_test_vehicles, test_global_state)
        inference_time = (time.time() - start_time) * 1000  # ms
        total_time += inference_time

    avg_large_inference_time = total_time / 10
    print(f"Large test ({len(large_test_vehicles)} vehicles): Selected {len(selected)} vehicles")
    print(f"Average inference time: {avg_large_inference_time:.2f} ms")
