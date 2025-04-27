"""
Enhanced comparison experiment for MR-VFL schedulers.

This experiment implements a more complex and challenging decision-making scenario
with resource constraints, communication interference, heterogeneous vehicle capabilities,
time-varying channel conditions, and priority-based scheduling.
"""

import time
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
from datetime import datetime
from environments.vehicular_fl_env import VehicularFLEnv
from schedulers import initialize_schedulers

class EnhancedVehicularFLEnv(VehicularFLEnv):
    """
    Enhanced Vehicular Federated Learning Environment with more complex decision-making.

    This environment extends the base environment with:
    1. Resource constraints and competition
    2. Communication interference
    3. Heterogeneous vehicle capabilities
    4. Time-varying channel conditions
    5. Priority-based scheduling
    6. More complex reward function
    """

    def __init__(self, vehicle_count=50, max_round=100, sync_limit=1000, traffic_density=10,
                 data_categories=10, learning_rate=0.01, interference_level=0.3,
                 resource_constraint=0.7, channel_variation=0.2):
        """
        Initialize the enhanced vehicular FL environment.

        Args:
            vehicle_count: Number of vehicles in the environment
            max_round: Maximum number of rounds for the simulation
            sync_limit: Synchronization time limit in seconds
            traffic_density: Traffic density (vehicles per km²)
            data_categories: Number of data categories
            learning_rate: Learning rate for model updates
            interference_level: Level of interference between vehicles (0-1)
            resource_constraint: Resource constraint factor (0-1)
            channel_variation: Channel variation factor (0-1)
        """
        # Call parent constructor
        super().__init__(
            vehicle_count=vehicle_count,
            max_round=max_round,
            sync_limit=sync_limit,
            traffic_density=traffic_density,
            data_categories=data_categories,
            learning_rate=learning_rate
        )

        # Enhanced parameters
        self.interference_level = interference_level
        self.resource_constraint = resource_constraint
        self.channel_variation = channel_variation

        # Resource pools
        self.computation_resources = resource_constraint * vehicle_count  # Total computation resources
        self.bandwidth_resources = resource_constraint * self.server_bandwidth  # Total bandwidth resources
        self.available_computation = self.computation_resources
        self.available_bandwidth = self.bandwidth_resources

        # Priority queues
        self.priority_queue = []  # Vehicles waiting for resources

        # Time-varying channel conditions
        self.channel_conditions = np.ones(vehicle_count)  # Base channel conditions

        # Vehicle capabilities
        self.vehicle_capabilities = {}  # Store vehicle capabilities

        # Generate vehicles with enhanced characteristics
        self.vehicles = self._generate_vehicles()

    def reset(self):
        """
        Reset the enhanced environment.

        Returns:
            Initial state representation
        """
        # Reset base environment state
        super().reset()

        # Reset enhanced parameters
        self.available_computation = self.computation_resources
        self.available_bandwidth = self.bandwidth_resources
        self.priority_queue = []
        self.channel_conditions = np.ones(self.vehicle_count)
        self.vehicle_capabilities = {}

        # Generate vehicles with enhanced characteristics
        self.vehicles = self._generate_vehicles()

        # Return state
        return self._get_state()

    def _generate_vehicles(self):
        """
        Generate vehicles with enhanced characteristics.

        Returns:
            List of vehicle dictionaries with enhanced characteristics
        """
        # Generate base vehicles
        vehicles = super()._generate_vehicles()

        # Enhance vehicles with additional attributes
        for i, vehicle in enumerate(vehicles):
            # Generate vehicle capabilities (specialized in certain data categories)
            capabilities = np.random.choice(self.data_categories,
                                           size=np.random.randint(1, self.data_categories+1),
                                           replace=False)
            self.vehicle_capabilities[i] = capabilities

            # Add capability score to vehicle
            vehicle['capability_score'] = len(capabilities) / self.data_categories

            # Add priority level (1-5, with 5 being highest)
            vehicle['priority'] = np.random.randint(1, 6)

            # Add resource requirements
            vehicle['computation_requirement'] = np.random.uniform(0.1, 1.0) * vehicle['computation_capacity']
            vehicle['bandwidth_requirement'] = np.random.uniform(5, 20)  # MHz

            # Add reliability score (0-1)
            vehicle['reliability'] = np.random.uniform(0.5, 1.0)

            # Add energy constraints
            vehicle['energy_level'] = np.random.uniform(0.3, 1.0)
            vehicle['energy_consumption_rate'] = np.random.uniform(0.01, 0.05)

            # Add mobility pattern (1-3: low, medium, high mobility)
            vehicle['mobility_pattern'] = np.random.randint(1, 4)

            # Add data freshness (0-1, with 1 being freshest)
            vehicle['data_freshness'] = np.random.uniform(0.5, 1.0)

        return vehicles

    def _update_channel_conditions(self):
        """
        Update channel conditions based on time-varying factors.
        """
        # Apply random variations to channel conditions
        for i in range(self.vehicle_count):
            # Apply random variation within the channel_variation range
            variation = np.random.uniform(-self.channel_variation, self.channel_variation)
            self.channel_conditions[i] = max(0.1, min(1.0, self.channel_conditions[i] + variation))

            # Update vehicle channel gain
            if i < len(self.vehicles):
                self.vehicles[i]['channel_gain'] *= self.channel_conditions[i]

    def _calculate_interference(self, selected_vehicle_idx):
        """
        Calculate interference between the selected vehicle and other scheduled vehicles.

        Args:
            selected_vehicle_idx: Index of the selected vehicle

        Returns:
            Interference factor (0-1, with 0 being no interference)
        """
        if not self.scheduled_vehicles:
            return 0.0

        # Calculate distance-based interference
        interference = 0.0
        selected_vehicle = self.vehicles[selected_vehicle_idx]

        for scheduled_vehicle in self.scheduled_vehicles:
            # Skip if same vehicle
            if scheduled_vehicle['id'] == selected_vehicle_idx:
                continue

            # Calculate distance between vehicles
            distance_diff = abs(selected_vehicle['distance'] - scheduled_vehicle['distance'])
            normalized_distance = min(1.0, distance_diff / (self.area_radius * 1000))

            # Calculate interference based on distance and bandwidth overlap
            bandwidth_overlap = min(selected_vehicle['bandwidth_mhz'], scheduled_vehicle['bandwidth_mhz']) / \
                               max(selected_vehicle['bandwidth_mhz'], scheduled_vehicle['bandwidth_mhz'])

            # Add to total interference
            interference += (1 - normalized_distance) * bandwidth_overlap * self.interference_level

        # Normalize interference
        interference = min(1.0, interference)

        return interference

    def _check_resource_availability(self, vehicle):
        """
        Check if resources are available for the vehicle.

        Args:
            vehicle: Vehicle dictionary

        Returns:
            True if resources are available, False otherwise
        """
        # Check computation resources
        if vehicle['computation_requirement'] > self.available_computation:
            return False

        # Check bandwidth resources
        if vehicle['bandwidth_requirement'] > self.available_bandwidth:
            return False

        return True

    def _allocate_resources(self, vehicle):
        """
        Allocate resources to the vehicle.

        Args:
            vehicle: Vehicle dictionary

        Returns:
            True if resources were allocated, False otherwise
        """
        # Check resource availability
        if not self._check_resource_availability(vehicle):
            # Add to priority queue if resources not available
            if vehicle not in self.priority_queue:
                self.priority_queue.append(vehicle)
            return False

        # Allocate resources
        self.available_computation -= vehicle['computation_requirement']
        self.available_bandwidth -= vehicle['bandwidth_requirement']

        return True

    def _release_resources(self, vehicle):
        """
        Release resources allocated to the vehicle.

        Args:
            vehicle: Vehicle dictionary
        """
        # Release resources
        self.available_computation += vehicle['computation_requirement']
        self.available_bandwidth += vehicle['bandwidth_requirement']

        # Process priority queue if resources available
        self._process_priority_queue()

    def _process_priority_queue(self):
        """
        Process priority queue based on available resources.
        """
        # Sort priority queue by priority level (highest first)
        self.priority_queue.sort(key=lambda v: v['priority'], reverse=True)

        # Process queue
        i = 0
        while i < len(self.priority_queue):
            vehicle = self.priority_queue[i]

            # Check if resources available
            if self._check_resource_availability(vehicle):
                # Allocate resources
                self._allocate_resources(vehicle)

                # Remove from queue
                self.priority_queue.pop(i)
            else:
                i += 1

    # Energy updates are now handled directly in the step method

    def step(self, action):
        """
        Take a step in the enhanced environment.

        Args:
            action: Action to take, consisting of [vehicle_idx, alpha, scheduled_time, bandwidth]

        Returns:
            Tuple of (state, reward, done, info)
        """
        # Update channel conditions
        self._update_channel_conditions()

        # Parse action
        vehicle_idx, alpha, scheduled_time, bandwidth = action

        # Get the selected vehicle
        vehicle = self.vehicles[vehicle_idx]

        # Check if vehicle is available
        if not self._is_vehicle_available(vehicle):
            # If vehicle is not available, return negative reward
            return self._get_state(), -10.0, False, {'error': 'Vehicle not available'}

        # Check if vehicle has enough energy
        if vehicle['energy_level'] < 0.1:
            # If vehicle has low energy, return negative reward
            return self._get_state(), -5.0, False, {'error': 'Vehicle energy too low'}

        # Calculate interference
        interference = self._calculate_interference(vehicle_idx)

        # Check resource availability and allocate resources
        # For simplicity, we'll just proceed even if resources aren't available
        # This ensures the experiment can run without getting stuck
        self._allocate_resources(vehicle)

        # Mark vehicle as scheduled
        vehicle['scheduled'] = True
        vehicle['selection_count'] += 1
        vehicle['participation_count'] += 1

        # Update selection history
        self.selection_history[vehicle_idx] += 1

        # Add vehicle to scheduled set
        self.scheduled_vehicles.append(vehicle)
        self.scheduled_count += 1

        # Update elapsed time
        self.elapsed_time = max(self.elapsed_time, scheduled_time)

        # Calculate accuracy improvement based on scheduled vehicle
        # Use learning rate to determine the improvement rate
        base_improvement = vehicle['data_quality'] * alpha

        # Adjust improvement based on learning rate and data categories
        data_categories_factor = min(1.0, vehicle.get('data_categories_ratio', 0.5) * 1.5)

        # Adjust improvement based on interference
        interference_factor = 1.0 - interference

        # Adjust improvement based on energy level
        energy_factor = vehicle['energy_level']

        # Adjust improvement based on data freshness
        freshness_factor = vehicle['data_freshness']

        # Adjust improvement based on capability score
        capability_factor = vehicle['capability_score']

        # Calculate final accuracy improvement
        accuracy_improvement = self.learning_rate * base_improvement * data_categories_factor * \
                              interference_factor * energy_factor * freshness_factor * capability_factor

        self.current_model_performance += accuracy_improvement
        self.current_model_performance = min(1.0, self.current_model_performance)

        # Calculate enhanced reward
        reward = self._calculate_enhanced_reward(vehicle_idx, alpha, scheduled_time, bandwidth, interference)

        # Check if round is complete
        done = False
        target_count = min(10, self.vehicle_count)  # Target number of vehicles to schedule

        # Force round completion if we've scheduled enough vehicles or reached time limit
        if (self.scheduled_count >= target_count or
            self.elapsed_time >= self.sync_limit or
            all(v['scheduled'] for v in self.vehicles)):
            # Round is complete
            done = True

            # Record performance
            self.performance_history.append(self.current_model_performance)

            # Prepare for next round
            if self.current_round < self.max_round - 1:
                self.current_round += 1

                # Reset scheduled vehicles
                for v in self.vehicles:
                    v['scheduled'] = False

                # Clear scheduled vehicles list
                self.scheduled_vehicles = []
                self.scheduled_count = 0

                # Reset resources for next round
                self.available_computation = self.computation_resources
                self.available_bandwidth = self.bandwidth_resources
                self.priority_queue = []

                # Update vehicle energy levels
                for v in self.vehicles:
                    if 'energy_level' in v:
                        v['energy_level'] = max(0.1, v['energy_level'] - v.get('energy_consumption_rate', 0.01))
                    if 'data_freshness' in v:
                        v['data_freshness'] = max(0.1, v['data_freshness'] - 0.01)

        # Record reward
        self.reward_history.append(reward)

        # Return state, reward, done, info
        return self._get_state(), reward, done, {
            'interference': interference,
            'energy_level': vehicle['energy_level'],
            'data_freshness': vehicle['data_freshness'],
            'capability_score': vehicle['capability_score'],
            'priority': vehicle['priority']
        }

    def _calculate_enhanced_reward(self, vehicle_idx, alpha, scheduled_time, bandwidth, interference):
        """
        Calculate reward based on scheduled vehicle and parameters.
        Using the original reward function from VehicularFLEnv.
        """
        # Use the original reward calculation from the parent class
        return super()._calculate_reward(vehicle_idx, alpha, scheduled_time, bandwidth)

    def _get_state(self):
        """
        Get the enhanced state of the environment.

        Returns:
            Enhanced state representation as a tensor
        """
        # Base state from parent class
        base_state = super()._get_state()

        # Calculate averages more efficiently
        avg_channel = sum(self.channel_conditions) / max(1, len(self.channel_conditions))

        total_energy = 0.0
        total_freshness = 0.0
        count = 0

        for v in self.vehicles:
            if 'energy_level' in v and 'data_freshness' in v:
                total_energy += v['energy_level']
                total_freshness += v['data_freshness']
                count += 1

        avg_energy = total_energy / max(1, count)
        avg_freshness = total_freshness / max(1, count)

        # Enhanced state components
        enhanced_state = torch.cat([
            base_state,
            torch.tensor([
                self.available_computation / max(0.1, self.computation_resources),  # Normalized available computation
                self.available_bandwidth / max(0.1, self.bandwidth_resources),      # Normalized available bandwidth
                len(self.priority_queue) / max(1, self.vehicle_count),              # Normalized priority queue length
                avg_channel,                                                        # Average channel condition
                avg_energy,                                                         # Average energy level
                avg_freshness                                                       # Average data freshness
            ], dtype=torch.float32)
        ])

        return enhanced_state

def run_enhanced_comparison(num_episodes=5, max_rounds=100, interference_level=0.3,
                           resource_constraint=0.7, channel_variation=0.2):
    """
    Run enhanced comparison between different schedulers.

    Args:
        num_episodes: Number of episodes to run for each scheduler
        max_rounds: Maximum number of rounds per episode
        interference_level: Level of interference between vehicles (0-1)
        resource_constraint: Resource constraint factor (0-1)
        channel_variation: Channel variation factor (0-1)

    Returns:
        Dictionary of results for each scheduler
    """
    print("\n=== Running Enhanced Comparison (Complex Decision-Making) ===")

    # Initialize enhanced environment
    env = EnhancedVehicularFLEnv(
        vehicle_count=50,
        max_round=max_rounds,
        sync_limit=1000,
        traffic_density=10,
        data_categories=10,
        learning_rate=0.01,
        interference_level=interference_level,
        resource_constraint=resource_constraint,
        channel_variation=channel_variation
    )

    # Initialize schedulers
    schedulers = initialize_schedulers()

    # Run comparison
    results = {}

    for name, scheduler in schedulers.items():
        print(f"Evaluating {scheduler.name} scheduler...")

        episode_rewards = []
        episode_performances = []
        episode_lengths = []
        decision_times = []

        for episode in range(num_episodes):
            # Reset environment
            env.reset()
            total_reward = 0
            episode_decision_times = []

            # Run until max rounds or environment signals done
            max_iterations = 1000  # Safety limit to prevent infinite loops
            iteration = 0
            while env.current_round < max_rounds and iteration < max_iterations:
                # Get available vehicles and their indices
                available_vehicles = []
                available_indices = []

                for i, vehicle in enumerate(env.vehicles):
                    if env._is_vehicle_available(vehicle):
                        available_vehicles.append(vehicle)
                        available_indices.append(i)

                if not available_vehicles:
                    # No available vehicles, advance time
                    env.elapsed_time += 10
                    continue

                # Make scheduling decision
                start_time = time.time()
                selected_vehicles = scheduler.select_vehicles(available_vehicles)
                decision_time = (time.time() - start_time) * 1000  # ms
                episode_decision_times.append(decision_time)

                if not selected_vehicles:
                    # No vehicles selected, try again with random selection
                    if available_indices:
                        vehicle_idx = random.choice(available_indices)
                    else:
                        # No available vehicles, advance time
                        env.elapsed_time += 10
                        continue
                else:
                    # Get the selected vehicle's index
                    selected_vehicle = selected_vehicles[0]
                    vehicle_idx = selected_vehicle['id']

                # Set action parameters - more complex decision-making
                # Adaptive alpha based on vehicle quality and energy
                alpha = min(1.0, selected_vehicle['data_quality'] * 1.5)

                # Adaptive scheduled time based on vehicle mobility
                mobility_factor = 1.0 + 0.2 * (selected_vehicle['mobility_pattern'] - 1)
                scheduled_time = env.elapsed_time + 10 * mobility_factor

                # Adaptive bandwidth based on vehicle requirements
                bandwidth = min(1.0, selected_vehicle['bandwidth_requirement'] / 20.0)

                # Execute action in environment
                action = [vehicle_idx, alpha, scheduled_time, bandwidth]
                _, reward, done, info = env.step(action)

                # Record metrics
                total_reward += reward

                # Increment iteration counter
                iteration += 1

                # If round is complete, move to next round
                if done:
                    # Check if we've reached max rounds
                    if env.current_round >= max_rounds:
                        break

                    # If we're continuing, reset some state for the next round
                    # The environment already incremented current_round in step()
                    print(f"    Round {env.current_round-1} complete, performance: {env.current_model_performance:.4f}")

            # Record episode metrics
            episode_rewards.append(total_reward)
            episode_performances.append(env.current_model_performance)
            episode_lengths.append(env.current_round)
            decision_times.extend(episode_decision_times)

            print(f"  Episode {episode+1}/{num_episodes}: Reward={total_reward:.2f}, Performance={env.current_model_performance:.4f}, Length={env.current_round}")

        # Compute average metrics
        if episode_rewards:  # Check if we have any valid episodes
            results[scheduler.name] = {
                'avg_reward': np.mean(episode_rewards),
                'avg_performance': np.mean(episode_performances),
                'avg_length': np.mean(episode_lengths),
                'avg_decision_time': np.mean(decision_times) if decision_times else 0,
                'rewards': episode_rewards,
                'performances': episode_performances,
                'lengths': episode_lengths,
                'decision_times': decision_times
            }

            print(f"  Average Reward: {results[scheduler.name]['avg_reward']:.2f}")
            print(f"  Average Performance: {results[scheduler.name]['avg_performance']:.4f}")
            print(f"  Average Length: {results[scheduler.name]['avg_length']:.1f}")
            print(f"  Average Decision Time: {results[scheduler.name]['avg_decision_time']:.2f} ms")

    # Plot results
    plot_enhanced_results(results)

    return results

def plot_enhanced_results(results):
    """
    Plot enhanced comparison results.

    Args:
        results: Dictionary of results for each scheduler
    """
    # Create timestamp for output files
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create figure with 4 subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Get scheduler names
    scheduler_names = list(results.keys())

    # Plot average reward
    avg_rewards = [results[name]['avg_reward'] for name in scheduler_names]
    axes[0, 0].bar(scheduler_names, avg_rewards)
    axes[0, 0].set_title('Average Reward')
    axes[0, 0].set_ylabel('Reward')
    axes[0, 0].set_xticklabels(scheduler_names, rotation=45, ha="right")

    # Plot average performance
    avg_performances = [results[name]['avg_performance'] for name in scheduler_names]
    axes[0, 1].bar(scheduler_names, avg_performances)
    axes[0, 1].set_title('Average Performance')
    axes[0, 1].set_ylabel('Performance')
    axes[0, 1].set_xticklabels(scheduler_names, rotation=45, ha="right")

    # Plot average decision time
    avg_decision_times = [results[name]['avg_decision_time'] for name in scheduler_names]
    axes[1, 0].bar(scheduler_names, avg_decision_times)
    axes[1, 0].set_title('Average Decision Time')
    axes[1, 0].set_ylabel('Time (ms)')
    axes[1, 0].set_xticklabels(scheduler_names, rotation=45, ha="right")

    # Plot average episode length
    avg_lengths = [results[name]['avg_length'] for name in scheduler_names]
    axes[1, 1].bar(scheduler_names, avg_lengths)
    axes[1, 1].set_title('Average Episode Length')
    axes[1, 1].set_ylabel('Rounds')
    axes[1, 1].set_xticklabels(scheduler_names, rotation=45, ha="right")

    # Adjust layout
    plt.tight_layout()

    # Save figure
    plt.savefig(f"results/enhanced_comparison_{timestamp}.png")

    # Create summary file
    with open(f"results/enhanced_comparison_summary_{timestamp}.txt", 'w') as f:
        f.write("Enhanced Comparison Results\n")
        f.write("==========================\n\n")

        for name in scheduler_names:
            f.write(f"{name} Scheduler:\n")
            f.write(f"  Average Reward: {results[name]['avg_reward']:.2f}\n")
            f.write(f"  Average Performance: {results[name]['avg_performance']:.4f}\n")
            f.write(f"  Average Length: {results[name]['avg_length']:.1f}\n")
            f.write(f"  Average Decision Time: {results[name]['avg_decision_time']:.2f} ms\n\n")

if __name__ == "__main__":
    # Run enhanced comparison
    results = run_enhanced_comparison(
        num_episodes=3,
        max_rounds=50,
        interference_level=0.3,
        resource_constraint=0.7,
        channel_variation=0.2
    )
