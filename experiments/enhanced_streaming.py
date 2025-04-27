"""
Enhanced streaming scenario for MR-VFL schedulers.

This experiment implements a more complex streaming scenario with bursty arrivals,
varying data quality, and time-sensitive scheduling requirements.
"""

import time
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
from datetime import datetime
from environments.streaming_env import StreamingVehicularFLEnv
from schedulers import initialize_schedulers

class EnhancedStreamingEnv(StreamingVehicularFLEnv):
    """
    Enhanced Streaming Vehicular Federated Learning Environment.

    This environment extends the streaming environment with:
    1. Bursty vehicle arrivals
    2. Time-varying data quality
    3. Deadline-sensitive scheduling
    4. Bandwidth contention
    5. Varying computation requirements
    6. Data staleness effects
    """

    def __init__(self, vehicle_count=100, max_round=50, sync_limit=1000, traffic_density=20,
                 burst_intensity=0.7, quality_variation=0.3, deadline_strictness=0.8):
        """
        Initialize the enhanced streaming environment.

        Args:
            vehicle_count: Number of vehicles in the environment
            max_round: Maximum number of rounds for the simulation
            sync_limit: Synchronization time limit in seconds
            traffic_density: Traffic density (vehicles per km²)
            burst_intensity: Intensity of arrival bursts (0-1)
            quality_variation: Variation in data quality over time (0-1)
            deadline_strictness: Strictness of deadlines (0-1)
        """
        # Call parent constructor
        super().__init__(
            vehicle_count=vehicle_count,
            max_round=max_round,
            sync_limit=sync_limit,
            traffic_density=traffic_density
        )

        # Enhanced parameters
        self.burst_intensity = burst_intensity
        self.quality_variation = quality_variation
        self.deadline_strictness = deadline_strictness

        # Burst parameters
        self.burst_probability = 0.2  # Probability of a burst occurring
        self.burst_size = int(vehicle_count * burst_intensity)  # Number of vehicles in a burst
        self.burst_duration = 50  # Duration of a burst in seconds
        self.next_burst_time = np.random.uniform(0, sync_limit / 4)  # Time of next burst

        # Deadline parameters
        self.deadlines = {}  # Vehicle deadlines

        # Bandwidth contention
        self.bandwidth_contention = np.zeros(sync_limit // 10)  # Contention per time slot

        # Data staleness
        self.data_staleness = {}  # Vehicle data staleness

        # Generate vehicles with enhanced streaming characteristics
        self.vehicles = self._generate_vehicles()

    def reset(self):
        """
        Reset the enhanced streaming environment.

        Returns:
            Initial state representation
        """
        # Reset base environment state
        super().reset()

        # Reset enhanced parameters
        self.next_burst_time = np.random.uniform(0, self.sync_limit / 4)
        self.deadlines = {}
        self.bandwidth_contention = np.zeros(self.sync_limit // 10)
        self.data_staleness = {}

        # Generate vehicles with enhanced streaming characteristics
        self.vehicles = self._generate_vehicles()

        # Return state
        return self._get_state()

    def _generate_vehicles(self):
        """
        Generate vehicles with enhanced streaming characteristics.

        Returns:
            List of vehicle dictionaries with enhanced streaming characteristics
        """
        # Generate base vehicles
        vehicles = super()._generate_vehicles()

        # Create burst patterns
        burst_times = []
        current_time = self.next_burst_time

        while current_time < self.sync_limit:
            burst_times.append(current_time)
            current_time += np.random.exponential(self.sync_limit / 5)

        # Assign vehicles to bursts
        vehicles_per_burst = min(self.burst_size, len(vehicles) // len(burst_times))
        remaining_vehicles = len(vehicles) - vehicles_per_burst * len(burst_times)

        vehicle_index = 0

        # Assign vehicles to bursts
        for burst_time in burst_times:
            burst_count = vehicles_per_burst
            if remaining_vehicles > 0:
                burst_count += 1
                remaining_vehicles -= 1

            # Assign arrival times within the burst
            for i in range(burst_count):
                if vehicle_index < len(vehicles):
                    # Set arrival time within the burst
                    vehicles[vehicle_index]['arrival_time'] = burst_time + np.random.uniform(0, self.burst_duration)

                    # Set deadline based on arrival time and sojourn time
                    deadline_factor = np.random.uniform(0.3, 0.7)  # Percentage of sojourn time as deadline
                    deadline = vehicles[vehicle_index]['arrival_time'] + vehicles[vehicle_index]['sojourn_time'] * deadline_factor
                    self.deadlines[vehicle_index] = deadline

                    # Set initial data staleness
                    self.data_staleness[vehicle_index] = 0.0

                    # Add deadline to vehicle
                    vehicles[vehicle_index]['deadline'] = deadline

                    # Add data staleness rate
                    vehicles[vehicle_index]['staleness_rate'] = np.random.uniform(0.001, 0.01)

                    # Add time-sensitive flag
                    vehicles[vehicle_index]['time_sensitive'] = np.random.random() < 0.3

                    # Add computation variation
                    vehicles[vehicle_index]['computation_variation'] = np.random.uniform(-0.2, 0.2)

                    # Add bandwidth demand
                    vehicles[vehicle_index]['bandwidth_demand'] = np.random.uniform(0.1, 1.0)

                    vehicle_index += 1

        # Update departure times
        for vehicle in vehicles:
            vehicle['departure_time'] = vehicle['arrival_time'] + vehicle['sojourn_time']

        return vehicles

    def _update_data_quality(self):
        """
        Update data quality based on time-varying factors.
        """
        for i, vehicle in enumerate(self.vehicles):
            # Update data staleness
            if i in self.data_staleness:
                self.data_staleness[i] += vehicle['staleness_rate']

                # Update data quality based on staleness
                staleness_factor = max(0.0, 1.0 - self.data_staleness[i])

                # Apply random variation within the quality_variation range
                variation = np.random.uniform(-self.quality_variation, self.quality_variation)

                # Update data quality
                vehicle['data_quality'] = max(0.1, min(0.9, vehicle['data_quality'] * staleness_factor + variation))

    def _update_bandwidth_contention(self):
        """
        Update bandwidth contention based on current time.
        """
        # Get current time slot
        time_slot = int(self.elapsed_time / 10)

        if time_slot < len(self.bandwidth_contention):
            # Increase contention for current time slot
            self.bandwidth_contention[time_slot] += 0.1

            # Apply decay to other time slots
            for i in range(len(self.bandwidth_contention)):
                if i != time_slot:
                    self.bandwidth_contention[i] = max(0.0, self.bandwidth_contention[i] - 0.01)

    def _check_deadline(self, vehicle_idx):
        """
        Check if vehicle has passed its deadline.

        Args:
            vehicle_idx: Index of the vehicle

        Returns:
            True if deadline has passed, False otherwise
        """
        if vehicle_idx in self.deadlines:
            return self.elapsed_time > self.deadlines[vehicle_idx]

        return False

    def _calculate_deadline_penalty(self, vehicle_idx):
        """
        Calculate penalty for approaching deadline.

        Args:
            vehicle_idx: Index of the vehicle

        Returns:
            Deadline penalty factor (0-1)
        """
        if vehicle_idx in self.deadlines:
            deadline = self.deadlines[vehicle_idx]
            time_to_deadline = max(0, deadline - self.elapsed_time)

            # Normalize time to deadline
            normalized_time = time_to_deadline / max(1.0, self.vehicles[vehicle_idx]['sojourn_time'])

            # Calculate penalty based on deadline strictness
            penalty = (1.0 - normalized_time) * self.deadline_strictness

            return min(1.0, max(0.0, penalty))

        return 0.0

    def step(self, action):
        """
        Take a step in the enhanced streaming environment.

        Args:
            action: Action to take, consisting of [vehicle_idx, alpha, scheduled_time, bandwidth]

        Returns:
            Tuple of (state, reward, done, info)
        """
        # Update data quality
        self._update_data_quality()

        # Update bandwidth contention
        self._update_bandwidth_contention()

        # Parse action
        vehicle_idx, alpha, scheduled_time, bandwidth = action

        # Get the selected vehicle
        vehicle = self.vehicles[vehicle_idx]

        # Check if vehicle is available
        if not self._is_vehicle_available(vehicle):
            # If vehicle is not available, return negative reward
            return self._get_state(), -10.0, False, {'error': 'Vehicle not available'}

        # Check if vehicle has passed its deadline
        if self._check_deadline(vehicle_idx):
            # If deadline has passed, return negative reward
            return self._get_state(), -15.0, False, {'error': 'Deadline passed'}

        # Calculate deadline penalty
        deadline_penalty = self._calculate_deadline_penalty(vehicle_idx)

        # Calculate bandwidth contention penalty
        time_slot = int(scheduled_time / 10)
        if time_slot < len(self.bandwidth_contention):
            contention_penalty = self.bandwidth_contention[time_slot] * bandwidth
        else:
            contention_penalty = 0.0

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

        # Adjust improvement based on deadline penalty
        deadline_factor = 1.0 - deadline_penalty

        # Adjust improvement based on contention penalty
        contention_factor = 1.0 - contention_penalty

        # Adjust improvement based on data staleness
        staleness_factor = 1.0 - min(1.0, self.data_staleness.get(vehicle_idx, 0.0))

        # Calculate final accuracy improvement
        accuracy_improvement = self.learning_rate * base_improvement * data_categories_factor * \
                              deadline_factor * contention_factor * staleness_factor

        self.current_model_performance += accuracy_improvement
        self.current_model_performance = min(1.0, self.current_model_performance)

        # Calculate enhanced reward
        reward = self._calculate_enhanced_reward(vehicle_idx, alpha, scheduled_time, bandwidth,
                                               deadline_penalty, contention_penalty)

        # Check if round is complete
        done = False
        target_count = min(10, self.vehicle_count)  # Target number of vehicles to schedule

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

                self.scheduled_vehicles = []
                self.scheduled_count = 0

                # Keep elapsed time and performance for next round

        # Record reward
        self.reward_history.append(reward)

        # Return state, reward, done, info
        return self._get_state(), reward, done, {
            'deadline_penalty': deadline_penalty,
            'contention_penalty': contention_penalty,
            'data_staleness': self.data_staleness.get(vehicle_idx, 0.0)
        }

    def _calculate_enhanced_reward(self, vehicle_idx, alpha, scheduled_time, bandwidth,
                                 deadline_penalty, contention_penalty):
        """
        Calculate reward based on scheduled vehicle and parameters.
        Using the original reward function from VehicularFLEnv.
        """
        # Use the original reward calculation from the parent class
        return super()._calculate_reward(vehicle_idx, alpha, scheduled_time, bandwidth)

    def _get_state(self):
        """
        Get the enhanced state of the streaming environment.

        Returns:
            Enhanced state representation as a tensor
        """
        # Base state from parent class
        base_state = super()._get_state()

        # Calculate averages more efficiently
        # Average deadline penalty
        total_deadline_penalty = 0.0
        for i in range(min(10, len(self.vehicles))):  # Limit to 10 vehicles for efficiency
            total_deadline_penalty += self._calculate_deadline_penalty(i)
        avg_deadline_penalty = total_deadline_penalty / max(1, min(10, len(self.vehicles)))

        # Average bandwidth contention
        avg_contention = sum(self.bandwidth_contention) / max(1, len(self.bandwidth_contention))

        # Average data staleness
        total_staleness = 0.0
        count = 0
        for i in range(min(10, len(self.vehicles))):  # Limit to 10 vehicles for efficiency
            total_staleness += self.data_staleness.get(i, 0.0)
            count += 1
        avg_staleness = total_staleness / max(1, count)

        # Enhanced state components
        enhanced_state = torch.cat([
            base_state,
            torch.tensor([
                self.next_burst_time / self.sync_limit,  # Normalized next burst time
                avg_deadline_penalty,                    # Average deadline penalty
                avg_contention,                          # Average bandwidth contention
                avg_staleness                            # Average data staleness
            ], dtype=torch.float32)
        ])

        return enhanced_state

def run_enhanced_streaming_scenario(burst_intensity=0.7, quality_variation=0.3, deadline_strictness=0.8):
    """
    Run enhanced streaming scenario to demonstrate scheduler performance in complex streaming environments.

    Args:
        burst_intensity: Intensity of arrival bursts (0-1)
        quality_variation: Variation in data quality over time (0-1)
        deadline_strictness: Strictness of deadlines (0-1)

    Returns:
        Dictionary of results for each scheduler
    """
    print("\n=== Running Enhanced Streaming Scenario (Complex Streaming) ===")

    # Initialize enhanced streaming environment
    env_streaming = EnhancedStreamingEnv(
        vehicle_count=100,  # Large vehicle pool
        max_round=50,
        sync_limit=1000,
        traffic_density=20,  # Higher traffic density
        burst_intensity=burst_intensity,
        quality_variation=quality_variation,
        deadline_strictness=deadline_strictness
    )

    # Initialize schedulers
    schedulers = initialize_schedulers()

    # Select schedulers to compare
    selected_schedulers = {
        "Mamba": schedulers["Mamba"],
        "Transformer": schedulers["Transformer"],
        "LSTM": schedulers["LSTM"],
        "Greedy-Quality": schedulers["Greedy-Quality"],
        "Random": schedulers["Random"]
    }

    # Run comparison
    results = {}

    for name, scheduler in selected_schedulers.items():
        print(f"  Evaluating {scheduler.name} scheduler in enhanced streaming environment...")

        # Reset environment
        env_streaming.reset()
        total_reward = 0
        decision_times = []

        # Track vehicles scheduled per time window
        time_windows = []
        vehicles_scheduled = []
        current_window = 0
        window_size = 50  # Time window size in seconds
        window_count = 0

        # Track deadline performance
        deadline_misses = 0
        deadline_total = 0

        # Track data staleness
        avg_staleness = []

        # Run until max rounds
        max_iterations = 1000  # Safety limit to prevent infinite loops
        iteration = 0
        while env_streaming.current_round < env_streaming.max_round and iteration < max_iterations:
            # Get available vehicles and their indices
            available_vehicles = []
            available_indices = []

            for i, vehicle in enumerate(env_streaming.vehicles):
                if env_streaming._is_vehicle_available(vehicle):
                    available_vehicles.append(vehicle)
                    available_indices.append(i)

                    # Count deadline if vehicle is available
                    if i in env_streaming.deadlines:
                        deadline_total += 1

            # Check if we need to record window data
            window_idx = int(env_streaming.elapsed_time / window_size)
            if window_idx > current_window:
                # Record previous window
                time_windows.append(current_window * window_size)
                vehicles_scheduled.append(window_count)

                # Reset for new window
                current_window = window_idx
                window_count = 0

            if not available_vehicles:
                # No available vehicles, advance time
                env_streaming.elapsed_time += 10
                continue

            # Make scheduling decision
            start_time = time.time()
            selected_vehicles = scheduler.select_vehicles(available_vehicles)
            decision_time = (time.time() - start_time) * 1000  # ms
            decision_times.append(decision_time)

            if not selected_vehicles:
                # No vehicles selected, try again with random selection
                if available_indices:
                    vehicle_idx = random.choice(available_indices)
                else:
                    # No available vehicles, advance time
                    env_streaming.elapsed_time += 10
                    continue
            else:
                # Get the selected vehicle's index
                selected_vehicle = selected_vehicles[0]
                vehicle_idx = selected_vehicle['id']

                # Increment window count
                window_count += 1

            # Set action parameters - more complex decision-making
            # Adaptive alpha based on data quality and staleness
            staleness = env_streaming.data_staleness.get(vehicle_idx, 0.0)
            alpha = min(1.0, selected_vehicle['data_quality'] * (1.0 - staleness) * 1.5)

            # Adaptive scheduled time based on deadline
            if vehicle_idx in env_streaming.deadlines:
                deadline = env_streaming.deadlines[vehicle_idx]
                time_to_deadline = max(10, deadline - env_streaming.elapsed_time)
                scheduled_time = env_streaming.elapsed_time + min(time_to_deadline / 2, 20)
            else:
                scheduled_time = env_streaming.elapsed_time + 10

            # Adaptive bandwidth based on contention
            time_slot = int(scheduled_time / 10)
            if time_slot < len(env_streaming.bandwidth_contention):
                contention = env_streaming.bandwidth_contention[time_slot]
                bandwidth = max(0.1, min(1.0, 0.8 - contention))
            else:
                bandwidth = 0.5

            # Execute action in environment
            action = [vehicle_idx, alpha, scheduled_time, bandwidth]
            _, reward, done, info = env_streaming.step(action)

            # Record metrics
            total_reward += reward

            # Check if deadline was missed
            if 'deadline_penalty' in info and info['deadline_penalty'] > 0.9:
                deadline_misses += 1

            # Record staleness
            if 'data_staleness' in info:
                avg_staleness.append(info['data_staleness'])

            # Increment iteration counter
            iteration += 1

            # If round is complete, move to next round
            if done:
                # Check if we've reached max rounds
                if env_streaming.current_round >= env_streaming.max_round:
                    break

                # If we're continuing, reset some state for the next round
                print(f"    Round {env_streaming.current_round-1} complete, performance: {env_streaming.current_model_performance:.4f}")

        # Record final window if needed
        if window_count > 0:
            time_windows.append(current_window * window_size)
            vehicles_scheduled.append(window_count)

        # Calculate deadline miss rate
        deadline_miss_rate = deadline_misses / max(1, deadline_total)

        # Calculate average staleness
        avg_staleness_value = np.mean(avg_staleness) if avg_staleness else 0.0

        # Record results
        results[scheduler.name] = {
            'total_reward': total_reward,
            'final_performance': env_streaming.current_model_performance,
            'avg_decision_time': np.mean(decision_times) if decision_times else 0,
            'time_windows': time_windows,
            'vehicles_scheduled': vehicles_scheduled,
            'deadline_miss_rate': deadline_miss_rate,
            'avg_staleness': avg_staleness_value
        }

        print(f"    Total Reward: {total_reward:.2f}")
        print(f"    Final Performance: {env_streaming.current_model_performance:.4f}")
        print(f"    Average Decision Time: {results[scheduler.name]['avg_decision_time']:.2f} ms")
        print(f"    Deadline Miss Rate: {deadline_miss_rate:.2f}")
        print(f"    Average Data Staleness: {avg_staleness_value:.4f}")

    # Plot results
    plot_enhanced_streaming_results(results)

    return results

def plot_enhanced_streaming_results(results):
    """
    Plot enhanced streaming scenario results.

    Args:
        results: Dictionary of results for each scheduler
    """
    # Create timestamp for output files
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create figure with 3 subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Get scheduler names
    scheduler_names = list(results.keys())

    # Plot total reward
    total_rewards = [results[name]['total_reward'] for name in scheduler_names]
    axes[0, 0].bar(scheduler_names, total_rewards)
    axes[0, 0].set_title('Total Reward')
    axes[0, 0].set_ylabel('Reward')
    axes[0, 0].set_xticklabels(scheduler_names, rotation=45, ha="right")

    # Plot final performance
    final_performances = [results[name]['final_performance'] for name in scheduler_names]
    axes[0, 1].bar(scheduler_names, final_performances)
    axes[0, 1].set_title('Final Performance')
    axes[0, 1].set_ylabel('Performance')
    axes[0, 1].set_xticklabels(scheduler_names, rotation=45, ha="right")

    # Plot deadline miss rate
    deadline_miss_rates = [results[name]['deadline_miss_rate'] for name in scheduler_names]
    axes[1, 0].bar(scheduler_names, deadline_miss_rates)
    axes[1, 0].set_title('Deadline Miss Rate')
    axes[1, 0].set_ylabel('Miss Rate')
    axes[1, 0].set_xticklabels(scheduler_names, rotation=45, ha="right")

    # Plot average data staleness
    avg_staleness = [results[name]['avg_staleness'] for name in scheduler_names]
    axes[1, 1].bar(scheduler_names, avg_staleness)
    axes[1, 1].set_title('Average Data Staleness')
    axes[1, 1].set_ylabel('Staleness')
    axes[1, 1].set_xticklabels(scheduler_names, rotation=45, ha="right")

    # Adjust layout
    plt.tight_layout()

    # Save figure
    plt.savefig(f"results/enhanced_streaming_{timestamp}.png")

    # Create summary file
    with open(f"results/enhanced_streaming_summary_{timestamp}.txt", 'w') as f:
        f.write("Enhanced Streaming Scenario Results\n")
        f.write("==================================\n\n")

        for name in scheduler_names:
            f.write(f"{name} Scheduler:\n")
            f.write(f"  Total Reward: {results[name]['total_reward']:.2f}\n")
            f.write(f"  Final Performance: {results[name]['final_performance']:.4f}\n")
            f.write(f"  Average Decision Time: {results[name]['avg_decision_time']:.2f} ms\n")
            f.write(f"  Deadline Miss Rate: {results[name]['deadline_miss_rate']:.2f}\n")
            f.write(f"  Average Data Staleness: {results[name]['avg_staleness']:.4f}\n\n")

if __name__ == "__main__":
    # Run enhanced streaming scenario
    results = run_enhanced_streaming_scenario(
        burst_intensity=0.7,
        quality_variation=0.3,
        deadline_strictness=0.8
    )
