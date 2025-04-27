"""
Simplified enhanced streaming scenario for MR-VFL schedulers.

This experiment implements a more complex streaming scenario
while ensuring stability and performance.
"""

import time
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
from datetime import datetime
from environments.streaming_env import StreamingVehicularFLEnv
from schedulers import initialize_schedulers

class SimplifiedEnhancedStreamingEnv(StreamingVehicularFLEnv):
    """
    Simplified Enhanced Streaming Vehicular Federated Learning Environment.

    This environment adds complexity to the streaming scenario
    while maintaining stability and performance.
    """

    def __init__(self, vehicle_count=100, max_round=50, sync_limit=1000, traffic_density=20):
        """
        Initialize the simplified enhanced streaming environment.
        """
        # Initialize enhanced parameters first
        self.burst_times = []
        self.current_burst_time = 0
        self.data_staleness = {}

        # Call parent constructor
        super().__init__(
            vehicle_count=vehicle_count,
            max_round=max_round,
            sync_limit=sync_limit,
            traffic_density=traffic_density
        )

        # Generate burst times
        self.burst_times = self._generate_burst_times()

        # Generate vehicles with enhanced streaming characteristics
        self.vehicles = self._generate_vehicles()

    def reset(self):
        """
        Reset the simplified enhanced streaming environment.
        """
        # Reset base environment state
        state = super().reset()

        # Reset enhanced parameters
        self.burst_times = self._generate_burst_times()
        self.current_burst_time = 0
        self.data_staleness = {}

        # Generate vehicles with enhanced streaming characteristics
        self.vehicles = self._generate_vehicles()

        return state

    def _generate_burst_times(self):
        """
        Generate burst times for vehicle arrivals.
        """
        burst_times = []
        current_time = 50  # First burst at 50 seconds

        while current_time < self.sync_limit:
            burst_times.append(current_time)
            current_time += np.random.exponential(self.sync_limit / 5)

        return burst_times

    def _generate_vehicles(self):
        """
        Generate vehicles with enhanced streaming characteristics.
        """
        # Generate base vehicles
        vehicles = super()._generate_vehicles()

        # Enhance vehicles with additional attributes
        for i, vehicle in enumerate(vehicles):
            # Add deadline
            deadline_factor = np.random.uniform(0.3, 0.7)
            deadline = vehicle['arrival_time'] + vehicle['sojourn_time'] * deadline_factor
            vehicle['deadline'] = deadline

            # Add data staleness rate
            vehicle['staleness_rate'] = np.random.uniform(0.001, 0.01)

            # Add time-sensitive flag
            vehicle['time_sensitive'] = np.random.random() < 0.3

            # Add bandwidth demand
            vehicle['bandwidth_demand'] = np.random.uniform(0.1, 1.0)

            # Initialize data staleness
            self.data_staleness[i] = 0.0

        return vehicles

    def _update_data_staleness(self):
        """
        Update data staleness for all vehicles.
        """
        for i, vehicle in enumerate(self.vehicles):
            if i in self.data_staleness:
                self.data_staleness[i] += vehicle.get('staleness_rate', 0.005)
                # Cap staleness at 1.0
                self.data_staleness[i] = min(1.0, self.data_staleness[i])

    def _check_deadline(self, vehicle_idx):
        """
        Check if vehicle has passed its deadline.
        """
        if vehicle_idx < len(self.vehicles):
            vehicle = self.vehicles[vehicle_idx]
            if 'deadline' in vehicle:
                return self.elapsed_time > vehicle['deadline']
        return False

    def step(self, action):
        """
        Take a step in the simplified enhanced streaming environment.
        """
        # Update data staleness
        self._update_data_staleness()

        # Check for burst arrivals
        if self.burst_times and self.current_burst_time < len(self.burst_times):
            if self.elapsed_time >= self.burst_times[self.current_burst_time]:
                print(f"    Burst arrival at time {self.elapsed_time:.1f}s")
                self.current_burst_time += 1

        # Call parent step method to maintain stability
        state, reward, done, info = super().step(action)

        # Add enhanced info
        vehicle_idx = action[0]
        if vehicle_idx < len(self.vehicles):
            vehicle = self.vehicles[vehicle_idx]
            deadline_passed = self._check_deadline(vehicle_idx)

            info.update({
                'deadline_passed': deadline_passed,
                'data_staleness': self.data_staleness.get(vehicle_idx, 0.0),
                'time_sensitive': vehicle.get('time_sensitive', False),
                'bandwidth_demand': vehicle.get('bandwidth_demand', 0.5)
            })

        return state, reward, done, info

def run_simplified_enhanced_streaming(num_episodes=1):
    """
    Run simplified enhanced streaming scenario.
    """
    print("\n=== Running Simplified Enhanced Streaming Scenario ===")

    # Initialize simplified enhanced streaming environment
    env_streaming = SimplifiedEnhancedStreamingEnv(
        vehicle_count=100,  # Large vehicle pool
        max_round=50,
        sync_limit=1000,
        traffic_density=20  # Higher traffic density
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
                    if 'deadline' in vehicle:
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
                iteration += 1
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
                    iteration += 1
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
            if 'deadline' in selected_vehicle:
                deadline = selected_vehicle['deadline']
                time_to_deadline = max(10, deadline - env_streaming.elapsed_time)
                scheduled_time = env_streaming.elapsed_time + min(time_to_deadline / 2, 20)
            else:
                scheduled_time = env_streaming.elapsed_time + 10

            # Adaptive bandwidth
            bandwidth = 0.5  # Default bandwidth

            # Execute action in environment
            action = [vehicle_idx, alpha, scheduled_time, bandwidth]
            _, reward, done, info = env_streaming.step(action)

            # Record metrics
            total_reward += reward

            # Check if deadline was missed
            if 'deadline_passed' in info and info['deadline_passed']:
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
    plot_simplified_enhanced_streaming_results(results)

    return results

def plot_simplified_enhanced_streaming_results(results):
    """
    Plot simplified enhanced streaming scenario results.
    """
    # Create timestamp for output files
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create figure with 4 subplots
    plt.figure(figsize=(15, 10))
    plt.subplot(2, 2, 1)

    # Get scheduler names
    scheduler_names = list(results.keys())

    # Plot total reward
    total_rewards = [results[name]['total_reward'] for name in scheduler_names]
    plt.bar(scheduler_names, total_rewards)
    plt.title('Total Reward')
    plt.ylabel('Reward')
    plt.xticks(rotation=45, ha="right")

    # Plot final performance
    plt.subplot(2, 2, 2)
    final_performances = [results[name]['final_performance'] for name in scheduler_names]
    plt.bar(scheduler_names, final_performances)
    plt.title('Final Performance')
    plt.ylabel('Performance')
    plt.xticks(rotation=45, ha="right")

    # Plot deadline miss rate
    plt.subplot(2, 2, 3)
    deadline_miss_rates = [results[name]['deadline_miss_rate'] for name in scheduler_names]
    plt.bar(scheduler_names, deadline_miss_rates)
    plt.title('Deadline Miss Rate')
    plt.ylabel('Miss Rate')
    plt.xticks(rotation=45, ha="right")

    # Plot average data staleness
    plt.subplot(2, 2, 4)
    avg_staleness = [results[name]['avg_staleness'] for name in scheduler_names]
    plt.bar(scheduler_names, avg_staleness)
    plt.title('Average Data Staleness')
    plt.ylabel('Staleness')
    plt.xticks(rotation=45, ha="right")

    # Adjust layout
    plt.tight_layout()

    # Save figure
    plt.savefig(f"results/simplified_enhanced_streaming_{timestamp}.png")

    # Create summary file
    with open(f"results/simplified_enhanced_streaming_summary_{timestamp}.txt", 'w') as f:
        f.write("Simplified Enhanced Streaming Scenario Results\n")
        f.write("===========================================\n\n")

        for name in scheduler_names:
            f.write(f"{name} Scheduler:\n")
            f.write(f"  Total Reward: {results[name]['total_reward']:.2f}\n")
            f.write(f"  Final Performance: {results[name]['final_performance']:.4f}\n")
            f.write(f"  Average Decision Time: {results[name]['avg_decision_time']:.2f} ms\n")
            f.write(f"  Deadline Miss Rate: {results[name]['deadline_miss_rate']:.2f}\n")
            f.write(f"  Average Data Staleness: {results[name]['avg_staleness']:.4f}\n\n")

if __name__ == "__main__":
    # Run simplified enhanced streaming scenario
    results = run_simplified_enhanced_streaming(num_episodes=1)
