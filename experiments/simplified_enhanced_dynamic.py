"""
Simplified enhanced dynamic scenario for MR-VFL schedulers.

This experiment implements a more complex dynamic scenario
while ensuring stability and performance.
"""

import time
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
from datetime import datetime
from environments.dynamic_env import DynamicVehicularFLEnv
from schedulers import initialize_schedulers

class SimplifiedEnhancedDynamicEnv(DynamicVehicularFLEnv):
    """
    Simplified Enhanced Dynamic Vehicular Federated Learning Environment.

    This environment adds complexity to the dynamic scenario
    while maintaining stability and performance.
    """

    def __init__(self, vehicle_count=50, max_round=40, sync_limit=1000, phase_length=10, traffic_density=10):
        """
        Initialize the simplified enhanced dynamic environment.
        """
        # Initialize phase characteristics first
        self.phase_characteristics = [
            {
                'name': 'Initial Phase',
                'description': 'Balanced vehicle distribution',
                'quality_range': (0.3, 0.7),
                'computation_range': (0.3, 0.7),
                'mobility_level': 'low'
            },
            {
                'name': 'High Quality Phase',
                'description': 'More high-quality vehicles',
                'quality_range': (0.5, 0.9),
                'computation_range': (0.2, 0.6),
                'mobility_level': 'medium'
            },
            {
                'name': 'High Mobility Phase',
                'description': 'Vehicles with high mobility',
                'quality_range': (0.2, 0.8),
                'computation_range': (0.4, 0.8),
                'mobility_level': 'high'
            },
            {
                'name': 'Resource Constrained Phase',
                'description': 'Limited resources available',
                'quality_range': (0.1, 0.9),
                'computation_range': (0.1, 0.5),
                'mobility_level': 'medium'
            }
        ]

        # Call parent constructor
        super().__init__(
            vehicle_count=vehicle_count,
            max_round=max_round,
            sync_limit=sync_limit,
            phase_length=phase_length,
            traffic_density=traffic_density
        )

        # Generate vehicles with enhanced dynamic characteristics
        self.vehicles = self._generate_vehicles()

    def reset(self):
        """
        Reset the simplified enhanced dynamic environment.
        """
        # Reset base environment state
        state = super().reset()

        # Generate vehicles with enhanced dynamic characteristics
        self.vehicles = self._generate_vehicles()

        return state

    def _generate_vehicles(self):
        """
        Generate vehicles with enhanced dynamic characteristics.
        """
        # Get current phase characteristics
        phase_chars = self.phase_characteristics[min(self.phase, len(self.phase_characteristics)-1)]

        # Generate base vehicles
        vehicles = super()._generate_vehicles()

        # Enhance vehicles with additional attributes
        for vehicle in vehicles:
            # Adjust data quality based on phase
            quality_min, quality_max = phase_chars['quality_range']
            vehicle['data_quality'] = np.random.uniform(quality_min, quality_max)

            # Adjust computation capacity based on phase
            comp_min, comp_max = phase_chars['computation_range']
            vehicle['computation_capacity'] = np.random.uniform(comp_min, comp_max)

            # Set mobility level
            if phase_chars['mobility_level'] == 'low':
                vehicle['mobility'] = np.random.uniform(0.1, 0.4)
            elif phase_chars['mobility_level'] == 'medium':
                vehicle['mobility'] = np.random.uniform(0.3, 0.7)
            else:  # high
                vehicle['mobility'] = np.random.uniform(0.6, 0.9)

            # Set specialization type
            vehicle['specialization'] = random.choice(['general', 'image', 'text', 'sensor', 'location'])

            # Set adaptability to phase changes (0-1)
            vehicle['adaptability'] = np.random.uniform(0.2, 0.8)

        return vehicles

    def _calculate_vehicle_effectiveness(self, vehicle):
        """
        Calculate how effective a vehicle is in the current phase.
        """
        # Calculate base effectiveness from data quality and computation capacity
        base_effectiveness = (vehicle['data_quality'] + vehicle['computation_capacity']) / 2

        # Adjust for mobility (different phases have different mobility requirements)
        phase_chars = self.phase_characteristics[min(self.phase, len(self.phase_characteristics)-1)]
        if phase_chars['mobility_level'] == 'low':
            mobility_match = 1.0 - vehicle.get('mobility', 0.5)
        elif phase_chars['mobility_level'] == 'medium':
            mobility_match = 1.0 - abs(0.5 - vehicle.get('mobility', 0.5))
        else:  # high
            mobility_match = vehicle.get('mobility', 0.5)

        # Adjust for adaptability
        adaptability_factor = 0.5 + 0.5 * vehicle.get('adaptability', 0.5)

        # Combine factors
        effectiveness = base_effectiveness * mobility_match * adaptability_factor

        # Ensure within bounds
        effectiveness = min(1.0, max(0.1, effectiveness))

        return effectiveness

    def step(self, action):
        """
        Take a step in the simplified enhanced dynamic environment.
        """
        # Check if we need to change phase
        current_phase = self.current_round // self.phase_length
        if current_phase != self.phase:
            self.phase = current_phase
            print(f"    Environment changed to phase {self.phase + 1}: {self.phase_characteristics[min(self.phase, len(self.phase_characteristics)-1)]['name']}")
            print(f"    Description: {self.phase_characteristics[min(self.phase, len(self.phase_characteristics)-1)]['description']}")

            # Regenerate vehicles for the new phase
            self.vehicles = self._generate_vehicles()

        # Call parent step method to maintain stability
        state, reward, done, info = super().step(action)

        # Add enhanced info
        vehicle_idx = action[0]
        if vehicle_idx < len(self.vehicles):
            vehicle = self.vehicles[vehicle_idx]
            effectiveness = self._calculate_vehicle_effectiveness(vehicle)

            info.update({
                'effectiveness': effectiveness,
                'phase': self.phase,
                'mobility': vehicle.get('mobility', 0.5),
                'specialization': vehicle.get('specialization', 'general'),
                'adaptability': vehicle.get('adaptability', 0.5)
            })

        return state, reward, done, info

def run_simplified_enhanced_dynamic():
    """
    Run simplified enhanced dynamic scenario.
    """
    print("\n=== Running Simplified Enhanced Dynamic Scenario ===")

    # Initialize simplified enhanced dynamic environment
    env_dynamic = SimplifiedEnhancedDynamicEnv(
        vehicle_count=50,
        max_round=40,  # 10 rounds per phase, 4 phases
        sync_limit=1000,
        phase_length=10,
        traffic_density=10
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
    phase_performances = {}

    for name, scheduler in selected_schedulers.items():
        print(f"  Evaluating {scheduler.name} scheduler in enhanced dynamic environment...")

        # Reset environment
        env_dynamic.reset()
        total_reward = 0
        decision_times = []

        # Track performance by phase
        phase_perf = [[] for _ in range(4)]  # 4 phases
        phase_effectiveness = [[] for _ in range(4)]  # Track effectiveness by phase

        # Run until max rounds
        max_iterations = 1000  # Safety limit to prevent infinite loops
        iteration = 0

        while env_dynamic.current_round < env_dynamic.max_round and iteration < max_iterations:
            # Get current phase
            current_phase = env_dynamic.phase

            # Get available vehicles and their indices
            available_vehicles = []
            available_indices = []

            for i, vehicle in enumerate(env_dynamic.vehicles):
                if env_dynamic._is_vehicle_available(vehicle):
                    available_vehicles.append(vehicle)
                    available_indices.append(i)

            if not available_vehicles:
                # No available vehicles, advance time
                env_dynamic.elapsed_time += 10
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
                    env_dynamic.elapsed_time += 10
                    iteration += 1
                    continue
            else:
                # Get the selected vehicle's index
                selected_vehicle = selected_vehicles[0]
                vehicle_idx = selected_vehicle['id']

            # Set action parameters - more complex decision-making
            # Adaptive alpha based on vehicle effectiveness
            effectiveness = env_dynamic._calculate_vehicle_effectiveness(selected_vehicle)
            alpha = min(1.0, selected_vehicle['data_quality'] * effectiveness * 1.5)

            # Adaptive scheduled time based on mobility
            mobility_factor = 1.0 + 0.2 * selected_vehicle.get('mobility', 0.5)
            scheduled_time = env_dynamic.elapsed_time + 10 * mobility_factor

            # Adaptive bandwidth
            bandwidth = 0.5  # Default bandwidth

            # Execute action in environment
            action = [vehicle_idx, alpha, scheduled_time, bandwidth]
            _, reward, done, info = env_dynamic.step(action)

            # Record metrics
            total_reward += reward

            # Record effectiveness for current phase
            if 'effectiveness' in info:
                phase_effectiveness[current_phase].append(info['effectiveness'])

            # Increment iteration counter
            iteration += 1

            # If round is complete, move to next round
            if done:
                # Record performance for this phase
                phase_perf[current_phase].append(env_dynamic.current_model_performance)

                # Check if we've reached max rounds
                if env_dynamic.current_round >= env_dynamic.max_round:
                    break

                # If we're continuing, reset some state for the next round
                print(f"    Round {env_dynamic.current_round-1} complete, phase {current_phase+1}, "
                      f"performance: {env_dynamic.current_model_performance:.4f}")

        # Calculate average effectiveness per phase
        phase_avg_effectiveness = []
        for phase in range(4):
            if phase_effectiveness[phase]:
                phase_avg_effectiveness.append(np.mean(phase_effectiveness[phase]))
            else:
                phase_avg_effectiveness.append(0)

        # Record results
        results[scheduler.name] = {
            'total_reward': total_reward,
            'final_performance': env_dynamic.current_model_performance,
            'avg_decision_time': np.mean(decision_times) if decision_times else 0,
            'phase_performances': phase_perf,
            'phase_effectiveness': phase_avg_effectiveness
        }

        # Calculate average performance per phase
        phase_avg = []
        for phase in range(4):
            if phase_perf[phase]:
                phase_avg.append(np.mean(phase_perf[phase]))
            else:
                phase_avg.append(0)

        results[scheduler.name]['avg_phase_performance'] = phase_avg
        phase_performances[scheduler.name] = phase_perf

        # Calculate adaptability score (how well the scheduler adapts to phase changes)
        # Higher score means better adaptability
        if len(phase_avg) >= 2:
            # Calculate performance improvements between phases
            phase_improvements = []
            for i in range(1, len(phase_avg)):
                if phase_avg[i-1] > 0:
                    improvement = (phase_avg[i] - phase_avg[i-1]) / phase_avg[i-1]
                    phase_improvements.append(improvement)

            # Adaptability score is the average improvement
            if phase_improvements:
                adaptability_score = np.mean(phase_improvements)
            else:
                adaptability_score = 0
        else:
            adaptability_score = 0

        results[scheduler.name]['adaptability_score'] = adaptability_score

        print(f"    Total Reward: {total_reward:.2f}")
        print(f"    Final Performance: {env_dynamic.current_model_performance:.4f}")
        print(f"    Average Decision Time: {results[scheduler.name]['avg_decision_time']:.2f} ms")
        print(f"    Phase Performance: {[f'{p:.4f}' for p in phase_avg]}")
        print(f"    Phase Effectiveness: {[f'{e:.4f}' for e in phase_avg_effectiveness]}")
        print(f"    Adaptability Score: {adaptability_score:.4f}")

    # Plot results
    plot_simplified_enhanced_dynamic_results(results, phase_performances)

    return results, phase_performances

def plot_simplified_enhanced_dynamic_results(results, phase_performances):
    """
    Plot simplified enhanced dynamic scenario results.
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

    # Plot adaptability score
    plt.subplot(2, 2, 3)
    adaptability_scores = [results[name]['adaptability_score'] for name in scheduler_names]
    plt.bar(scheduler_names, adaptability_scores)
    plt.title('Adaptability Score')
    plt.ylabel('Score')
    plt.xticks(rotation=45, ha="right")

    # Plot phase performance
    plt.subplot(2, 2, 4)
    x = np.arange(4)  # 4 phases
    width = 0.15  # width of bars

    for i, name in enumerate(scheduler_names):
        plt.bar(x + i*width, results[name]['avg_phase_performance'], width, label=name)

    plt.title('Performance by Phase')
    plt.ylabel('Performance')
    plt.xlabel('Phase')
    plt.xticks(x + width * (len(scheduler_names) - 1) / 2, ['Phase 1', 'Phase 2', 'Phase 3', 'Phase 4'])
    plt.legend(loc='upper left', fontsize='small')

    # Adjust layout
    plt.tight_layout()

    # Save figure
    plt.savefig(f"results/simplified_enhanced_dynamic_{timestamp}.png")

    # Create summary file
    with open(f"results/simplified_enhanced_dynamic_summary_{timestamp}.txt", 'w') as f:
        f.write("Simplified Enhanced Dynamic Scenario Results\n")
        f.write("=========================================\n\n")

        for name in scheduler_names:
            f.write(f"{name} Scheduler:\n")
            f.write(f"  Total Reward: {results[name]['total_reward']:.2f}\n")
            f.write(f"  Final Performance: {results[name]['final_performance']:.4f}\n")
            f.write(f"  Average Decision Time: {results[name]['avg_decision_time']:.2f} ms\n")
            f.write(f"  Adaptability Score: {results[name]['adaptability_score']:.4f}\n")
            f.write(f"  Phase Performance: {[f'{p:.4f}' for p in results[name]['avg_phase_performance']]}\n")
            f.write(f"  Phase Effectiveness: {[f'{e:.4f}' for e in results[name]['phase_effectiveness']]}\n\n")

    # Create additional plot for phase performance comparison
    plt.figure(figsize=(12, 8))

    for i, name in enumerate(scheduler_names):
        phase_avg = results[name]['avg_phase_performance']
        plt.plot(range(1, 5), phase_avg, marker='o', label=name)

    plt.title('Performance Across Phases')
    plt.xlabel('Phase')
    plt.ylabel('Average Performance')
    plt.xticks(range(1, 5))
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()

    # Save additional figure
    plt.savefig(f"results/simplified_enhanced_dynamic_phases_{timestamp}.png")

if __name__ == "__main__":
    # Run simplified enhanced dynamic scenario
    results, phase_performances = run_simplified_enhanced_dynamic()
