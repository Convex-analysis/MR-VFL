"""
Enhanced dynamic scenario for MR-VFL schedulers.

This experiment implements a more complex dynamic scenario with multiple phases,
changing vehicle characteristics, and challenging decision-making requirements.
"""

import time
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
from datetime import datetime
from environments.dynamic_env import DynamicVehicularFLEnv
from schedulers import initialize_schedulers

class EnhancedDynamicEnv(DynamicVehicularFLEnv):
    """
    Enhanced Dynamic Vehicular Federated Learning Environment.

    This environment extends the dynamic environment with:
    1. More complex phase transitions
    2. Heterogeneous vehicle capabilities
    3. Phase-specific challenges
    4. Adaptive resource requirements
    5. Varying model complexity
    6. Specialized vehicle types
    """

    def __init__(self, vehicle_count=50, max_round=40, sync_limit=1000, phase_length=10,
                 traffic_density=10, phase_difficulty=0.7, specialization_level=0.8):
        """
        Initialize the enhanced dynamic environment.

        Args:
            vehicle_count: Number of vehicles in the environment
            max_round: Maximum number of rounds for the simulation
            sync_limit: Synchronization time limit in seconds
            phase_length: Number of rounds per phase
            traffic_density: Traffic density (vehicles per km²)
            phase_difficulty: Difficulty increase per phase (0-1)
            specialization_level: Level of vehicle specialization (0-1)
        """
        # Call parent constructor
        super().__init__(
            vehicle_count=vehicle_count,
            max_round=max_round,
            sync_limit=sync_limit,
            phase_length=phase_length,
            traffic_density=traffic_density
        )

        # Enhanced parameters
        self.phase_difficulty = phase_difficulty
        self.specialization_level = specialization_level

        # Phase characteristics
        self.phase_characteristics = [
            {
                'name': 'Initial Phase',
                'description': 'Balanced vehicle distribution',
                'quality_range': (0.3, 0.7),
                'computation_range': (0.3, 0.7),
                'mobility_level': 'low',
                'specialization_types': ['general'],
                'model_complexity': 1.0
            },
            {
                'name': 'High Quality Phase',
                'description': 'More high-quality vehicles',
                'quality_range': (0.5, 0.9),
                'computation_range': (0.2, 0.6),
                'mobility_level': 'medium',
                'specialization_types': ['image', 'text'],
                'model_complexity': 1.5
            },
            {
                'name': 'High Mobility Phase',
                'description': 'Vehicles with high mobility',
                'quality_range': (0.2, 0.8),
                'computation_range': (0.4, 0.8),
                'mobility_level': 'high',
                'specialization_types': ['sensor', 'location'],
                'model_complexity': 1.2
            },
            {
                'name': 'Resource Constrained Phase',
                'description': 'Limited resources available',
                'quality_range': (0.1, 0.9),
                'computation_range': (0.1, 0.5),
                'mobility_level': 'medium',
                'specialization_types': ['all'],
                'model_complexity': 2.0
            }
        ]

        # Specialization types and their characteristics
        self.specialization_types = {
            'general': {'data_types': ['all'], 'efficiency': 1.0},
            'image': {'data_types': ['image'], 'efficiency': 1.5},
            'text': {'data_types': ['text'], 'efficiency': 1.3},
            'sensor': {'data_types': ['sensor'], 'efficiency': 1.4},
            'location': {'data_types': ['location'], 'efficiency': 1.2},
            'all': {'data_types': ['image', 'text', 'sensor', 'location'], 'efficiency': 0.8}
        }

        # Data types in the current phase
        self.current_data_types = ['all']

        # Model complexity (affects computation requirements)
        self.model_complexity = 1.0

        # Resource constraints
        self.resource_constraints = {
            'computation': 1.0,
            'bandwidth': 1.0,
            'energy': 1.0,
            'time': 1.0
        }

        # Generate vehicles with enhanced dynamic characteristics
        self.vehicles = self._generate_vehicles()

    def reset(self):
        """
        Reset the enhanced dynamic environment.

        Returns:
            Initial state representation
        """
        # Reset base environment state
        super().reset()

        # Reset enhanced parameters
        self.current_data_types = ['all']
        self.model_complexity = 1.0
        self.resource_constraints = {
            'computation': 1.0,
            'bandwidth': 1.0,
            'energy': 1.0,
            'time': 1.0
        }

        # Generate vehicles with enhanced dynamic characteristics
        self.vehicles = self._generate_vehicles()

        # Return state
        return self._get_state()

    def _generate_vehicles(self):
        """
        Generate vehicles with enhanced dynamic characteristics.

        Returns:
            List of vehicle dictionaries with enhanced dynamic characteristics
        """
        # Get current phase characteristics
        phase_chars = self.phase_characteristics[self.phase]

        # Generate base vehicles
        vehicles = super()._generate_vehicles()

        # Enhance vehicles with additional attributes
        for i, vehicle in enumerate(vehicles):
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
            if 'all' in phase_chars['specialization_types']:
                spec_type = random.choice(list(self.specialization_types.keys()))
            else:
                spec_type = random.choice(phase_chars['specialization_types'])

            vehicle['specialization'] = spec_type
            vehicle['specialization_efficiency'] = self.specialization_types[spec_type]['efficiency']

            # Set data types the vehicle can handle
            vehicle['data_types'] = self.specialization_types[spec_type]['data_types']

            # Set resource requirements based on model complexity
            vehicle['computation_requirement'] = vehicle['computation_capacity'] * self.model_complexity
            vehicle['bandwidth_requirement'] = np.random.uniform(5, 20) * self.model_complexity

            # Set energy constraints
            vehicle['energy_level'] = np.random.uniform(0.3, 1.0)
            vehicle['energy_consumption_rate'] = np.random.uniform(0.01, 0.05) * self.model_complexity

            # Set reliability
            vehicle['reliability'] = np.random.uniform(0.5, 1.0)

            # Set adaptability to phase changes (0-1)
            vehicle['adaptability'] = np.random.uniform(0.2, 0.8)

        return vehicles

    def _update_phase_characteristics(self):
        """
        Update environment based on current phase.
        """
        # Get current phase characteristics
        phase_chars = self.phase_characteristics[self.phase]

        # Update model complexity
        self.model_complexity = phase_chars['model_complexity']

        # Update current data types
        if 'all' in phase_chars['specialization_types']:
            self.current_data_types = ['image', 'text', 'sensor', 'location']
        else:
            self.current_data_types = []
            for spec_type in phase_chars['specialization_types']:
                self.current_data_types.extend(self.specialization_types[spec_type]['data_types'])
            self.current_data_types = list(set(self.current_data_types))

        # Update resource constraints based on phase
        if self.phase == 3:  # Resource constrained phase
            self.resource_constraints = {
                'computation': 0.6,
                'bandwidth': 0.7,
                'energy': 0.8,
                'time': 0.5
            }
        else:
            self.resource_constraints = {
                'computation': 1.0,
                'bandwidth': 1.0,
                'energy': 1.0,
                'time': 1.0
            }

    def _calculate_vehicle_effectiveness(self, vehicle):
        """
        Calculate how effective a vehicle is in the current phase.

        Args:
            vehicle: Vehicle dictionary

        Returns:
            Effectiveness score (0-1)
        """
        # Check if vehicle can handle current data types
        data_type_match = False
        for data_type in vehicle['data_types']:
            if data_type == 'all' or data_type in self.current_data_types:
                data_type_match = True
                break

        if not data_type_match:
            return 0.1  # Very low effectiveness if can't handle current data types

        # Calculate base effectiveness from data quality and computation capacity
        base_effectiveness = (vehicle['data_quality'] + vehicle['computation_capacity']) / 2

        # Adjust for specialization efficiency
        specialization_factor = vehicle['specialization_efficiency']

        # Adjust for mobility (different phases have different mobility requirements)
        phase_chars = self.phase_characteristics[self.phase]
        if phase_chars['mobility_level'] == 'low':
            mobility_match = 1.0 - vehicle['mobility']
        elif phase_chars['mobility_level'] == 'medium':
            mobility_match = 1.0 - abs(0.5 - vehicle['mobility'])
        else:  # high
            mobility_match = vehicle['mobility']

        # Adjust for adaptability
        adaptability_factor = 0.5 + 0.5 * vehicle['adaptability']

        # Adjust for reliability
        reliability_factor = vehicle['reliability']

        # Combine factors
        effectiveness = base_effectiveness * specialization_factor * mobility_match * adaptability_factor * reliability_factor

        # Ensure within bounds
        effectiveness = min(1.0, max(0.1, effectiveness))

        return effectiveness

    def step(self, action):
        """
        Take a step in the enhanced dynamic environment.

        Args:
            action: Action to take, consisting of [vehicle_idx, alpha, scheduled_time, bandwidth]

        Returns:
            Tuple of (state, reward, done, info)
        """
        # Check if we need to change phase
        current_phase = self.current_round // self.phase_length
        if current_phase != self.phase:
            self.phase = current_phase
            print(f"    Environment changed to phase {self.phase + 1}: {self.phase_characteristics[self.phase]['name']}")
            print(f"    Description: {self.phase_characteristics[self.phase]['description']}")

            # Update phase characteristics
            self._update_phase_characteristics()

            # Regenerate vehicles for the new phase
            self.vehicles = self._generate_vehicles()

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

        # Calculate vehicle effectiveness in current phase
        effectiveness = self._calculate_vehicle_effectiveness(vehicle)

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

        # Calculate accuracy improvement based on scheduled vehicle and effectiveness
        # Use learning rate to determine the improvement rate
        base_improvement = vehicle['data_quality'] * alpha

        # Adjust improvement based on learning rate and data categories
        data_categories_factor = min(1.0, vehicle.get('data_categories_ratio', 0.5) * 1.5)

        # Adjust improvement based on effectiveness
        effectiveness_factor = effectiveness

        # Adjust improvement based on model complexity
        complexity_factor = 1.0 / max(1.0, self.model_complexity)

        # Calculate final accuracy improvement
        accuracy_improvement = self.learning_rate * base_improvement * data_categories_factor * \
                              effectiveness_factor * complexity_factor

        self.current_model_performance += accuracy_improvement
        self.current_model_performance = min(1.0, self.current_model_performance)

        # Calculate reward using the original reward function
        reward = self._calculate_reward(vehicle_idx, alpha, scheduled_time, bandwidth)

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

                # Check if we need to change phase
                current_phase = self.current_round // self.phase_length
                if current_phase != self.phase:
                    self.phase = current_phase
                    print(f"    Environment changed to phase {self.phase + 1}: {self.phase_characteristics[self.phase]['name']}")
                    print(f"    Description: {self.phase_characteristics[self.phase]['description']}")

                    # Update phase characteristics
                    self._update_phase_characteristics()

                    # Regenerate vehicles for the new phase
                    self.vehicles = self._generate_vehicles()

        # Record reward
        self.reward_history.append(reward)

        # Return state, reward, done, info
        return self._get_state(), reward, done, {
            'effectiveness': effectiveness,
            'phase': self.phase,
            'model_complexity': self.model_complexity,
            'specialization': vehicle['specialization']
        }

    def _get_state(self):
        """
        Get the enhanced state of the dynamic environment.

        Returns:
            Enhanced state representation as a tensor
        """
        # Base state from parent class
        base_state = super()._get_state()

        # Enhanced state components with safety checks
        phase_norm = min(1.0, float(self.phase) / 4)
        complexity_norm = min(1.0, self.model_complexity / 2.0)
        comp_constraint = self.resource_constraints.get('computation', 1.0)
        bw_constraint = self.resource_constraints.get('bandwidth', 1.0)
        data_types_norm = min(1.0, len(self.current_data_types) / 4)

        enhanced_state = torch.cat([
            base_state,
            torch.tensor([
                phase_norm,          # Normalized phase
                complexity_norm,     # Normalized model complexity
                comp_constraint,     # Computation constraint
                bw_constraint,       # Bandwidth constraint
                data_types_norm      # Normalized number of data types
            ], dtype=torch.float32)
        ])

        return enhanced_state

def run_enhanced_dynamic_scenario(phase_difficulty=0.7, specialization_level=0.8):
    """
    Run enhanced dynamic scenario to demonstrate scheduler adaptability.

    Args:
        phase_difficulty: Difficulty increase per phase (0-1)
        specialization_level: Level of vehicle specialization (0-1)

    Returns:
        Dictionary of results for each scheduler and phase performances
    """
    print("\n=== Running Enhanced Dynamic Scenario (Complex Adaptability) ===")

    # Initialize enhanced dynamic environment
    env_dynamic = EnhancedDynamicEnv(
        vehicle_count=50,
        max_round=40,  # 10 rounds per phase, 4 phases
        sync_limit=1000,
        phase_length=10,
        traffic_density=10,
        phase_difficulty=phase_difficulty,
        specialization_level=specialization_level
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

            # Adaptive bandwidth based on requirements and constraints
            bandwidth_factor = env_dynamic.resource_constraints['bandwidth']
            bandwidth = min(1.0, selected_vehicle.get('bandwidth_requirement', 10) / 20.0) * bandwidth_factor

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
    plot_enhanced_dynamic_results(results, phase_performances)

    return results, phase_performances

def plot_enhanced_dynamic_results(results, phase_performances):
    """
    Plot enhanced dynamic scenario results.

    Args:
        results: Dictionary of results for each scheduler
        phase_performances: Dictionary of phase performances for each scheduler
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

    # Plot adaptability score
    adaptability_scores = [results[name]['adaptability_score'] for name in scheduler_names]
    axes[1, 0].bar(scheduler_names, adaptability_scores)
    axes[1, 0].set_title('Adaptability Score')
    axes[1, 0].set_ylabel('Score')
    axes[1, 0].set_xticklabels(scheduler_names, rotation=45, ha="right")

    # Plot phase performance
    x = np.arange(4)  # 4 phases
    width = 0.15  # width of bars

    for i, name in enumerate(scheduler_names):
        axes[1, 1].bar(x + i*width, results[name]['avg_phase_performance'], width, label=name)

    axes[1, 1].set_title('Performance by Phase')
    axes[1, 1].set_ylabel('Performance')
    axes[1, 1].set_xlabel('Phase')
    axes[1, 1].set_xticks(x + width * (len(scheduler_names) - 1) / 2)
    axes[1, 1].set_xticklabels(['Phase 1', 'Phase 2', 'Phase 3', 'Phase 4'])
    axes[1, 1].legend(loc='upper left', fontsize='small')

    # Adjust layout
    plt.tight_layout()

    # Save figure
    plt.savefig(f"results/enhanced_dynamic_{timestamp}.png")

    # Create summary file
    with open(f"results/enhanced_dynamic_summary_{timestamp}.txt", 'w') as f:
        f.write("Enhanced Dynamic Scenario Results\n")
        f.write("================================\n\n")

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
    plt.savefig(f"results/enhanced_dynamic_phases_{timestamp}.png")

if __name__ == "__main__":
    # Run enhanced dynamic scenario
    results, phase_performances = run_enhanced_dynamic_scenario(
        phase_difficulty=0.7,
        specialization_level=0.8
    )
