"""
Simplified enhanced comparison experiment for MR-VFL schedulers.

This experiment implements a more complex decision-making scenario
while ensuring stability and performance.
"""

import time
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
from datetime import datetime
from environments.vehicular_fl_env import VehicularFLEnv
from schedulers import initialize_schedulers

class SimplifiedEnhancedEnv(VehicularFLEnv):
    """
    Simplified Enhanced Vehicular Federated Learning Environment.
    
    This environment adds complexity to the decision-making process
    while maintaining stability and performance.
    """
    
    def __init__(self, vehicle_count=50, max_round=100, sync_limit=1000, traffic_density=10, 
                 data_categories=10, learning_rate=0.01):
        """
        Initialize the simplified enhanced environment.
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
        
        # Generate vehicles with enhanced attributes
        self.vehicles = self._generate_vehicles()
    
    def _generate_vehicles(self):
        """
        Generate vehicles with enhanced attributes.
        """
        # Generate base vehicles
        vehicles = super()._generate_vehicles()
        
        # Enhance vehicles with additional attributes
        for vehicle in vehicles:
            # Add priority level (1-5, with 5 being highest)
            vehicle['priority'] = np.random.randint(1, 6)
            
            # Add reliability score (0-1)
            vehicle['reliability'] = np.random.uniform(0.5, 1.0)
            
            # Add energy level
            vehicle['energy_level'] = np.random.uniform(0.3, 1.0)
            
            # Add mobility pattern (1-3: low, medium, high mobility)
            vehicle['mobility_pattern'] = np.random.randint(1, 4)
            
            # Add data freshness (0-1, with 1 being freshest)
            vehicle['data_freshness'] = np.random.uniform(0.5, 1.0)
            
            # Add specialized capability
            vehicle['specialized_capability'] = random.choice(['image', 'text', 'sensor', 'location', 'general'])
            
            # Add capability efficiency
            vehicle['capability_efficiency'] = np.random.uniform(0.8, 1.5)
        
        return vehicles
    
    def step(self, action):
        """
        Take a step in the simplified enhanced environment.
        """
        # Call parent step method to maintain stability
        state, reward, done, info = super().step(action)
        
        # Add enhanced info
        vehicle_idx = action[0]
        if vehicle_idx < len(self.vehicles):
            vehicle = self.vehicles[vehicle_idx]
            info.update({
                'priority': vehicle.get('priority', 1),
                'reliability': vehicle.get('reliability', 0.5),
                'energy_level': vehicle.get('energy_level', 0.5),
                'mobility_pattern': vehicle.get('mobility_pattern', 1),
                'data_freshness': vehicle.get('data_freshness', 0.5),
                'specialized_capability': vehicle.get('specialized_capability', 'general'),
                'capability_efficiency': vehicle.get('capability_efficiency', 1.0)
            })
        
        return state, reward, done, info

def run_simplified_enhanced_comparison(num_episodes=5, max_rounds=100):
    """
    Run simplified enhanced comparison between different schedulers.
    """
    print("\n=== Running Simplified Enhanced Comparison (Complex Decision-Making) ===")
    
    # Initialize simplified enhanced environment
    env = SimplifiedEnhancedEnv(
        vehicle_count=50,
        max_round=max_rounds,
        sync_limit=1000,
        traffic_density=10,
        data_categories=10,
        learning_rate=0.01
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
                    iteration += 1
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
                        iteration += 1
                        continue
                else:
                    # Get the selected vehicle's index
                    selected_vehicle = selected_vehicles[0]
                    vehicle_idx = selected_vehicle['id']
                
                # Set action parameters - more complex decision-making
                # Adaptive alpha based on vehicle quality
                alpha = min(1.0, selected_vehicle['data_quality'] * 1.5)
                
                # Adaptive scheduled time based on vehicle mobility
                mobility_factor = 1.0 + 0.2 * (selected_vehicle.get('mobility_pattern', 1) - 1)
                scheduled_time = env.elapsed_time + 10 * mobility_factor
                
                # Adaptive bandwidth based on vehicle requirements
                bandwidth = 0.5  # Default bandwidth
                
                # Execute action in environment
                action = [vehicle_idx, alpha, scheduled_time, bandwidth]
                _, reward, done, _ = env.step(action)
                
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
    plot_simplified_enhanced_results(results)
    
    return results

def plot_simplified_enhanced_results(results):
    """
    Plot simplified enhanced comparison results.
    """
    # Create timestamp for output files
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create figure with 4 subplots
    plt.figure(figsize=(15, 10))
    plt.subplot(2, 2, 1)
    
    # Get scheduler names
    scheduler_names = list(results.keys())
    
    # Plot average reward
    avg_rewards = [results[name]['avg_reward'] for name in scheduler_names]
    plt.bar(scheduler_names, avg_rewards)
    plt.title('Average Reward')
    plt.ylabel('Reward')
    plt.xticks(rotation=45, ha="right")
    
    # Plot average performance
    plt.subplot(2, 2, 2)
    avg_performances = [results[name]['avg_performance'] for name in scheduler_names]
    plt.bar(scheduler_names, avg_performances)
    plt.title('Average Performance')
    plt.ylabel('Performance')
    plt.xticks(rotation=45, ha="right")
    
    # Plot average decision time
    plt.subplot(2, 2, 3)
    avg_decision_times = [results[name]['avg_decision_time'] for name in scheduler_names]
    plt.bar(scheduler_names, avg_decision_times)
    plt.title('Average Decision Time')
    plt.ylabel('Time (ms)')
    plt.xticks(rotation=45, ha="right")
    
    # Plot average episode length
    plt.subplot(2, 2, 4)
    avg_lengths = [results[name]['avg_length'] for name in scheduler_names]
    plt.bar(scheduler_names, avg_lengths)
    plt.title('Average Episode Length')
    plt.ylabel('Rounds')
    plt.xticks(rotation=45, ha="right")
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure
    plt.savefig(f"results/simplified_enhanced_comparison_{timestamp}.png")
    
    # Create summary file
    with open(f"results/simplified_enhanced_comparison_summary_{timestamp}.txt", 'w') as f:
        f.write("Simplified Enhanced Comparison Results\n")
        f.write("====================================\n\n")
        
        for name in scheduler_names:
            f.write(f"{name} Scheduler:\n")
            f.write(f"  Average Reward: {results[name]['avg_reward']:.2f}\n")
            f.write(f"  Average Performance: {results[name]['avg_performance']:.4f}\n")
            f.write(f"  Average Length: {results[name]['avg_length']:.1f}\n")
            f.write(f"  Average Decision Time: {results[name]['avg_decision_time']:.2f} ms\n\n")

if __name__ == "__main__":
    # Run simplified enhanced comparison
    results = run_simplified_enhanced_comparison(
        num_episodes=3,
        max_rounds=50
    )
