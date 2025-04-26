"""
Scheduling count scenario experiment for MR-VFL schedulers.
"""

import time
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
import os
import traceback
from datetime import datetime
from ..environments.scheduling_count_env import SchedulingCountEnv
from ..schedulers import initialize_schedulers
from ..utils.visualization import plot_scheduling_count_results

# Create output directory if it doesn't exist
output_dir = os.path.join("MR-VFL", "results")
os.makedirs(output_dir, exist_ok=True)

def run_vehicle_scheduling_count_scenario():
    """
    Run a scenario to count and compare how many vehicles are successfully scheduled by each scheduler.
    
    This scenario focuses on:
    1. Total number of vehicles scheduled
    2. Scheduling efficiency (percentage of available vehicles scheduled)
    3. Quality of scheduled vehicles (average data quality)
    4. Fairness of scheduling (distribution across different vehicle types)
    """
    print("\n=== Running Vehicle Scheduling Count Scenario ===")
    
    # Initialize schedulers
    schedulers = initialize_schedulers()
    
    # Select schedulers for comparison
    selected_schedulers = {
        "Mamba": schedulers["Mamba"],
        "Transformer": schedulers["Transformer"],
        "LSTM": schedulers["LSTM"],
        "Greedy-Quality": schedulers["Greedy-Quality"],
        "Greedy-Compute": schedulers["Greedy-Compute"],
        "Random": schedulers["Random"]
    }
    
    # Initialize environment with specified parameters
    env = SchedulingCountEnv(
        vehicle_count=100,
        max_round=20,
        sync_limit=1000,
        traffic_density=10  # Traffic density of 10 vehicles per km²
    )
    
    # Run comparison
    scheduling_results = {}
    
    for name, scheduler in selected_schedulers.items():
        print(f"\nEvaluating {name} scheduler...")
        
        # Run multiple episodes
        episode_stats = []
        
        for episode in range(3):
            # Reset environment
            env.reset()
            total_reward = 0
            
            # Run until max rounds or environment signals done
            while env.current_round < env.max_round:
                # Get available vehicles and their indices
                available_vehicles = []
                available_indices = []
                
                for i, vehicle in enumerate(env.vehicles):
                    if env._is_vehicle_available(vehicle):
                        available_vehicles.append(vehicle)
                        available_indices.append(i)
                
                # Track available vehicles
                env.available_vehicle_count += len(available_vehicles)
                
                if not available_vehicles:
                    # If no vehicles are available, advance time
                    # Find the next vehicle that will arrive
                    next_arrival = float('inf')
                    for vehicle in env.vehicles:
                        if not vehicle['scheduled'] and vehicle['arrival_time'] > env.elapsed_time:
                            next_arrival = min(next_arrival, vehicle['arrival_time'])
                    
                    if next_arrival < float('inf'):
                        # Advance time to next arrival
                        env.elapsed_time = next_arrival
                        continue
                    else:
                        # No more vehicles will arrive, end episode
                        break
                
                # Make scheduling decision
                selected_vehicles = scheduler.select_vehicles(available_vehicles)
                
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
                
                # Set action parameters
                alpha = 1.0  # Default amplification factor
                scheduled_time = env.elapsed_time + 10  # Default scheduled time
                bandwidth = 0.5  # Default bandwidth
                
                # Execute action in environment
                _, reward, done, _ = env.step([vehicle_idx, alpha, scheduled_time, bandwidth])
                total_reward += reward
                
                # If round is complete, move to next round
                if done:
                    # Check if we've reached max rounds
                    if env.current_round >= env.max_round:
                        break
            
            # Get scheduling statistics
            stats = env.get_scheduling_stats()
            stats['episode'] = episode + 1
            stats['total_reward'] = total_reward
            stats['final_performance'] = env.current_model_performance
            episode_stats.append(stats)
            
            print(f"  Episode {episode+1}: Scheduled {stats['unique_vehicles_scheduled']}/{env.vehicle_count} vehicles "
                  f"(High: {stats['high_quality_scheduled']}, Low: {stats['low_quality_scheduled']})")
        
        # Compute average metrics
        avg_stats = {
            'unique_vehicles_scheduled': np.mean([s['unique_vehicles_scheduled'] for s in episode_stats]),
            'high_quality_scheduled': np.mean([s['high_quality_scheduled'] for s in episode_stats]),
            'low_quality_scheduled': np.mean([s['low_quality_scheduled'] for s in episode_stats]),
            'scheduling_success_rate': np.mean([s['scheduling_success_rate'] for s in episode_stats]),
            'total_attempts': np.mean([s['total_attempts'] for s in episode_stats]),
            'total_successes': np.mean([s['total_successes'] for s in episode_stats]),
            'episode_stats': episode_stats
        }
        
        scheduling_results[name] = avg_stats
        
        print(f"  Average: Scheduled {avg_stats['unique_vehicles_scheduled']:.1f} vehicles "
              f"(Success Rate: {avg_stats['scheduling_success_rate']:.2f})")
    
    # Plot results
    try:
        plot_scheduling_count_results(scheduling_results)
    except Exception as e:
        print(f"Warning: Error plotting scheduling results: {e}")
        traceback.print_exc()
    
    return scheduling_results
