"""
Standard comparison experiment for MR-VFL schedulers.
"""

import time
import random
import numpy as np
import torch
from ..environments.vehicular_fl_env import VehicularFLEnv
from ..schedulers import initialize_schedulers

def run_comparison(num_episodes=5, max_rounds=100):
    """
    Run comparison between different schedulers.
    
    Args:
        num_episodes: Number of episodes to run for each scheduler
        max_rounds: Maximum number of rounds per episode
        
    Returns:
        Dictionary of results for each scheduler
    """
    # Initialize environment
    env = VehicularFLEnv(
        vehicle_count=50,
        max_round=max_rounds,
        sync_limit=1000,  # Synchronous limit of 1000s
        traffic_density=10,  # Traffic density of 10 vehicles per km²
        data_categories=10,  # 10 data categories
        learning_rate=0.01  # Learning rate of 0.01
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
            while env.current_round < max_rounds:
                # Get available vehicles and their indices
                available_vehicles = []
                available_indices = []
                
                for i, vehicle in enumerate(env.vehicles):
                    if env._is_vehicle_available(vehicle):
                        available_vehicles.append(vehicle)
                        available_indices.append(i)
                
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
                
                # Set action parameters
                alpha = 1.0  # Default amplification factor
                scheduled_time = env.elapsed_time + 10  # Default scheduled time
                bandwidth = 0.5  # Default bandwidth
                
                # Execute action in environment
                action = [vehicle_idx, alpha, scheduled_time, bandwidth]
                _, reward, done, _ = env.step(action)
                
                # Record metrics
                total_reward += reward
                
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
            
            print(f"  Episode {episode+1}/{num_episodes}: Reward={total_reward:.2f}, "
                  f"Performance={env.current_model_performance:.4f}, "
                  f"Length={env.current_round}")
        
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
        else:
            print("  No valid episodes completed")
        
        print()
    
    return results
