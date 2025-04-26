"""
Streaming scenario experiment for MR-VFL schedulers.
"""

import time
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
import os
from datetime import datetime
from environments.streaming_env import StreamingVehicularFLEnv
from schedulers import initialize_schedulers

# Create output directory if it doesn't exist
output_dir = os.path.join("MR-VFL", "results")
os.makedirs(output_dir, exist_ok=True)

def run_streaming_scenario():
    """
    Run streaming scenario to demonstrate Mamba scheduler advantages.
    
    This scenario simulates a high-arrival-rate environment where vehicles
    arrive in bursts, testing the scheduler's ability to handle streaming data.
    """
    print("\n=== Running Streaming Scenario (Mamba Advantage) ===")
    
    # Initialize streaming environment with higher traffic density
    env_streaming = StreamingVehicularFLEnv(
        vehicle_count=100,  # Large vehicle pool
        max_round=50,
        sync_limit=1000,  # Synchronous limit of 1000s
        traffic_density=20  # Higher traffic density (20 vehicles per km²)
    )
    
    # Initialize schedulers
    schedulers = initialize_schedulers()
    
    # Select schedulers to compare
    selected_schedulers = {
        "Mamba": schedulers["Mamba"],
        "Transformer": schedulers["Transformer"],
        "Greedy-Quality": schedulers["Greedy-Quality"],
        "Random": schedulers["Random"]
    }
    
    # Run comparison
    results = {}
    
    for name, scheduler in selected_schedulers.items():
        print(f"  Evaluating {scheduler.name} scheduler in streaming environment...")
        
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
        
        # Run until max rounds or environment signals done
        while env_streaming.current_round < env_streaming.max_round:
            # Get available vehicles
            available_vehicles = []
            available_indices = []
            
            for i, vehicle in enumerate(env_streaming.vehicles):
                if env_streaming._is_vehicle_available(vehicle):
                    available_vehicles.append(vehicle)
                    available_indices.append(i)
            
            if not available_vehicles:
                # If no vehicles are available, advance time
                # Find the next vehicle that will arrive
                next_arrival = float('inf')
                for vehicle in env_streaming.vehicles:
                    if not vehicle['scheduled'] and vehicle['arrival_time'] > env_streaming.elapsed_time:
                        next_arrival = min(next_arrival, vehicle['arrival_time'])
                
                if next_arrival < float('inf'):
                    # Advance time to next arrival
                    env_streaming.elapsed_time = next_arrival
                    continue
                else:
                    # No more vehicles will arrive, end episode
                    break
            
            # Check if we've entered a new time window
            new_window = int(env_streaming.elapsed_time / window_size)
            if new_window > current_window:
                # Record vehicles scheduled in previous window
                time_windows.append(current_window * window_size)
                vehicles_scheduled.append(window_count)
                
                # Reset for new window
                current_window = new_window
                window_count = 0
            
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
            
            # Set action parameters
            alpha = 1.0  # Default amplification factor
            scheduled_time = env_streaming.elapsed_time + 10  # Default scheduled time
            bandwidth = 0.5  # Default bandwidth
            
            # Execute action in environment
            action = [vehicle_idx, alpha, scheduled_time, bandwidth]
            _, reward, done, _ = env_streaming.step(action)
            
            # Record metrics
            total_reward += reward
            
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
        
        # Record results
        results[scheduler.name] = {
            'total_reward': total_reward,
            'final_performance': env_streaming.current_model_performance,
            'avg_decision_time': np.mean(decision_times) if decision_times else 0,
            'time_windows': time_windows,
            'vehicles_scheduled': vehicles_scheduled
        }
        
        print(f"    Total Reward: {total_reward:.2f}")
        print(f"    Final Performance: {env_streaming.current_model_performance:.4f}")
        print(f"    Average Decision Time: {results[scheduler.name]['avg_decision_time']:.2f} ms")
    
    # Plot results
    plot_streaming_results(results)
    
    return results

def plot_streaming_results(results):
    """
    Plot streaming scenario results.
    
    Args:
        results: Dictionary of results for each scheduler
    """
    # Create timestamp for output files
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Plot vehicles scheduled per time window
    plt.figure(figsize=(12, 6))
    
    for name, result in results.items():
        if name == "Mamba":
            plt.plot(result['time_windows'], result['vehicles_scheduled'], 'g-', linewidth=2, label=name)
        else:
            plt.plot(result['time_windows'], result['vehicles_scheduled'], '--', alpha=0.7, label=name)
    
    plt.title('Vehicles Scheduled per Time Window')
    plt.xlabel('Time (s)')
    plt.ylabel('Vehicles Scheduled')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, f"streaming_scenario_{timestamp}.png"))
    
    # Plot final performance
    plt.figure(figsize=(10, 6))
    
    names = list(results.keys())
    performances = [results[name]['final_performance'] for name in names]
    
    bars = plt.bar(names, performances)
    
    # Highlight Mamba bar
    for i, name in enumerate(names):
        if name == "Mamba":
            bars[i].set_color('green')
    
    plt.title('Final Performance in Streaming Scenario')
    plt.ylabel('Performance')
    plt.grid(True, axis='y')
    
    # Add values on top of bars
    for i, v in enumerate(performances):
        plt.text(i, v, f"{v:.4f}", ha='center', va='bottom')
    
    plt.savefig(os.path.join(output_dir, f"streaming_performance_{timestamp}.png"))
    
    # Create summary report
    with open(os.path.join(output_dir, f"streaming_scenario_summary_{timestamp}.txt"), "w") as f:
        f.write("# MR-VFL Streaming Scenario Results\n\n")
        
        f.write("## Performance Metrics\n\n")
        f.write("| Scheduler | Total Reward | Final Performance | Avg Decision Time (ms) |\n")
        f.write("|-----------|--------------|-------------------|------------------------|\n")
        
        for name in results:
            f.write(f"| {name} | {results[name]['total_reward']:.2f} | {results[name]['final_performance']:.4f} | {results[name]['avg_decision_time']:.2f} |\n")
        
        f.write("\n## Conclusion\n\n")
        
        # Find best scheduler for each metric
        best_reward = max(results.items(), key=lambda x: x[1]['total_reward'])[0]
        best_performance = max(results.items(), key=lambda x: x[1]['final_performance'])[0]
        fastest = min(results.items(), key=lambda x: x[1]['avg_decision_time'] if x[1]['avg_decision_time'] > 0 else float('inf'))[0]
        
        f.write(f"- Best reward: {best_reward}\n")
        f.write(f"- Best performance: {best_performance}\n")
        f.write(f"- Fastest decision time: {fastest}\n")
        
        # Calculate Mamba's advantage if present
        if "Mamba" in results and "Transformer" in results:
            mamba_perf = results["Mamba"]["final_performance"]
            transformer_perf = results["Transformer"]["final_performance"]
            
            perf_advantage = (mamba_perf / transformer_perf - 1) * 100
            
            f.write(f"\n## Mamba's Advantage\n\n")
            f.write(f"- Performance advantage over Transformer: {perf_advantage:.1f}%\n")
            
            if "Random" in results:
                random_perf = results["Random"]["final_performance"]
                mamba_vs_random = (mamba_perf / random_perf - 1) * 100
                f.write(f"- Performance advantage over Random: {mamba_vs_random:.1f}%\n")
    
    print(f"Streaming scenario results saved to {output_dir}")
    print(f"Streaming summary: {os.path.join(output_dir, f'streaming_scenario_summary_{timestamp}.txt')}")
