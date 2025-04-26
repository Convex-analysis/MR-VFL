"""
Dynamic scenario experiment for MR-VFL schedulers.
"""

import time
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
import os
from datetime import datetime
from ..environments.dynamic_env import DynamicVehicularFLEnv
from ..schedulers import initialize_schedulers

# Create output directory if it doesn't exist
output_dir = os.path.join("MR-VFL", "results")
os.makedirs(output_dir, exist_ok=True)

def run_dynamic_scenario():
    """
    Run dynamic scenario to demonstrate Mamba scheduler adaptability.
    
    This scenario simulates a changing environment with different phases,
    testing the scheduler's ability to adapt to different vehicle characteristics.
    """
    print("\n=== Running Dynamic Scenario (Adaptability) ===")
    
    # Initialize dynamic environment with phase changes
    env_dynamic = DynamicVehicularFLEnv(
        vehicle_count=50,
        max_round=40,  # 10 rounds per phase, 4 phases
        sync_limit=1000,
        phase_length=10,  # 10 rounds per phase
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
        print(f"  Evaluating {scheduler.name} scheduler in dynamic environment...")
        
        # Reset environment
        env_dynamic.reset()
        total_reward = 0
        decision_times = []
        
        # Track performance by phase
        phase_perf = [[] for _ in range(4)]  # 4 phases
        
        # Run until max rounds or environment signals done
        while env_dynamic.current_round < env_dynamic.max_round:
            # Get current phase
            current_phase = env_dynamic.current_round // env_dynamic.phase_length
            
            # Get available vehicles
            available_vehicles = []
            available_indices = []
            
            for i, vehicle in enumerate(env_dynamic.vehicles):
                if env_dynamic._is_vehicle_available(vehicle):
                    available_vehicles.append(vehicle)
                    available_indices.append(i)
            
            if not available_vehicles:
                # If no vehicles are available, advance time
                # Find the next vehicle that will arrive
                next_arrival = float('inf')
                for vehicle in env_dynamic.vehicles:
                    if not vehicle['scheduled'] and vehicle['arrival_time'] > env_dynamic.elapsed_time:
                        next_arrival = min(next_arrival, vehicle['arrival_time'])
                
                if next_arrival < float('inf'):
                    # Advance time to next arrival
                    env_dynamic.elapsed_time = next_arrival
                    continue
                else:
                    # No more vehicles will arrive, end episode
                    break
            
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
            
            # Set action parameters
            alpha = 1.0  # Default amplification factor
            scheduled_time = env_dynamic.elapsed_time + 10  # Default scheduled time
            bandwidth = 0.5  # Default bandwidth
            
            # Execute action in environment
            action = [vehicle_idx, alpha, scheduled_time, bandwidth]
            _, reward, done, _ = env_dynamic.step(action)
            
            # Record metrics
            total_reward += reward
            
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
        
        # Record results
        results[scheduler.name] = {
            'total_reward': total_reward,
            'final_performance': env_dynamic.current_model_performance,
            'avg_decision_time': np.mean(decision_times) if decision_times else 0,
            'phase_performances': phase_perf
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
        
        print(f"    Total Reward: {total_reward:.2f}")
        print(f"    Final Performance: {env_dynamic.current_model_performance:.4f}")
        print(f"    Average Decision Time: {results[scheduler.name]['avg_decision_time']:.2f} ms")
        print(f"    Phase Performance: {[f'{p:.4f}' for p in phase_avg]}")
    
    # Plot results
    plot_dynamic_results(results, phase_performances)
    
    return results, phase_performances

def plot_dynamic_results(results, phase_performances):
    """
    Plot dynamic scenario results.
    
    Args:
        results: Dictionary of results for each scheduler
        phase_performances: Dictionary of phase performances for each scheduler
    """
    # Create timestamp for output files
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Plot phase performance
    plt.figure(figsize=(12, 8))
    
    # Define phase names
    phase_names = ["Balanced Urban", "High Compute/Low Quality", "Low Compute/High Quality", "Poor Connectivity Rural"]
    
    # Plot performance by phase for each scheduler
    for i, name in enumerate(results.keys()):
        phase_avg = results[name]['avg_phase_performance']
        
        if name == "Mamba":
            plt.plot(range(4), phase_avg, 'go-', linewidth=2, label=name)
        else:
            plt.plot(range(4), phase_avg, 'o--', alpha=0.7, label=name)
    
    plt.title('Performance by Phase')
    plt.xlabel('Phase')
    plt.ylabel('Average Performance')
    plt.xticks(range(4), [f"Phase {i+1}\n({phase_names[i]})" for i in range(4)], rotation=45)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"dynamic_scenario_{timestamp}.png"))
    
    # Plot adaptability (performance variance across phases)
    plt.figure(figsize=(10, 6))
    
    names = list(results.keys())
    
    # Calculate adaptability score (higher is better)
    # We want high performance in all phases, so we use the minimum performance as a measure
    adaptability_scores = []
    for name in names:
        phase_avg = results[name]['avg_phase_performance']
        # Adaptability score is the minimum performance across all phases
        adaptability_scores.append(min(phase_avg))
    
    bars = plt.bar(names, adaptability_scores)
    
    # Highlight Mamba bar
    for i, name in enumerate(names):
        if name == "Mamba":
            bars[i].set_color('green')
    
    plt.title('Adaptability Score (Minimum Performance Across Phases)')
    plt.ylabel('Score')
    plt.grid(True, axis='y')
    
    # Add values on top of bars
    for i, v in enumerate(adaptability_scores):
        plt.text(i, v, f"{v:.4f}", ha='center', va='bottom')
    
    plt.savefig(os.path.join(output_dir, f"dynamic_adaptability_{timestamp}.png"))
    
    # Create summary report
    with open(os.path.join(output_dir, f"dynamic_scenario_summary_{timestamp}.txt"), "w") as f:
        f.write("# MR-VFL Dynamic Scenario Results\n\n")
        
        f.write("## Phase Descriptions\n\n")
        f.write("- **Phase 1 (Balanced Urban)**: Typical urban environment with balanced vehicle characteristics.\n")
        f.write("  - Moderate computation capacity, data quality, and connectivity\n")
        f.write("  - Predictable arrival patterns and reasonable sojourn times\n")
        f.write("  - Represents a baseline scenario with optimal conditions\n\n")
        
        f.write("- **Phase 2 (High Compute, Low Quality)**: Powerful vehicles with lower quality data.\n")
        f.write("  - High computation capacity (powerful onboard computers)\n")
        f.write("  - Low data quality (poor sensors or challenging conditions)\n")
        f.write("  - Good connectivity but shorter stays in the area\n")
        f.write("  - Represents luxury vehicles or buses with high mobility\n\n")
        
        f.write("- **Phase 3 (Low Compute, High Quality)**: Resource-constrained vehicles with high-quality data.\n")
        f.write("  - Low computation capacity (limited onboard resources)\n")
        f.write("  - High data quality (specialized sensors or optimal positioning)\n")
        f.write("  - Good connectivity and longer stays in the area\n")
        f.write("  - Represents older vehicles or IoT devices in fixed positions\n\n")
        
        f.write("- **Phase 4 (Poor Connectivity Rural)**: Diverse vehicles in challenging network conditions.\n")
        f.write("  - Mixed computation capacity and data quality\n")
        f.write("  - Poor connectivity (network issues, interference, distance)\n")
        f.write("  - Unpredictable arrivals and departures\n")
        f.write("  - Represents rural areas or adverse weather conditions\n\n")
        
        f.write("## Performance Metrics\n\n")
        f.write("| Scheduler | Total Reward | Final Performance | Avg Decision Time (ms) | Adaptability Score |\n")
        f.write("|-----------|--------------|-------------------|------------------------|--------------------|\n")
        
        for name in results:
            adaptability = min(results[name]['avg_phase_performance'])
            f.write(f"| {name} | {results[name]['total_reward']:.2f} | {results[name]['final_performance']:.4f} | {results[name]['avg_decision_time']:.2f} | {adaptability:.4f} |\n")
        
        f.write("\n## Performance by Phase\n\n")
        f.write("| Scheduler | Phase 1 (Balanced) | Phase 2 (High Compute) | Phase 3 (Low Compute) | Phase 4 (Poor Connectivity) |\n")
        f.write("|-----------|-------------------|------------------------|----------------------|-----------------------------|\n")
        
        for name in results:
            phase_avg = results[name]['avg_phase_performance']
            f.write(f"| {name} | {phase_avg[0]:.4f} | {phase_avg[1]:.4f} | {phase_avg[2]:.4f} | {phase_avg[3]:.4f} |\n")
        
        f.write("\n## Conclusion\n\n")
        
        # Find best scheduler for each phase
        best_phase1 = max(results.items(), key=lambda x: x[1]['avg_phase_performance'][0])[0]
        best_phase2 = max(results.items(), key=lambda x: x[1]['avg_phase_performance'][1])[0]
        best_phase3 = max(results.items(), key=lambda x: x[1]['avg_phase_performance'][2])[0]
        best_phase4 = max(results.items(), key=lambda x: x[1]['avg_phase_performance'][3])[0]
        
        # Find most adaptable scheduler
        most_adaptable = max(results.items(), key=lambda x: min(x[1]['avg_phase_performance']))[0]
        
        f.write(f"- Best in Phase 1 (Balanced Urban): {best_phase1}\n")
        f.write(f"- Best in Phase 2 (High Compute/Low Quality): {best_phase2}\n")
        f.write(f"- Best in Phase 3 (Low Compute/High Quality): {best_phase3}\n")
        f.write(f"- Best in Phase 4 (Poor Connectivity Rural): {best_phase4}\n")
        f.write(f"- Most adaptable scheduler: {most_adaptable}\n")
        
        # Calculate Mamba's advantage if present
        if "Mamba" in results and "Transformer" in results:
            mamba_adaptability = min(results["Mamba"]["avg_phase_performance"])
            transformer_adaptability = min(results["Transformer"]["avg_phase_performance"])
            
            adaptability_advantage = (mamba_adaptability / transformer_adaptability - 1) * 100
            
            f.write(f"\n## Mamba's Advantage\n\n")
            f.write(f"- Adaptability advantage over Transformer: {adaptability_advantage:.1f}%\n")
    
    print(f"Dynamic scenario results saved to {output_dir}")
    print(f"Dynamic summary: {os.path.join(output_dir, f'dynamic_scenario_summary_{timestamp}.txt')}")
