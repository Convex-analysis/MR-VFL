"""
Evaluation script for the Mamba scheduler in MR-VFL.

This script evaluates the trained Mamba scheduler and compares it with other schedulers
in different scenarios.
"""

import os
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm import tqdm

# Import MR-VFL components
from schedulers import initialize_schedulers
from experiments.standard_comparison import run_comparison
from experiments.streaming_scenario import run_streaming_scenario
from experiments.dynamic_scenario import run_dynamic_scenario
from experiments.scheduling_count_scenario import run_vehicle_scheduling_count_scenario

def evaluate_scheduler(args):
    """Evaluate the Mamba scheduler and compare with other schedulers"""
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join("results", "evaluation", timestamp)
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize schedulers
    schedulers = initialize_schedulers()
    
    # Run evaluations based on specified scenarios
    results = {}
    
    if "standard" in args.scenarios:
        print("\n=== Running Standard Comparison ===")
        standard_results = run_comparison(num_episodes=args.num_episodes, max_rounds=args.max_rounds)
        results["standard"] = standard_results
        
        # Plot standard comparison results
        plot_standard_comparison(standard_results, output_dir)
    
    if "streaming" in args.scenarios:
        print("\n=== Running Streaming Scenario ===")
        streaming_results = run_streaming_scenario()
        results["streaming"] = streaming_results
        
        # Plot streaming scenario results
        plot_streaming_scenario(streaming_results, output_dir)
    
    if "dynamic" in args.scenarios:
        print("\n=== Running Dynamic Scenario ===")
        dynamic_results, phase_performances = run_dynamic_scenario()
        results["dynamic"] = dynamic_results
        results["phase_performances"] = phase_performances
        
        # Plot dynamic scenario results
        plot_dynamic_scenario(dynamic_results, phase_performances, output_dir)
    
    if "scheduling_count" in args.scenarios:
        print("\n=== Running Vehicle Scheduling Count Scenario ===")
        scheduling_count_results = run_vehicle_scheduling_count_scenario()
        results["scheduling_count"] = scheduling_count_results
        
        # Plot scheduling count results
        plot_scheduling_count(scheduling_count_results, output_dir)
    
    # Save results
    save_results_summary(results, output_dir)
    
    return results

def plot_standard_comparison(results, output_dir):
    """Plot standard comparison results"""
    # Extract data
    scheduler_names = list(results.keys())
    rewards = [results[name]["avg_reward"] for name in scheduler_names]
    performances = [results[name]["avg_performance"] for name in scheduler_names]
    decision_times = [results[name]["avg_decision_time"] for name in scheduler_names]
    
    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Plot rewards
    axes[0].bar(scheduler_names, rewards)
    axes[0].set_title("Average Reward")
    axes[0].set_ylabel("Reward")
    axes[0].set_xticklabels(scheduler_names, rotation=45, ha="right")
    
    # Plot performances
    axes[1].bar(scheduler_names, performances)
    axes[1].set_title("Average Performance")
    axes[1].set_ylabel("Performance")
    axes[1].set_xticklabels(scheduler_names, rotation=45, ha="right")
    
    # Plot decision times
    axes[2].bar(scheduler_names, decision_times)
    axes[2].set_title("Average Decision Time (ms)")
    axes[2].set_ylabel("Time (ms)")
    axes[2].set_xticklabels(scheduler_names, rotation=45, ha="right")
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure
    plt.savefig(os.path.join(output_dir, "standard_comparison.png"))
    plt.close()

def plot_streaming_scenario(results, output_dir):
    """Plot streaming scenario results"""
    # Extract data
    scheduler_names = list(results.keys())
    rewards = [results[name]["total_reward"] for name in scheduler_names]
    performances = [results[name]["final_performance"] for name in scheduler_names]
    decision_times = [results[name]["avg_decision_time"] for name in scheduler_names]
    
    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Plot rewards
    axes[0].bar(scheduler_names, rewards)
    axes[0].set_title("Total Reward")
    axes[0].set_ylabel("Reward")
    axes[0].set_xticklabels(scheduler_names, rotation=45, ha="right")
    
    # Plot performances
    axes[1].bar(scheduler_names, performances)
    axes[1].set_title("Final Performance")
    axes[1].set_ylabel("Performance")
    axes[1].set_xticklabels(scheduler_names, rotation=45, ha="right")
    
    # Plot decision times
    axes[2].bar(scheduler_names, decision_times)
    axes[2].set_title("Average Decision Time (ms)")
    axes[2].set_ylabel("Time (ms)")
    axes[2].set_xticklabels(scheduler_names, rotation=45, ha="right")
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure
    plt.savefig(os.path.join(output_dir, "streaming_scenario.png"))
    plt.close()

def plot_dynamic_scenario(results, phase_performances, output_dir):
    """Plot dynamic scenario results"""
    # Extract data
    scheduler_names = list(results.keys())
    rewards = [results[name]["total_reward"] for name in scheduler_names]
    final_performances = [results[name]["final_performance"] for name in scheduler_names]
    adaptability_scores = [min(results[name]["avg_phase_performance"]) for name in scheduler_names]
    
    # Create figure for overall results
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Plot rewards
    axes[0].bar(scheduler_names, rewards)
    axes[0].set_title("Total Reward")
    axes[0].set_ylabel("Reward")
    axes[0].set_xticklabels(scheduler_names, rotation=45, ha="right")
    
    # Plot final performances
    axes[1].bar(scheduler_names, final_performances)
    axes[1].set_title("Final Performance")
    axes[1].set_ylabel("Performance")
    axes[1].set_xticklabels(scheduler_names, rotation=45, ha="right")
    
    # Plot adaptability scores
    axes[2].bar(scheduler_names, adaptability_scores)
    axes[2].set_title("Adaptability Score")
    axes[2].set_ylabel("Score")
    axes[2].set_xticklabels(scheduler_names, rotation=45, ha="right")
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure
    plt.savefig(os.path.join(output_dir, "dynamic_scenario_overall.png"))
    plt.close()
    
    # Create figure for phase performances
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Set width of bars
    bar_width = 0.15
    
    # Set positions of bars on x-axis
    r = np.arange(len(scheduler_names))
    
    # Plot bars for each phase
    for i, phase in enumerate(["Phase 1", "Phase 2", "Phase 3", "Phase 4"]):
        phase_data = [results[name]["avg_phase_performance"][i] for name in scheduler_names]
        ax.bar(r + i * bar_width, phase_data, width=bar_width, label=phase)
    
    # Add labels and legend
    ax.set_xlabel("Scheduler")
    ax.set_ylabel("Performance")
    ax.set_title("Performance by Phase")
    ax.set_xticks(r + bar_width * 1.5)
    ax.set_xticklabels(scheduler_names, rotation=45, ha="right")
    ax.legend()
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure
    plt.savefig(os.path.join(output_dir, "dynamic_scenario_phases.png"))
    plt.close()

def plot_scheduling_count(results, output_dir):
    """Plot scheduling count results"""
    # Extract data
    scheduler_names = list(results.keys())
    unique_vehicles = [results[name]["unique_vehicles"] for name in scheduler_names]
    high_quality = [results[name]["high_quality"] for name in scheduler_names]
    low_quality = [results[name]["low_quality"] for name in scheduler_names]
    
    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Plot unique vehicles
    axes[0].bar(scheduler_names, unique_vehicles)
    axes[0].set_title("Unique Vehicles Scheduled")
    axes[0].set_ylabel("Count")
    axes[0].set_xticklabels(scheduler_names, rotation=45, ha="right")
    
    # Plot high quality vehicles
    axes[1].bar(scheduler_names, high_quality)
    axes[1].set_title("High Quality Vehicles Scheduled")
    axes[1].set_ylabel("Count")
    axes[1].set_xticklabels(scheduler_names, rotation=45, ha="right")
    
    # Plot low quality vehicles
    axes[2].bar(scheduler_names, low_quality)
    axes[2].set_title("Low Quality Vehicles Scheduled")
    axes[2].set_ylabel("Count")
    axes[2].set_xticklabels(scheduler_names, rotation=45, ha="right")
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure
    plt.savefig(os.path.join(output_dir, "scheduling_count.png"))
    plt.close()

def save_results_summary(results, output_dir):
    """Save results summary to file"""
    with open(os.path.join(output_dir, "results_summary.txt"), "w") as f:
        f.write("# MR-VFL Scheduler Evaluation Results\n\n")
        
        # Standard comparison results
        if "standard" in results:
            f.write("## Standard Comparison\n\n")
            f.write("| Scheduler | Avg Reward | Avg Performance | Avg Decision Time (ms) |\n")
            f.write("|-----------|------------|-----------------|------------------------|\n")
            
            for name, data in results["standard"].items():
                f.write(f"| {name} | {data['avg_reward']:.2f} | {data['avg_performance']:.4f} | {data['avg_decision_time']:.2f} |\n")
            
            f.write("\n")
        
        # Streaming scenario results
        if "streaming" in results:
            f.write("## Streaming Scenario\n\n")
            f.write("| Scheduler | Total Reward | Final Performance | Avg Decision Time (ms) |\n")
            f.write("|-----------|--------------|-------------------|------------------------|\n")
            
            for name, data in results["streaming"].items():
                f.write(f"| {name} | {data['total_reward']:.2f} | {data['final_performance']:.4f} | {data['avg_decision_time']:.2f} |\n")
            
            f.write("\n")
        
        # Dynamic scenario results
        if "dynamic" in results:
            f.write("## Dynamic Scenario\n\n")
            f.write("| Scheduler | Total Reward | Final Performance | Adaptability Score |\n")
            f.write("|-----------|--------------|-------------------|--------------------|\n")
            
            for name, data in results["dynamic"].items():
                adaptability = min(data["avg_phase_performance"])
                f.write(f"| {name} | {data['total_reward']:.2f} | {data['final_performance']:.4f} | {adaptability:.4f} |\n")
            
            f.write("\n")
            
            f.write("### Performance by Phase\n\n")
            f.write("| Scheduler | Phase 1 | Phase 2 | Phase 3 | Phase 4 |\n")
            f.write("|-----------|---------|---------|---------|----------|\n")
            
            for name, data in results["dynamic"].items():
                phase_avg = data["avg_phase_performance"]
                f.write(f"| {name} | {phase_avg[0]:.4f} | {phase_avg[1]:.4f} | {phase_avg[2]:.4f} | {phase_avg[3]:.4f} |\n")
            
            f.write("\n")
        
        # Scheduling count results
        if "scheduling_count" in results:
            f.write("## Scheduling Count\n\n")
            f.write("| Scheduler | Unique Vehicles | High Quality | Low Quality | Success Rate (%) |\n")
            f.write("|-----------|----------------|--------------|-------------|------------------|\n")
            
            for name, data in results["scheduling_count"].items():
                f.write(f"| {name} | {data['unique_vehicles']:.1f} | {data['high_quality']:.1f} | {data['low_quality']:.1f} | {data['success_rate']:.1f} |\n")
            
            f.write("\n")
        
        # Mamba advantage
        if "standard" in results and "Mamba" in results["standard"] and "Transformer" in results["standard"]:
            f.write("## Mamba's Advantage over Transformer\n\n")
            
            # Standard comparison
            mamba_perf = results["standard"]["Mamba"]["avg_performance"]
            transformer_perf = results["standard"]["Transformer"]["avg_performance"]
            perf_advantage = (mamba_perf / transformer_perf - 1) * 100 if transformer_perf > 0 else float('inf')
            
            mamba_time = results["standard"]["Mamba"]["avg_decision_time"]
            transformer_time = results["standard"]["Transformer"]["avg_decision_time"]
            time_advantage = (transformer_time / mamba_time - 1) * 100 if mamba_time > 0 else float('inf')
            
            f.write(f"- Standard Performance Advantage: {perf_advantage:.1f}%\n")
            f.write(f"- Standard Decision Time Advantage: {time_advantage:.1f}%\n")
            
            # Streaming scenario
            if "streaming" in results and "Mamba" in results["streaming"] and "Transformer" in results["streaming"]:
                mamba_perf = results["streaming"]["Mamba"]["final_performance"]
                transformer_perf = results["streaming"]["Transformer"]["final_performance"]
                perf_advantage = (mamba_perf / transformer_perf - 1) * 100 if transformer_perf > 0 else float('inf')
                
                f.write(f"- Streaming Performance Advantage: {perf_advantage:.1f}%\n")
            
            # Dynamic scenario
            if "dynamic" in results and "Mamba" in results["dynamic"] and "Transformer" in results["dynamic"]:
                mamba_adapt = min(results["dynamic"]["Mamba"]["avg_phase_performance"])
                transformer_adapt = min(results["dynamic"]["Transformer"]["avg_phase_performance"])
                adapt_advantage = (mamba_adapt / transformer_adapt - 1) * 100 if transformer_adapt > 0 else float('inf')
                
                f.write(f"- Adaptability Advantage: {adapt_advantage:.1f}%\n")
            
            f.write("\n")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Evaluate Mamba scheduler for MR-VFL")
    
    # Evaluation parameters
    parser.add_argument("--scenarios", nargs="+", default=["standard", "streaming", "dynamic", "scheduling_count"],
                        help="Scenarios to evaluate")
    parser.add_argument("--num_episodes", type=int, default=5, help="Number of episodes for standard comparison")
    parser.add_argument("--max_rounds", type=int, default=100, help="Maximum number of rounds per episode")
    
    args = parser.parse_args()
    
    # Evaluate the scheduler
    evaluate_scheduler(args)

if __name__ == "__main__":
    main()
