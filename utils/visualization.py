"""
Visualization utilities for MR-VFL scheduler comparison.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import traceback

# Create output directory if it doesn't exist
output_dir = os.path.join("MR-VFL", "results")
os.makedirs(output_dir, exist_ok=True)

def plot_results(results):
    """
    Plot comparison results.
    
    Args:
        results: Dictionary of results for each scheduler
    """
    # Create timestamp for output files
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    try:
        # Plot metrics
        metrics = ['avg_reward', 'avg_performance', 'avg_length', 'avg_decision_time']
        titles = ['Average Reward', 'Average Performance', 'Average Episode Length', 'Average Decision Time (ms)']
        ylabels = ['Reward', 'Performance', 'Rounds', 'Time (ms)']
        
        plt.figure(figsize=(15, 10))
        
        for i, (metric, title, ylabel) in enumerate(zip(metrics, titles, ylabels)):
            plt.subplot(2, 2, i+1)
            
            # Get scheduler names and values
            names = list(results.keys())
            values = [results[name][metric] for name in names]
            
            # Create bar chart
            bars = plt.bar(names, values)
            plt.title(title)
            plt.ylabel(ylabel)
            plt.xticks(rotation=45)
            
            # Highlight Mamba bar if present
            for j, name in enumerate(names):
                if name == "Mamba":
                    bars[j].set_color('green')
            
            # Add values on top of bars
            for j, v in enumerate(values):
                if metric == 'avg_performance':
                    plt.text(j, v, f"{v:.4f}", ha='center', va='bottom')
                else:
                    plt.text(j, v, f"{v:.2f}", ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"scheduler_comparison_{timestamp}.png"))
        
        # Plot performance over time for each scheduler
        plt.figure(figsize=(10, 6))
        for name in results:
            if name == "Mamba":
                plt.plot(results[name]['performances'], 'g-', linewidth=2, label=name)
            else:
                plt.plot(results[name]['performances'], '--', alpha=0.7, label=name)
        
        plt.title('Model Performance by Episode')
        plt.xlabel('Episode')
        plt.ylabel('Performance')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, f"performance_comparison_{timestamp}.png"))
        
        # Plot decision time distribution
        plt.figure(figsize=(10, 6))
        
        # Use boxplot for better visualization
        boxplot_data = []
        boxplot_labels = []
        
        for name in results:
            if results[name]['decision_times']:
                boxplot_data.append(results[name]['decision_times'])
                boxplot_labels.append(name)
        
        if boxplot_data:
            box = plt.boxplot(boxplot_data, labels=boxplot_labels, patch_artist=True)
            
            # Customize boxplot colors
            for i, patch in enumerate(box['boxes']):
                if boxplot_labels[i] == "Mamba":
                    patch.set_facecolor('lightgreen')
                else:
                    patch.set_facecolor('lightblue')
        
        plt.title('Decision Time Distribution')
        plt.ylabel('Time (ms)')
        plt.xticks(rotation=45)
        plt.grid(True, axis='y')
        plt.savefig(os.path.join(output_dir, f"decision_time_comparison_{timestamp}.png"))
        
        # Save results to CSV
        summary = {name: {
            'Avg Reward': results[name]['avg_reward'],
            'Avg Performance': results[name]['avg_performance'],
            'Avg Episode Length': results[name]['avg_length'],
            'Avg Decision Time (ms)': results[name]['avg_decision_time']
        } for name in results}
        
        df = pd.DataFrame(summary).T
        df.to_csv(os.path.join(output_dir, f"scheduler_comparison_{timestamp}.csv"))
        
        # Create summary report
        with open(os.path.join(output_dir, f"comparison_summary_{timestamp}.txt"), "w") as f:
            f.write("# MR-VFL Scheduler Comparison Results\n\n")
            
            f.write("## Average Metrics\n\n")
            f.write("| Scheduler | Reward | Performance | Length | Decision Time (ms) |\n")
            f.write("|-----------|--------|-------------|--------|-------------------|\n")
            
            for name in results:
                f.write(f"| {name} | {results[name]['avg_reward']:.2f} | {results[name]['avg_performance']:.4f} | {results[name]['avg_length']:.1f} | {results[name]['avg_decision_time']:.2f} |\n")
            
            f.write("\n## Conclusion\n\n")
            
            # Find best scheduler for each metric
            best_reward = max(results.items(), key=lambda x: x[1]['avg_reward'])[0]
            best_performance = max(results.items(), key=lambda x: x[1]['avg_performance'])[0]
            fastest = min(results.items(), key=lambda x: x[1]['avg_decision_time'] if x[1]['avg_decision_time'] > 0 else float('inf'))[0]
            
            f.write(f"- Best reward: {best_reward}\n")
            f.write(f"- Best performance: {best_performance}\n")
            f.write(f"- Fastest decision time: {fastest}\n")
            
            # Add comparison with baseline
            if "Random" in results:
                baseline = "Random"
                f.write(f"\n## Improvement Over {baseline} Baseline\n\n")
                f.write("| Scheduler | Reward Improvement | Performance Improvement | Speed Improvement |\n")
                f.write("|-----------|-------------------|-------------------------|------------------|\n")
                
                baseline_reward = results[baseline]['avg_reward']
                baseline_performance = results[baseline]['avg_performance']
                baseline_time = results[baseline]['avg_decision_time']
                
                for name in results:
                    if name != baseline:
                        reward_improvement = (results[name]['avg_reward'] / baseline_reward - 1) * 100
                        performance_improvement = (results[name]['avg_performance'] / baseline_performance - 1) * 100
                        
                        if baseline_time > 0 and results[name]['avg_decision_time'] > 0:
                            speed_improvement = (baseline_time / results[name]['avg_decision_time'] - 1) * 100
                        else:
                            speed_improvement = 0
                        
                        f.write(f"| {name} | {reward_improvement:.1f}% | {performance_improvement:.1f}% | {speed_improvement:.1f}% |\n")
    except Exception as e:
        print(f"Warning: Error plotting results: {e}")
        traceback.print_exc()
    
    print(f"Results saved to {output_dir}")
    print(f"Summary report: {os.path.join(output_dir, f'comparison_summary_{timestamp}.txt')}")

def plot_scheduling_count_results(results):
    """
    Plot scheduling count results.
    
    Args:
        results: Dictionary of scheduling count results for each scheduler
    """
    # Create timestamp for output files
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    try:
        # Plot unique vehicles scheduled
        plt.figure(figsize=(12, 8))
        
        # Get scheduler names and values
        names = list(results.keys())
        unique_counts = [results[name]['unique_vehicles_scheduled'] for name in names]
        high_quality_counts = [results[name]['high_quality_scheduled'] for name in names]
        low_quality_counts = [results[name]['low_quality_scheduled'] for name in names]
        
        # Set up bar positions
        x = np.arange(len(names))
        width = 0.25
        
        # Create grouped bar chart
        plt.bar(x - width, unique_counts, width, label='Unique Vehicles')
        plt.bar(x, high_quality_counts, width, label='High Quality')
        plt.bar(x + width, low_quality_counts, width, label='Low Quality')
        
        plt.title('Vehicle Scheduling Counts by Scheduler')
        plt.xlabel('Scheduler')
        plt.ylabel('Count')
        plt.xticks(x, names, rotation=45)
        plt.legend()
        plt.grid(True, axis='y')
        plt.tight_layout()
        
        plt.savefig(os.path.join(output_dir, f"scheduling_count_comparison_{timestamp}.png"))
        
        # Plot success rate
        plt.figure(figsize=(10, 6))
        
        success_rates = [results[name]['scheduling_success_rate'] * 100 for name in names]
        
        bars = plt.bar(names, success_rates)
        
        # Highlight Mamba bar if present
        for i, name in enumerate(names):
            if name == "Mamba":
                bars[i].set_color('green')
        
        plt.title('Scheduling Success Rate by Scheduler')
        plt.xlabel('Scheduler')
        plt.ylabel('Success Rate (%)')
        plt.xticks(rotation=45)
        plt.grid(True, axis='y')
        
        # Add values on top of bars
        for i, v in enumerate(success_rates):
            plt.text(i, v, f"{v:.1f}%", ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"scheduling_success_rate_{timestamp}.png"))
        
        # Save results to CSV
        summary = {name: {
            'Unique Vehicles': results[name]['unique_vehicles_scheduled'],
            'High Quality': results[name]['high_quality_scheduled'],
            'Low Quality': results[name]['low_quality_scheduled'],
            'Success Rate (%)': results[name]['scheduling_success_rate'] * 100,
            'Total Attempts': results[name]['total_attempts'],
            'Total Successes': results[name]['total_successes']
        } for name in results}
        
        df = pd.DataFrame(summary).T
        df.to_csv(os.path.join(output_dir, f"scheduling_count_comparison_{timestamp}.csv"))
        
        # Create summary report
        with open(os.path.join(output_dir, f"scheduling_count_summary_{timestamp}.txt"), "w") as f:
            f.write("# MR-VFL Scheduler Scheduling Count Results\n\n")
            
            f.write("## Scheduling Metrics\n\n")
            f.write("| Scheduler | Unique Vehicles | High Quality | Low Quality | Success Rate (%) |\n")
            f.write("|-----------|----------------|--------------|-------------|------------------|\n")
            
            for name in results:
                f.write(f"| {name} | {results[name]['unique_vehicles_scheduled']} | {results[name]['high_quality_scheduled']} | {results[name]['low_quality_scheduled']} | {results[name]['scheduling_success_rate']*100:.1f} |\n")
            
            f.write("\n## Conclusion\n\n")
            
            # Find best scheduler for each metric
            best_unique = max(results.items(), key=lambda x: x[1]['unique_vehicles_scheduled'])[0]
            best_high_quality = max(results.items(), key=lambda x: x[1]['high_quality_scheduled'])[0]
            best_success_rate = max(results.items(), key=lambda x: x[1]['scheduling_success_rate'])[0]
            
            f.write(f"- Most unique vehicles scheduled: {best_unique}\n")
            f.write(f"- Most high-quality vehicles scheduled: {best_high_quality}\n")
            f.write(f"- Best scheduling success rate: {best_success_rate}\n")
            
            # Add comparison with baseline
            if "Random" in results:
                baseline = "Random"
                f.write(f"\n## Improvement Over {baseline} Baseline\n\n")
                f.write("| Scheduler | Unique Vehicles Improvement | High Quality Improvement | Success Rate Improvement |\n")
                f.write("|-----------|------------------------------|--------------------------|---------------------------|\n")
                
                baseline_unique = results[baseline]['unique_vehicles_scheduled']
                baseline_high_quality = results[baseline]['high_quality_scheduled']
                baseline_success_rate = results[baseline]['scheduling_success_rate']
                
                for name in results:
                    if name != baseline:
                        unique_improvement = (results[name]['unique_vehicles_scheduled'] / baseline_unique - 1) * 100
                        
                        if baseline_high_quality > 0:
                            high_quality_improvement = (results[name]['high_quality_scheduled'] / baseline_high_quality - 1) * 100
                        else:
                            high_quality_improvement = 0
                            
                        success_rate_improvement = (results[name]['scheduling_success_rate'] / baseline_success_rate - 1) * 100
                        
                        f.write(f"| {name} | {unique_improvement:.1f}% | {high_quality_improvement:.1f}% | {success_rate_improvement:.1f}% |\n")
    except Exception as e:
        print(f"Warning: Error plotting scheduling count results: {e}")
        traceback.print_exc()
    
    print(f"Scheduling count results saved to {output_dir}")
    print(f"Scheduling count summary: {os.path.join(output_dir, f'scheduling_count_summary_{timestamp}.txt')}")
