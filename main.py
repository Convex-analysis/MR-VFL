"""
Main script for MR-VFL scheduler comparison.

This script runs various experiments to compare different schedulers
in the MR-VFL framework, with a focus on demonstrating the advantages
of the Mamba-based scheduler.
"""

import os
import random
import numpy as np
import torch
import argparse
from datetime import datetime

# Import configuration
from config import random_seed, output_dir

# Import experiments
from experiments.standard_comparison import run_comparison
from experiments.streaming_scenario import run_streaming_scenario
from experiments.dynamic_scenario import run_dynamic_scenario
from experiments.scheduling_count_scenario import run_vehicle_scheduling_count_scenario

# Import visualization utilities
from utils.visualization import plot_results
from utils.metrics import calculate_metrics

# Set random seeds for reproducibility
random.seed(random_seed)
np.random.seed(random_seed)
torch.manual_seed(random_seed)

def main():
    """Main execution function"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="MR-VFL Scheduler Comparison")
    parser.add_argument("--experiment", type=str, default="all",
                        choices=["all", "standard", "streaming", "dynamic", "scheduling_count"],
                        help="Experiment to run")
    parser.add_argument("--output", type=str, default=output_dir,
                        help="Output directory for results")
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output, exist_ok=True)
    
    # Print header
    print("=" * 80)
    print("MR-VFL Scheduler Comparison")
    print("=" * 80)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Output directory: {args.output}")
    print(f"Device: {torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')}")
    print("=" * 80)
    
    # Run experiments
    if args.experiment in ["all", "standard"]:
        print("\n=== Running Standard Comparison ===")
        results = run_comparison()
        metrics = calculate_metrics(results)
        plot_results(results)
    
    if args.experiment in ["all", "streaming"]:
        streaming_results = run_streaming_scenario()
    
    if args.experiment in ["all", "dynamic"]:
        dynamic_results, phase_performances = run_dynamic_scenario()
    
    if args.experiment in ["all", "scheduling_count"]:
        scheduling_results = run_vehicle_scheduling_count_scenario()
    
    # Print conclusion
    print("\n" + "=" * 80)
    print("Experiment(s) completed successfully")
    print(f"Results saved to {args.output}")
    print("=" * 80)

if __name__ == "__main__":
    main()
