"""
Example script demonstrating how to use the trained Mamba scheduler in experiments.

This script shows how to load the model trained by train_mamba_scheduler.py
and use it in various experiments.
"""

import os
import argparse
import torch
import numpy as np
from datetime import datetime

# Import MR-VFL components
from schedulers import initialize_schedulers
from experiments.standard_comparison import run_comparison
from experiments.streaming_scenario import run_streaming_scenario
from experiments.dynamic_scenario import run_dynamic_scenario
from experiments.scheduling_count_scenario import run_vehicle_scheduling_count_scenario

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Use trained Mamba scheduler in experiments")
    
    # Experiment selection
    parser.add_argument("--experiment", type=str, default="all",
                        choices=["all", "standard", "streaming", "dynamic", "scheduling_count"],
                        help="Experiment to run")
    
    # Model path
    parser.add_argument("--model_path", type=str, default="models/mamba_scheduler.pth",
                        help="Path to the trained model")
    
    # Environment parameters
    parser.add_argument("--vehicle_count", type=int, default=None,
                        help="Override number of vehicles in the environment")
    parser.add_argument("--traffic_density", type=int, default=None,
                        help="Override traffic density (vehicles per km²)")
    
    args = parser.parse_args()
    
    # Update config to use the specified model path
    from config import scheduler_config
    scheduler_config["mamba"]["model_path"] = args.model_path
    
    # Print header
    print("=" * 80)
    print("MR-VFL Trained Mamba Scheduler Experiments")
    print("=" * 80)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Model path: {args.model_path}")
    print(f"Device: {torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')}")
    print("=" * 80)
    
    # Initialize schedulers with the trained model
    schedulers = initialize_schedulers()
    
    # Override environment parameters if specified
    if args.vehicle_count is not None or args.traffic_density is not None:
        from config import env_config
        
        # Update standard environment
        if args.vehicle_count is not None:
            env_config["standard"]["vehicle_count"] = args.vehicle_count
            env_config["dynamic"]["vehicle_count"] = args.vehicle_count
        
        if args.traffic_density is not None:
            env_config["standard"]["traffic_density"] = args.traffic_density
            env_config["streaming"]["traffic_density"] = args.traffic_density
            env_config["dynamic"]["traffic_density"] = args.traffic_density
            env_config["scheduling_count"]["traffic_density"] = args.traffic_density
        
        print("\nOverriding environment parameters:")
        if args.vehicle_count is not None:
            print(f"  Vehicle count: {args.vehicle_count}")
        if args.traffic_density is not None:
            print(f"  Traffic density: {args.traffic_density}")
        print("=" * 80)
    
    # Run experiments
    if args.experiment in ["all", "standard"]:
        print("\n=== Running Standard Comparison ===")
        results = run_comparison()
    
    if args.experiment in ["all", "streaming"]:
        streaming_results = run_streaming_scenario()
    
    if args.experiment in ["all", "dynamic"]:
        dynamic_results, phase_performances = run_dynamic_scenario()
    
    if args.experiment in ["all", "scheduling_count"]:
        scheduling_results = run_vehicle_scheduling_count_scenario()
    
    # Print conclusion
    print("\n" + "=" * 80)
    print("Experiment(s) completed successfully")
    print("=" * 80)

if __name__ == "__main__":
    main()
