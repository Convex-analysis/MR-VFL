"""
Script to run experiments with the model trained by train_mamba_scheduler.py.

This script demonstrates how all experiments in the experiments/ directory
can use the model trained by train_mamba_scheduler.py.
"""

import os
import argparse
import torch
from datetime import datetime

# Import experiment functions
from experiments.standard_comparison import run_comparison
from experiments.streaming_scenario import run_streaming_scenario
from experiments.dynamic_scenario import run_dynamic_scenario
from experiments.scheduling_count_scenario import run_vehicle_scheduling_count_scenario

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Run experiments with trained Mamba model")
    
    # Experiment selection
    parser.add_argument("--experiment", type=str, default="all",
                        choices=["all", "standard", "streaming", "dynamic", "scheduling_count"],
                        help="Experiment to run")
    
    # Environment parameters
    parser.add_argument("--vehicle_count", type=int, default=None,
                        help="Override number of vehicles in the environment")
    parser.add_argument("--traffic_density", type=int, default=None,
                        help="Override traffic density (vehicles per km²)")
    
    args = parser.parse_args()
    
    # Print header
    print("=" * 80)
    print("Running Experiments with Trained Mamba Model")
    print("=" * 80)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Model path: models/mamba_scheduler.pth")
    print(f"Device: {torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')}")
    print("=" * 80)
    
    # Check if the trained model exists
    if not os.path.exists("models/mamba_scheduler.pth"):
        print("\nWARNING: Trained model not found at models/mamba_scheduler.pth")
        print("Please run train_mamba_scheduler.py first to train the model.")
        print("Using untrained model for experiments.")
    else:
        print("\nUsing trained model from models/mamba_scheduler.pth")
    
    # Override environment parameters if specified
    if args.vehicle_count is not None or args.traffic_density is not None:
        from config import env_config
        
        # Update environment configurations
        if args.vehicle_count is not None:
            env_config["standard"]["vehicle_count"] = args.vehicle_count
            env_config["dynamic"]["vehicle_count"] = args.vehicle_count
            env_config["scheduling_count"]["vehicle_count"] = args.vehicle_count
            env_config["streaming"]["vehicle_count"] = args.vehicle_count
        
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
        print("\n=== Running Streaming Scenario ===")
        streaming_results = run_streaming_scenario()
    
    if args.experiment in ["all", "dynamic"]:
        print("\n=== Running Dynamic Scenario ===")
        dynamic_results, phase_performances = run_dynamic_scenario()
    
    if args.experiment in ["all", "scheduling_count"]:
        print("\n=== Running Scheduling Count Scenario ===")
        scheduling_results = run_vehicle_scheduling_count_scenario()
    
    # Print conclusion
    print("\n" + "=" * 80)
    print("Experiment(s) completed successfully")
    print("=" * 80)

if __name__ == "__main__":
    main()
