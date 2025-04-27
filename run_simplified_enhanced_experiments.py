"""
Script to run simplified enhanced experiments for MR-VFL.

This script runs the simplified enhanced experiments with more complex decision-making scenarios
while ensuring stability and performance.
"""

import argparse
from datetime import datetime
import torch

# Import simplified enhanced experiments
from experiments.simplified_enhanced_comparison import run_simplified_enhanced_comparison
from experiments.simplified_enhanced_streaming import run_simplified_enhanced_streaming
from experiments.simplified_enhanced_dynamic import run_simplified_enhanced_dynamic

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Run simplified enhanced experiments for MR-VFL")
    
    # Experiment selection
    parser.add_argument("--experiment", type=str, default="all",
                        choices=["all", "comparison", "streaming", "dynamic"],
                        help="Experiment to run")
    
    # General parameters
    parser.add_argument("--num_episodes", type=int, default=1,
                        help="Number of episodes to run for comparison")
    parser.add_argument("--max_rounds", type=int, default=10,
                        help="Maximum number of rounds per episode")
    
    args = parser.parse_args()
    
    # Print header
    print("=" * 80)
    print("MR-VFL Simplified Enhanced Experiments")
    print("=" * 80)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Device: {torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')}")
    print("=" * 80)
    
    # Run experiments
    if args.experiment in ["all", "comparison"]:
        print("\n=== Running Simplified Enhanced Comparison Experiment ===")
        run_simplified_enhanced_comparison(
            num_episodes=args.num_episodes,
            max_rounds=args.max_rounds
        )
    
    if args.experiment in ["all", "streaming"]:
        print("\n=== Running Simplified Enhanced Streaming Experiment ===")
        run_simplified_enhanced_streaming(
            num_episodes=args.num_episodes
        )
    
    if args.experiment in ["all", "dynamic"]:
        print("\n=== Running Simplified Enhanced Dynamic Experiment ===")
        run_simplified_enhanced_dynamic()
    
    # Print conclusion
    print("\n" + "=" * 80)
    print("Simplified enhanced experiment(s) completed successfully")
    print("=" * 80)

if __name__ == "__main__":
    main()
