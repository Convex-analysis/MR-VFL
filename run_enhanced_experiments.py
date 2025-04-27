"""
Script to run enhanced experiments for MR-VFL.

This script runs the enhanced experiments with more complex decision-making scenarios.
"""

import argparse
from datetime import datetime
import torch

# Import enhanced experiments
from experiments.enhanced_comparison import run_enhanced_comparison
from experiments.enhanced_streaming import run_enhanced_streaming_scenario
from experiments.enhanced_dynamic import run_enhanced_dynamic_scenario

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Run enhanced experiments for MR-VFL")
    
    # Experiment selection
    parser.add_argument("--experiment", type=str, default="all",
                        choices=["all", "comparison", "streaming", "dynamic"],
                        help="Experiment to run")
    
    # Enhanced comparison parameters
    parser.add_argument("--interference_level", type=float, default=0.3,
                        help="Level of interference between vehicles (0-1)")
    parser.add_argument("--resource_constraint", type=float, default=0.7,
                        help="Resource constraint factor (0-1)")
    parser.add_argument("--channel_variation", type=float, default=0.2,
                        help="Channel variation factor (0-1)")
    
    # Enhanced streaming parameters
    parser.add_argument("--burst_intensity", type=float, default=0.7,
                        help="Intensity of arrival bursts (0-1)")
    parser.add_argument("--quality_variation", type=float, default=0.3,
                        help="Variation in data quality over time (0-1)")
    parser.add_argument("--deadline_strictness", type=float, default=0.8,
                        help="Strictness of deadlines (0-1)")
    
    # Enhanced dynamic parameters
    parser.add_argument("--phase_difficulty", type=float, default=0.7,
                        help="Difficulty increase per phase (0-1)")
    parser.add_argument("--specialization_level", type=float, default=0.8,
                        help="Level of vehicle specialization (0-1)")
    
    # General parameters
    parser.add_argument("--num_episodes", type=int, default=1,
                        help="Number of episodes to run for comparison")
    parser.add_argument("--max_rounds", type=int, default=10,
                        help="Maximum number of rounds per episode")
    
    args = parser.parse_args()
    
    # Print header
    print("=" * 80)
    print("MR-VFL Enhanced Experiments")
    print("=" * 80)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Device: {torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')}")
    print("=" * 80)
    
    # Run experiments
    if args.experiment in ["all", "comparison"]:
        print("\n=== Running Enhanced Comparison Experiment ===")
        run_enhanced_comparison(
            num_episodes=args.num_episodes,
            max_rounds=args.max_rounds,
            interference_level=args.interference_level,
            resource_constraint=args.resource_constraint,
            channel_variation=args.channel_variation
        )
    
    if args.experiment in ["all", "streaming"]:
        print("\n=== Running Enhanced Streaming Experiment ===")
        run_enhanced_streaming_scenario(
            burst_intensity=args.burst_intensity,
            quality_variation=args.quality_variation,
            deadline_strictness=args.deadline_strictness
        )
    
    if args.experiment in ["all", "dynamic"]:
        print("\n=== Running Enhanced Dynamic Experiment ===")
        run_enhanced_dynamic_scenario(
            phase_difficulty=args.phase_difficulty,
            specialization_level=args.specialization_level
        )
    
    # Print conclusion
    print("\n" + "=" * 80)
    print("Enhanced experiment(s) completed successfully")
    print("=" * 80)

if __name__ == "__main__":
    main()
