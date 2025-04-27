"""
Experiments for MR-VFL scheduler comparison.
"""

# Standard experiments
from .standard_comparison import run_comparison
from .streaming_scenario import run_streaming_scenario
from .dynamic_scenario import run_dynamic_scenario
from .scheduling_count_scenario import run_vehicle_scheduling_count_scenario

# Enhanced experiments with more complex decision-making
from .enhanced_comparison import run_enhanced_comparison
from .enhanced_streaming import run_enhanced_streaming_scenario
from .enhanced_dynamic import run_enhanced_dynamic_scenario

# Simplified enhanced experiments with more complex decision-making
from .simplified_enhanced_comparison import run_simplified_enhanced_comparison
from .simplified_enhanced_streaming import run_simplified_enhanced_streaming
from .simplified_enhanced_dynamic import run_simplified_enhanced_dynamic
