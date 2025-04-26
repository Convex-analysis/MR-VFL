"""
Greedy schedulers for MR-VFL.
"""

import random
from .base_scheduler import BaseScheduler
from .heuristic_scheduler import HeuristicScheduler

class GreedyQualityScheduler(HeuristicScheduler):
    """Greedy scheduler that selects vehicles based on data quality"""
    
    def __init__(self):
        """Initialize the greedy quality scheduler"""
        super().__init__("Greedy-Quality")
    
    def sort_vehicles(self, vehicles):
        """
        Sort vehicles by data quality (descending).
        
        Args:
            vehicles: List of eligible vehicle dictionaries
            
        Returns:
            Sorted list of vehicle dictionaries
        """
        return sorted(vehicles, key=lambda v: v['data_quality'], reverse=True)


class GreedyComputeScheduler(HeuristicScheduler):
    """Greedy scheduler that selects vehicles based on computation capacity"""
    
    def __init__(self):
        """Initialize the greedy compute scheduler"""
        super().__init__("Greedy-Compute")
    
    def sort_vehicles(self, vehicles):
        """
        Sort vehicles by computation capacity (descending).
        
        Args:
            vehicles: List of eligible vehicle dictionaries
            
        Returns:
            Sorted list of vehicle dictionaries
        """
        return sorted(vehicles, key=lambda v: v['computation_capacity'], reverse=True)


class RandomScheduler(BaseScheduler):
    """Random scheduler that selects vehicles randomly"""
    
    def __init__(self):
        """Initialize the random scheduler"""
        super().__init__("Random")
    
    def select_vehicles(self, vehicles, target_count=10):
        """
        Select vehicles randomly.
        
        Args:
            vehicles: List of vehicle dictionaries from the environment
            target_count: Maximum number of vehicles to select
            
        Returns:
            List of selected vehicle dictionaries
        """
        # Get eligible vehicles
        eligible_vehicles = self.get_eligible_vehicles(vehicles)
        
        if not eligible_vehicles:
            return []
        
        # Select random vehicles
        selected_count = min(target_count, len(eligible_vehicles))
        return random.sample(eligible_vehicles, selected_count)
