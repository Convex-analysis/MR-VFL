"""
Heuristic scheduler for MR-VFL.
"""

from abc import abstractmethod
from .base_scheduler import BaseScheduler

class HeuristicScheduler(BaseScheduler):
    """
    Base class for heuristic schedulers.
    
    This class provides common functionality for schedulers that use heuristic methods.
    """
    
    def __init__(self, name):
        """
        Initialize the heuristic scheduler.
        
        Args:
            name: Name of the scheduler
        """
        super().__init__(name)
    
    @abstractmethod
    def sort_vehicles(self, vehicles):
        """
        Sort vehicles according to the heuristic.
        
        Args:
            vehicles: List of eligible vehicle dictionaries
            
        Returns:
            Sorted list of vehicle dictionaries
        """
        pass
    
    def select_vehicles(self, vehicles, target_count=10):
        """
        Select vehicles using the heuristic.
        
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
        
        # Sort vehicles according to the heuristic
        sorted_vehicles = self.sort_vehicles(eligible_vehicles)
        
        # Select up to target_count vehicles
        return sorted_vehicles[:target_count]
