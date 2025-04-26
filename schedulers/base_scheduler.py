"""
Base scheduler class for MR-VFL.
"""

from abc import ABC, abstractmethod

class BaseScheduler(ABC):
    """
    Base class for all schedulers.
    
    This abstract class defines the interface that all schedulers must implement.
    """
    
    def __init__(self, name):
        """
        Initialize the base scheduler.
        
        Args:
            name: Name of the scheduler
        """
        self.name = name
    
    def get_eligible_vehicles(self, vehicles):
        """
        Get eligible vehicles for scheduling.
        
        Args:
            vehicles: List of vehicle dictionaries from the environment
            
        Returns:
            List of eligible vehicle dictionaries
        """
        return [v for v in vehicles if not v['scheduled'] and v['sojourn_time'] >= 1.0]
    
    @abstractmethod
    def select_vehicles(self, vehicles, target_count=10):
        """
        Select vehicles for scheduling.
        
        Args:
            vehicles: List of vehicle dictionaries from the environment
            target_count: Maximum number of vehicles to select
            
        Returns:
            List of selected vehicle dictionaries
        """
        pass
