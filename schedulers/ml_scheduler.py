"""
Machine learning-based scheduler for MR-VFL.
"""

import torch
import numpy as np
from .base_scheduler import BaseScheduler

# Set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class GlobalState:
    """Global state for ML-based schedulers"""
    
    def __init__(self):
        """Initialize global state"""
        pass
    
    def to_tensor(self):
        """
        Convert global state to tensor.
        
        Returns:
            Tensor representation of global state
        """
        return torch.tensor([
            0.0,  # current_model_performance
            0.0,  # round_number
            0.0,  # elapsed_time
            0.0,  # scheduled_count
            10.0, # target_vehicle_count
            0.0   # performance_gap
        ], dtype=torch.float32).to(device)

class MLBasedScheduler(BaseScheduler):
    """
    Base class for machine learning-based schedulers.
    
    This class provides common functionality for schedulers that use ML models.
    """
    
    def __init__(self, model, name):
        """
        Initialize the ML-based scheduler.
        
        Args:
            model: ML model for vehicle selection
            name: Name of the scheduler
        """
        super().__init__(name)
        self.model = model
        self.global_state = GlobalState()
    
    def vehicle_to_tensor(self, vehicle):
        """
        Convert a vehicle dictionary to a tensor.
        
        Args:
            vehicle: Vehicle dictionary
            
        Returns:
            Tensor representation of the vehicle
        """
        return torch.tensor([
            vehicle.get('model_version', 0.5),
            vehicle['sojourn_time'],
            vehicle['computation_capacity'],
            vehicle['data_quality'],
            vehicle['channel_gain'],
            0  # Default vehicle type
        ], dtype=torch.float32).to(device)
    
    def select_vehicles(self, vehicles, target_count=10):
        """
        Select vehicles using the ML model.
        
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
        
        # Convert vehicles to tensors
        vehicle_tensors = [self.vehicle_to_tensor(v) for v in eligible_vehicles]
        
        # Create mask
        mask = torch.ones(len(eligible_vehicles)).to(device)
        
        # Get global state
        global_state_tensor = self.global_state.to_tensor()
        
        # Get selection probabilities
        with torch.no_grad():
            probs = self.model(vehicle_tensors, global_state_tensor, mask)
        
        # Select vehicles based on probabilities
        selected_indices = []
        remaining_indices = list(range(len(eligible_vehicles)))
        
        # Select up to target_count vehicles
        for _ in range(min(target_count, len(remaining_indices))):
            if not remaining_indices:
                break
            
            # Normalize probabilities for remaining vehicles
            remaining_probs = probs[remaining_indices]
            remaining_probs = remaining_probs / remaining_probs.sum()
            
            # Sample based on probabilities
            idx = np.random.choice(len(remaining_indices), p=remaining_probs.cpu().numpy())
            selected_idx = remaining_indices[idx]
            selected_indices.append(selected_idx)
            remaining_indices.remove(selected_idx)
        
        # Return selected vehicles
        return [eligible_vehicles[i] for i in selected_indices]
