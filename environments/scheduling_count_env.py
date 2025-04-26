"""
Scheduling Count Environment for MR-VFL.
"""

import numpy as np
from .vehicular_fl_env import VehicularFLEnv

class SchedulingCountEnv(VehicularFLEnv):
    """
    Scheduling Count Environment for MR-VFL.
    
    This environment tracks the number of vehicles successfully scheduled by
    different schedulers, with a focus on vehicle quality and scheduling success rate.
    """
    
    def __init__(self, vehicle_count=100, max_round=20, sync_limit=1000, traffic_density=10):
        """
        Initialize the scheduling count environment.
        
        Args:
            vehicle_count: Number of vehicles in the environment
            max_round: Maximum number of rounds for the simulation
            sync_limit: Synchronization time limit in seconds
            traffic_density: Traffic density (vehicles per km²), default 10
        """
        # Initialize tracking metrics
        self.scheduled_vehicle_ids = set()
        self.available_vehicle_count = 0
        self.high_quality_scheduled = 0  # Vehicles with data_quality > 0.7
        self.low_quality_scheduled = 0   # Vehicles with data_quality <= 0.3
        self.scheduling_attempts = 0
        self.scheduling_successes = 0

        # Call parent constructor with specified parameters
        super().__init__(
            vehicle_count=vehicle_count,
            max_round=max_round,
            sync_limit=sync_limit,
            traffic_density=traffic_density,
            data_categories=10,
            learning_rate=0.01
        )

        # Generate vehicles with diverse characteristics
        self.vehicles = self._generate_vehicles()
    
    def reset(self):
        """
        Reset environment and tracking metrics.
        
        Returns:
            Initial state representation
        """
        # Reset tracking metrics
        self.scheduled_vehicle_ids = set()
        self.available_vehicle_count = 0
        self.high_quality_scheduled = 0
        self.low_quality_scheduled = 0
        self.scheduling_attempts = 0
        self.scheduling_successes = 0
        
        # Reset base environment state
        self.current_round = 0
        self.selection_history = np.zeros(self.vehicle_count)
        self.current_model_performance = 0.0
        self.elapsed_time = 0.0
        self.scheduled_count = 0
        self.scheduled_vehicles = []
        self.available_bandwidth = self.server_bandwidth / 50.0  # Normalized available bandwidth
        self.fl_workload = 1.0
        self.time_slots = list(range(0, self.sync_limit, 10))
        self.performance_history = []
        self.reward_history = []
        
        # Generate vehicles with diverse characteristics
        self.vehicles = self._generate_vehicles()
        
        # Return state
        return self._get_state()
    
    def _calculate_comm_time(self, bandwidth_mhz, channel_gain):
        """
        Calculate communication time based on bandwidth and channel conditions.
        
        Args:
            bandwidth_mhz: Vehicle bandwidth in MHz
            channel_gain: Normalized channel gain
            
        Returns:
            Communication time in seconds
        """
        # Base communication time (inversely proportional to bandwidth)
        base_comm_time = 50 / bandwidth_mhz
        
        # Adjust for channel conditions (worse channel = longer time)
        channel_factor = 1 + (1 - channel_gain) * 2
        
        # Final communication time
        comm_time = base_comm_time * channel_factor
        
        # Ensure reasonable bounds
        return max(5, min(50, comm_time))
        
    def _calculate_training_time(self, computation_capacity, data_quality):
        """
        Calculate training time based on computation capacity and data quality.
        
        Args:
            computation_capacity: Vehicle's computation capacity
            data_quality: Data quality of the vehicle
            
        Returns:
            Training time in seconds
        """
        # Base training time (inversely proportional to computation capacity)
        base_training_time = 30 / computation_capacity
        
        # Adjust for data quality (higher quality = more processing)
        quality_factor = 1 + data_quality
        
        # Final training time
        training_time = base_training_time * quality_factor
        
        # Ensure reasonable bounds
        return max(10, min(100, training_time))
    
    def step(self, action):
        """
        Override step to track scheduling statistics.
        
        Args:
            action: Action to take, consisting of [vehicle_idx, alpha, scheduled_time, bandwidth]
            
        Returns:
            Tuple of (state, reward, done, info)
        """
        # Track scheduling attempt
        self.scheduling_attempts += 1

        # Get vehicle index from action
        vehicle_idx = action[0]  # First element is the vehicle index

        # Check if vehicle is available
        vehicle = self.vehicles[vehicle_idx]
        if self._is_vehicle_available(vehicle):
            # Track unique vehicles scheduled
            self.scheduled_vehicle_ids.add(vehicle_idx)

            # Track quality of scheduled vehicles
            if vehicle['data_quality'] > 0.7:
                self.high_quality_scheduled += 1
            elif vehicle['data_quality'] <= 0.3:
                self.low_quality_scheduled += 1

            # Track successful scheduling
            self.scheduling_successes += 1

        # Call parent step method
        return super().step(action)
    
    def _generate_vehicles(self):
        """
        Generate vehicles with diverse characteristics.
        
        Returns:
            List of vehicle dictionaries with diverse characteristics
        """
        vehicles = []
        
        # Calculate expected number of vehicles based on traffic density and area
        area = np.pi * (self.area_radius ** 2)  # Area in km²
        expected_vehicles = int(self.traffic_density * area)
        
        # Use the vehicle count from initialization, but log if it differs from expected
        if expected_vehicles != self.vehicle_count:
            print(f"  Note: Expected {expected_vehicles} vehicles based on traffic density, using {self.vehicle_count} as specified")

        # Generate vehicles with diverse characteristics
        for i in range(self.vehicle_count):
            # Determine vehicle type (high quality, low quality, or medium)
            vehicle_type = np.random.choice(['high', 'medium', 'low'], p=[0.3, 0.4, 0.3])
            
            # Randomly assign a road type (1-8)
            road_type = np.random.randint(1, self.road_types + 1)
            
            # Determine vehicle's distance from the server (in meters)
            distance = np.random.uniform(0, self.area_radius * 1000)  # Convert km to meters
            
            # Generate vehicle bandwidth (10-30 MHz)
            bandwidth_mhz = np.random.uniform(10, 30)
            
            # Generate transmission power (23-30 dBm)
            tx_power_dbm = np.random.uniform(23, 30)
            
            # Calculate channel gain using path loss model
            pl_d0 = self.path_loss_constant
            
            # Path loss at distance d (dB)
            if distance > 0:
                pl_d = pl_d0 + 10 * self.path_loss_factor * np.log10(distance / self.reference_distance)
            else:
                pl_d = pl_d0
                
            # Add shadowing effect (dB)
            shadowing = np.random.normal(0, 8)  # Standard deviation of 8 dB
            
            # Total path loss (dB)
            total_path_loss = pl_d + shadowing
            
            # Convert path loss to linear scale
            path_loss_linear = 10 ** (total_path_loss / 10)
            
            # Calculate channel gain (inverse of path loss)
            channel_gain = 1 / max(1, path_loss_linear)
            
            # Normalize channel gain to [0, 1] for compatibility with existing code
            normalized_channel_gain = min(1.0, channel_gain * 1e6)
            
            # Adjust data categories ratio based on vehicle type
            if vehicle_type == 'high':
                # High-quality vehicles have more data categories
                data_categories_ratio = np.random.uniform(0.7, 1.0)
                # High-quality vehicles
                computation_capacity = np.random.uniform(0.5, 1.0)
                sojourn_time = np.random.uniform(300, 1000)  # Longer stays
                # Adjust channel gain based on vehicle type
                channel_gain = normalized_channel_gain * 1.2  # Better connectivity
                channel_gain = min(1.0, channel_gain)  # Cap at 1.0
            elif vehicle_type == 'low':
                # Low-quality vehicles have fewer data categories
                data_categories_ratio = np.random.uniform(0.1, 0.3)
                # Low-quality vehicles
                computation_capacity = np.random.uniform(0.1, 0.5)
                sojourn_time = np.random.uniform(300, 1000)   # Shorter stays
                # Adjust channel gain based on vehicle type
                channel_gain = normalized_channel_gain * 0.6  # Worse connectivity
            else:
                # Medium-quality vehicles have moderate data categories
                data_categories_ratio = np.random.uniform(0.3, 0.7)
                # Medium-quality vehicles
                computation_capacity = np.random.uniform(0.3, 0.8)
                sojourn_time = np.random.uniform(300, 1000)   # Medium stays
                # Use the calculated channel gain
                channel_gain = normalized_channel_gain
            
            # Calculate data quality based on data categories ratio
            data_quality = np.random.uniform(0.1, 1.0) * data_categories_ratio

            # Base vehicle attributes
            vehicle = {
                'id': i,
                'data_quality': data_quality,
                'computation_capacity': computation_capacity,
                'data_size': np.random.randint(100, 1000),
                'sojourn_time': sojourn_time,
                'arrival_time': np.random.uniform(0, self.sync_limit * 0.7),
                'departure_time': 0,
                'channel_gain': channel_gain,
                'scheduled': False,
                
                # Communication parameters
                'bandwidth_mhz': bandwidth_mhz,
                'tx_power_dbm': tx_power_dbm,
                'distance': distance,
                'road_type': road_type,
                'data_categories_ratio': data_categories_ratio,

                # Additional attributes required by the updated environment
                'selection_count': 0,
                'participation_count': 0,
                'comm_time': self._calculate_comm_time(bandwidth_mhz, channel_gain),
                'training_time': self._calculate_training_time(computation_capacity, data_quality),
                'predicted_arrival': 0,  # Will be set below
                'vehicle_type': vehicle_type  # Store vehicle type for analysis
            }
            
            # Set departure time
            vehicle['departure_time'] = vehicle['arrival_time'] + vehicle['sojourn_time']
            
            # Set predicted arrival time
            prediction_error = np.random.uniform(-10, 10)
            vehicle['predicted_arrival'] = max(0, vehicle['arrival_time'] + prediction_error)
            
            vehicles.append(vehicle)
            
        return vehicles
    
    def get_scheduling_stats(self):
        """
        Get scheduling statistics.
        
        Returns:
            Dictionary of scheduling statistics
        """
        return {
            'unique_vehicles_scheduled': len(self.scheduled_vehicle_ids),
            'high_quality_scheduled': self.high_quality_scheduled,
            'low_quality_scheduled': self.low_quality_scheduled,
            'scheduling_success_rate': self.scheduling_successes / max(1, self.scheduling_attempts),
            'total_attempts': self.scheduling_attempts,
            'total_successes': self.scheduling_successes
        }
