"""
Dynamic Vehicular Federated Learning Environment.
"""

import numpy as np
from .vehicular_fl_env import VehicularFLEnv

class DynamicVehicularFLEnv(VehicularFLEnv):
    """
    Dynamic Vehicular Federated Learning Environment.
    
    This environment simulates a scenario with changing vehicle characteristics
    over time, representing different phases of a dynamic environment.
    """
    
    def __init__(self, vehicle_count=30, max_round=100, sync_limit=1000, phase_length=10, traffic_density=10):
        """
        Initialize the dynamic vehicular FL environment.
        
        Args:
            vehicle_count: Number of vehicles in the environment
            max_round: Maximum number of rounds for the simulation
            sync_limit: Synchronization time limit in seconds
            phase_length: Number of rounds per phase
            traffic_density: Traffic density (vehicles per km²), default 10
        """
        # Set phase attributes before calling parent constructor
        self.phase = 0
        self.phase_length = phase_length  # rounds per phase

        # Call parent constructor with specified parameters
        super().__init__(
            vehicle_count=vehicle_count,
            max_round=max_round,
            sync_limit=sync_limit,
            traffic_density=traffic_density,
            data_categories=10,
            learning_rate=0.01
        )

        # Generate vehicles with phase-specific characteristics
        self.vehicles = self._generate_vehicles()
    
    def reset(self):
        """
        Reset environment with phase tracking.
        
        Returns:
            Initial state representation
        """
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

        # Reset phase
        self.phase = 0

        # Generate vehicles with phase-specific characteristics
        self.vehicles = self._generate_vehicles()

        # Return state
        return self._get_state()
    
    def step(self, action):
        """
        Override step to update phase if needed.
        
        Args:
            action: Action to take, consisting of [vehicle_idx, alpha, scheduled_time, bandwidth]
            
        Returns:
            Tuple of (state, reward, done, info)
        """
        # Check if we need to change phase
        current_phase = self.current_round // self.phase_length
        if current_phase != self.phase:
            self.phase = current_phase
            print(f"    Environment changed to phase {self.phase + 1}")
            
            # Regenerate vehicles for the new phase
            self.vehicles = self._generate_vehicles()

        # Call parent step method
        return super().step(action)
    
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
    
    def _generate_vehicles(self):
        """
        Generate vehicles with characteristics based on current phase.
        
        Returns:
            List of vehicle dictionaries with phase-specific characteristics
        """
        vehicles = []

        # Determine current phase
        current_phase = self.current_round // self.phase_length
        if current_phase != self.phase:
            self.phase = current_phase
            print(f"    Environment changed to phase {self.phase + 1}")
            
        # Calculate expected number of vehicles based on traffic density and area
        area = np.pi * (self.area_radius ** 2)  # Area in km²
        expected_vehicles = int(self.traffic_density * area)
        
        # Use the vehicle count from initialization, but log if it differs from expected
        if expected_vehicles != self.vehicle_count:
            print(f"  Note: Expected {expected_vehicles} vehicles based on traffic density, using {self.vehicle_count} as specified")

        # Generate vehicles based on current phase
        for i in range(self.vehicle_count):
            # Randomly assign a road type (1-8)
            road_type = np.random.randint(1, self.road_types + 1)
            
            # Determine vehicle's distance from the server (in meters)
            # Vehicles are distributed within the area radius
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
            
            # Calculate data quality based on the vehicle's characteristics
            # Higher quality for vehicles with better computation and channel conditions
            base_data_quality = np.random.uniform(0.1, 1.0)
            
            # Adjust data quality based on the number of data categories
            # Vehicles with more diverse data categories have higher quality
            data_categories_ratio = np.random.randint(1, self.data_categories + 1) / self.data_categories
            
            # Base vehicle attributes
            vehicle = {
                'id': i,
                'data_size': np.random.randint(100, 1000),
                'arrival_time': np.random.uniform(0, self.sync_limit/2),
                'departure_time': 0,
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
                'comm_time': np.random.uniform(5, 20),
                'training_time': np.random.uniform(10, 50),
                'predicted_arrival': 0  # Will be set below
            }

            # Phase 0: Normal/Balanced conditions
            # In this phase, vehicles have balanced characteristics with moderate values
            # This represents a typical urban environment with a mix of vehicle types
            if self.phase == 0:
                # Balanced computation capacity (mid-range)
                vehicle['computation_capacity'] = np.random.uniform(0.5, 1.0)

                # Balanced data quality (mid-range)
                vehicle['data_quality'] = base_data_quality * data_categories_ratio

                # Moderate sojourn times (vehicles stay for a reasonable duration)
                vehicle['sojourn_time'] = np.random.uniform(100, 1000)

                # Good connectivity (stable network conditions)
                vehicle['channel_gain'] = normalized_channel_gain

                # Moderate communication and training times
                vehicle['comm_time'] = self._calculate_comm_time(bandwidth_mhz, normalized_channel_gain)
                vehicle['training_time'] = self._calculate_training_time(vehicle['computation_capacity'], vehicle['data_quality'])

                # Predictable arrival patterns
                prediction_error = np.random.uniform(-5, 5)  # Small prediction error

                # Set phase-specific metadata
                vehicle['phase_type'] = 'balanced'
                vehicle['environment'] = 'urban'

            # Phase 1: High compute, low quality vehicles
            # This phase represents a scenario with powerful vehicles (e.g., luxury cars, buses)
            # that have strong computing capabilities but collect lower quality data
            # (e.g., due to sensor limitations or environmental factors)
            elif self.phase == 1:
                # High computation capacity (powerful onboard computers)
                vehicle['computation_capacity'] = np.random.uniform(0.7, 1.5)

                # Low data quality (poor sensors or challenging conditions)
                vehicle['data_quality'] = base_data_quality * 0.5  # Reduce data quality

                # Shorter stays (high mobility, less time in the area)
                vehicle['sojourn_time'] = np.random.uniform(50, 500)

                # Good connectivity (premium network access)
                vehicle['channel_gain'] = normalized_channel_gain * 1.2  # Better connectivity
                vehicle['channel_gain'] = min(1.0, vehicle['channel_gain'])  # Cap at 1.0

                # Fast communication but longer training due to data issues
                vehicle['comm_time'] = self._calculate_comm_time(bandwidth_mhz * 1.2, vehicle['channel_gain'])  # Better bandwidth
                vehicle['training_time'] = self._calculate_training_time(vehicle['computation_capacity'], vehicle['data_quality'])

                # Less predictable arrivals
                prediction_error = np.random.uniform(-15, 15)  # Larger prediction error

                # Set phase-specific metadata
                vehicle['phase_type'] = 'high_compute_low_quality'
                vehicle['environment'] = 'highway'

            # Phase 2: Low compute, high quality vehicles
            # This phase represents a scenario with resource-constrained vehicles
            # (e.g., older vehicles, small IoT devices) that collect high-quality data
            # (e.g., due to strategic positioning or specialized sensors)
            elif self.phase == 2:
                # Low computation capacity (limited onboard resources)
                vehicle['computation_capacity'] = np.random.uniform(0.1, 0.5)

                # High data quality (specialized sensors or optimal positioning)
                vehicle['data_quality'] = base_data_quality * 1.5  # Increase data quality
                vehicle['data_quality'] = min(1.0, vehicle['data_quality'])  # Cap at 1.0

                # Longer stays (stationary or slow-moving vehicles)
                vehicle['sojourn_time'] = np.random.uniform(200, 1500)

                # Good connectivity (stable positions allow good connections)
                vehicle['channel_gain'] = normalized_channel_gain

                # Slower communication due to limited hardware
                vehicle['comm_time'] = self._calculate_comm_time(bandwidth_mhz * 0.8, vehicle['channel_gain'])  # Reduced bandwidth
                vehicle['training_time'] = self._calculate_training_time(vehicle['computation_capacity'], vehicle['data_quality'])

                # Highly predictable arrivals (fixed routes or schedules)
                prediction_error = np.random.uniform(-3, 3)  # Very small prediction error

                # Set phase-specific metadata
                vehicle['phase_type'] = 'low_compute_high_quality'
                vehicle['environment'] = 'urban_fixed'

            # Phase 3: Mixed vehicles with connectivity issues
            # This phase represents a challenging environment with diverse vehicles
            # experiencing poor network conditions (e.g., rural areas, bad weather)
            elif self.phase == 3:
                # Mixed computation capacity
                vehicle['computation_capacity'] = np.random.uniform(0.3, 0.8)

                # Mixed data quality
                vehicle['data_quality'] = base_data_quality

                # Variable sojourn times
                vehicle['sojourn_time'] = np.random.uniform(100, 1000)

                # Poor connectivity (network issues, interference, distance)
                vehicle['channel_gain'] = normalized_channel_gain * 0.5  # Worse connectivity

                # Slow and unreliable communication
                vehicle['comm_time'] = self._calculate_comm_time(bandwidth_mhz * 0.6, vehicle['channel_gain'])  # Reduced bandwidth
                vehicle['training_time'] = self._calculate_training_time(vehicle['computation_capacity'], vehicle['data_quality'])

                # Unpredictable arrivals and departures
                prediction_error = np.random.uniform(-25, 25)  # Very large prediction error

                # Set phase-specific metadata
                vehicle['phase_type'] = 'connectivity_challenged'
                vehicle['environment'] = 'rural_adverse'

            # Set departure time
            vehicle['departure_time'] = vehicle['arrival_time'] + vehicle['sojourn_time']
            
            # Set predicted arrival time
            vehicle['predicted_arrival'] = max(0, vehicle['arrival_time'] + prediction_error)
            
            vehicles.append(vehicle)
        
        return vehicles
