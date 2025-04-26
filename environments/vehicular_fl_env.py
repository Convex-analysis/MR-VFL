"""
Vehicular Federated Learning Environment.
"""

import numpy as np
import torch

class VehicularFLEnv:
    """
    Vehicular Federated Learning Environment.
    
    This environment simulates a vehicular federated learning scenario where
    vehicles arrive and depart dynamically, and a scheduler must select vehicles
    for participation in federated learning rounds.
    """
    
    def __init__(self, vehicle_count=30, max_round=100, sync_limit=1000, traffic_density=10, data_categories=10, learning_rate=0.01):
        """
        Initialize the Vehicular Federated Learning Environment.
        
        Args:
            vehicle_count: Number of vehicles in the environment
            max_round: Maximum number of rounds for the simulation
            sync_limit: Synchronization time limit in seconds
            traffic_density: Traffic density (vehicles per km²), default 10
            data_categories: Number of data categories, default 10
            learning_rate: Learning rate for model updates, default 0.01
        """
        self.vehicle_count = vehicle_count
        self.max_round = max_round
        self.sync_limit = sync_limit
        self.current_round = 0
        self.selection_history = np.zeros(vehicle_count)  # γ_v^t tracking (selection frequency)
        self.fairness_threshold = 50  # Γ_t
        
        # Mobility parameters
        self.traffic_density = traffic_density  # Vehicles per km²
        self.area_radius = 2.5  # Radius of selected area in km
        self.road_types = 8  # Total road types
        
        # Communication parameters
        self.server_bandwidth = 50  # Server bandwidth in MHz
        self.server_tx_power = 30  # Server transmission power in dBm
        self.path_loss_constant = -40  # Path loss constant in dB
        self.path_loss_factor = 4  # Large-scale path loss factor
        self.reference_distance = 1  # Reference distance in meters
        self.noise_power_density = -174  # Noise power spectral density in dBm/Hz
        
        # Learning parameters
        self.data_categories = data_categories  # Number of data categories
        self.learning_rate = learning_rate  # Learning rate for model updates
        
        # Reward function weights
        self.c1 = 0.3  # Weight for time utilization and scheduling completion
        self.c2 = 0.5  # Weight for accuracy improvement
        self.c3 = 0.2  # Weight for fairness penalty

        # State variables
        self.vehicles = []
        self.scheduled_vehicles = []  # V^r_t: currently scheduled vehicle set
        self.current_model_performance = 0.0
        self.elapsed_time = 0.0
        self.scheduled_count = 0
        self.available_bandwidth = self.server_bandwidth / 50.0  # B^r_t: normalized available bandwidth
        self.fl_workload = 1.0  # w(r): FL workload for current round
        
        # Time slots
        self.time_slots = list(range(0, sync_limit, 10))  # T^r_t: available time slots
        
        # Performance tracking
        self.performance_history = []
        self.reward_history = []
        
        # Initialize vehicle states - only if this is the base class
        # Subclasses should handle their own initialization
        if self.__class__.__name__ == 'VehicularFLEnv':
            self.vehicles = self._generate_vehicles()
    
    def reset(self):
        """
        Reset the environment to initial state.
        
        Returns:
            Initial state representation
        """
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

        # Initialize vehicle states - only if this is the base class
        # Subclasses should handle their own initialization
        if self.__class__.__name__ == 'VehicularFLEnv':
            self.vehicles = self._generate_vehicles()

        # Return state (for backward compatibility, return tensor representation)
        return self._get_state()
    
    def _generate_vehicles(self):
        """
        Generate vehicles with random attributes based on the environment parameters.
        
        This method creates vehicles with realistic communication and mobility parameters
        according to the specified traffic density and area characteristics.
        
        Returns:
            List of vehicle dictionaries with all required attributes
        """
        vehicles = []
        
        # Calculate expected number of vehicles based on traffic density and area
        area = np.pi * (self.area_radius ** 2)  # Area in km²
        expected_vehicles = int(self.traffic_density * area)
        
        # Use the vehicle count from initialization, but log if it differs from expected
        if expected_vehicles != self.vehicle_count:
            print(f"Note: Expected {expected_vehicles} vehicles based on traffic density, using {self.vehicle_count} as specified")
        
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
            # PL(d) = PL(d0) + 10 * n * log10(d/d0) + X_sigma
            # where PL(d0) is the path loss at reference distance d0,
            # n is the path loss exponent, and X_sigma is the shadowing effect
            
            # Path loss at reference distance (dB)
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
            computation_capacity = np.random.uniform(0.5, 2.0)
            base_data_quality = np.random.uniform(0.1, 1.0)
            
            # Adjust data quality based on the number of data categories
            # Vehicles with more diverse data categories have higher quality
            data_categories_ratio = np.random.randint(1, self.data_categories + 1) / self.data_categories
            data_quality = base_data_quality * data_categories_ratio
            
            # Basic vehicle attributes
            vehicle = {
                'id': i,
                'computation_capacity': computation_capacity,
                'data_quality': data_quality,  # D_v: data quality
                'data_size': np.random.randint(100, 1000),
                'sojourn_time': np.random.uniform(100, 2000),
                'arrival_time': np.random.uniform(0, self.sync_limit/2),  # t^ar_v: arrival time
                'departure_time': 0,  # t^dp_v: departure time
                'channel_gain': normalized_channel_gain,
                'scheduled': False,
                
                # Communication parameters
                'bandwidth_mhz': bandwidth_mhz,  # Vehicle bandwidth in MHz
                'tx_power_dbm': tx_power_dbm,  # Transmission power in dBm
                'distance': distance,  # Distance from server in meters
                'road_type': road_type,  # Road type (1-8)
                
                # Additional vehicle status attributes (ST^r_t)
                'selection_count': 0,  # sl_v: times selected by edge server
                'participation_count': 0,  # cumulative number of times participated in training
                'comm_time': self._calculate_comm_time(bandwidth_mhz, normalized_channel_gain),  # Communication time
                'training_time': self._calculate_training_time(computation_capacity, data_quality),  # Training time
                'predicted_arrival': 0  # t^ar_v: arrival time predicted by edge server
            }
            
            # Calculate departure time
            vehicle['departure_time'] = vehicle['arrival_time'] + vehicle['sojourn_time']
            
            # Set predicted arrival time (could be slightly different from actual arrival time)
            prediction_error = np.random.uniform(-10, 10)  # Small prediction error
            vehicle['predicted_arrival'] = max(0, vehicle['arrival_time'] + prediction_error)
            
            vehicles.append(vehicle)
            
        return vehicles
        
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
    
    def _is_vehicle_available(self, vehicle):
        """
        Check if a vehicle is available for scheduling.
        
        Args:
            vehicle: Vehicle dictionary
            
        Returns:
            True if the vehicle is available, False otherwise
        """
        # Vehicle is available if:
        # 1. It has not been scheduled yet
        # 2. It has arrived (current time >= arrival time)
        # 3. It has not departed (current time < departure time)
        # 4. It has sufficient sojourn time (sojourn_time >= 1.0)
        return (not vehicle['scheduled'] and
                self.elapsed_time >= vehicle['arrival_time'] and
                self.elapsed_time < vehicle['departure_time'] and
                vehicle['sojourn_time'] >= 1.0)
    
    def step(self, action):
        """
        Take a step in the environment.
        
        Args:
            action: Action to take, consisting of [vehicle_idx, alpha, scheduled_time, bandwidth]
            
        Returns:
            Tuple of (state, reward, done, info)
        """
        # Parse action
        vehicle_idx, alpha, scheduled_time, bandwidth = action
        
        # Get the selected vehicle
        vehicle = self.vehicles[vehicle_idx]
        
        # Check if vehicle is available
        if not self._is_vehicle_available(vehicle):
            # If vehicle is not available, return negative reward
            return self._get_state(), -10.0, False, {}
        
        # Mark vehicle as scheduled
        vehicle['scheduled'] = True
        vehicle['selection_count'] += 1
        vehicle['participation_count'] += 1
        
        # Update selection history
        self.selection_history[vehicle_idx] += 1
        
        # Add vehicle to scheduled set
        self.scheduled_vehicles.append(vehicle)
        self.scheduled_count += 1
        
        # Update elapsed time
        self.elapsed_time = max(self.elapsed_time, scheduled_time)
        
        # Calculate accuracy improvement based on scheduled vehicle
        # Use learning rate to determine the improvement rate
        base_improvement = vehicle['data_quality'] * alpha
        
        # Adjust improvement based on learning rate and data categories
        data_categories_factor = min(1.0, vehicle.get('data_categories_ratio', 0.5) * 1.5)
        accuracy_improvement = self.learning_rate * base_improvement * data_categories_factor
        
        self.current_model_performance += accuracy_improvement
        self.current_model_performance = min(1.0, self.current_model_performance)
        
        # Calculate reward
        reward = self._calculate_reward(vehicle_idx, alpha, scheduled_time, bandwidth)
        
        # Check if round is complete
        # Round is complete when:
        # 1. We've scheduled enough vehicles (scheduled_count >= target_count)
        # 2. Or we've reached the synchronization limit (elapsed_time >= sync_limit)
        # 3. Or all vehicles have been scheduled
        done = False
        target_count = min(10, self.vehicle_count)  # Target number of vehicles to schedule
        
        if (self.scheduled_count >= target_count or
            self.elapsed_time >= self.sync_limit or
            all(v['scheduled'] for v in self.vehicles)):
            # Round is complete
            done = True
            
            # Record performance
            self.performance_history.append(self.current_model_performance)
            
            # Prepare for next round
            if self.current_round < self.max_round - 1:
                self.current_round += 1
                
                # Reset scheduled vehicles
                for v in self.vehicles:
                    v['scheduled'] = False
                
                self.scheduled_vehicles = []
                self.scheduled_count = 0
                
                # Keep elapsed time and performance for next round
        
        # Record reward
        self.reward_history.append(reward)
        
        # Return state, reward, done, info
        return self._get_state(), reward, done, {}
    
    def _calculate_reward(self, vehicle_idx, alpha, scheduled_time, bandwidth):
        """
        Calculate reward based on scheduled vehicle and parameters
        
        Enhanced reward function:
        r(t) = x_v * (c1 * (V^r_t/N^r) * (T_{syn}^r - T_{v}^s) + c2 * θ_{acc,v} - c3 * (e^{γ_v^t} - 1)) * η_v
        
        where:
        - x_v is the selection indicator (always 1 for selected vehicles)
        - V^r_t is the number of scheduled vehicles
        - N^r is the total number of vehicles
        - T_{syn}^r is the synchronization limit
        - T_{v}^s is the scheduled time
        - θ_{acc,v} is the accuracy improvement
        - γ_v^t is the selection frequency
        - η_v is the communication efficiency factor
        - c1, c2, c3 are weights
        """
        vehicle = self.vehicles[vehicle_idx]
        
        # Calculate components of the reward function
        
        # 1. Time utilization and scheduling completion rate
        # (V^r_t/N^r) * (T_{syn}^r - T_{v}^s)
        scheduled_ratio = (len(self.scheduled_vehicles) + 1) / self.vehicle_count  # +1 for current vehicle
        time_efficiency = self.sync_limit - scheduled_time
        time_component = scheduled_ratio * time_efficiency
        
        # 2. Accuracy improvement (θ_{acc,v})
        # Based on data quality, amplification factor, and learning rate
        # Higher learning rate leads to faster accuracy improvement
        accuracy_improvement = vehicle['data_quality'] * alpha * (1 + self.learning_rate * 10)
        
        # Adjust accuracy improvement based on data categories
        # More data categories can lead to better model generalization
        data_categories_factor = min(1.0, vehicle.get('data_categories_ratio', 0.5) * 1.5)
        accuracy_improvement *= data_categories_factor
        
        # 3. Fairness penalty (e^{γ_v^t} - 1)
        # Based on selection frequency
        gamma_v = self.selection_history[vehicle_idx]
        fairness_penalty = np.exp(min(5, gamma_v)) - 1  # Limit exponential growth
        
        # 4. Communication efficiency factor (η_v)
        # Calculate achievable data rate using Shannon's formula
        # R = B * log2(1 + SNR)
        
        # Get vehicle's bandwidth in Hz
        vehicle_bandwidth_hz = vehicle['bandwidth_mhz'] * 1e6
        
        # Calculate signal power at receiver (using path loss model)
        # P_rx = P_tx * channel_gain
        tx_power_mw = 10 ** (vehicle['tx_power_dbm'] / 10)  # Convert dBm to mW
        rx_power_mw = tx_power_mw * vehicle['channel_gain']
        
        # Calculate noise power
        # P_noise = N0 * B
        noise_power_density_mw = 10 ** (self.noise_power_density / 10)  # Convert dBm/Hz to mW/Hz
        noise_power_mw = noise_power_density_mw * vehicle_bandwidth_hz
        
        # Calculate SNR
        snr = rx_power_mw / max(1e-10, noise_power_mw)  # Avoid division by zero
        
        # Calculate achievable data rate (bits/s)
        data_rate = vehicle_bandwidth_hz * np.log2(1 + snr)
        
        # Normalize data rate for reward calculation
        # Higher data rate means better communication efficiency
        max_data_rate = 30e6 * np.log2(1 + 1000)  # Maximum possible data rate (30 MHz, SNR=1000)
        normalized_data_rate = min(1.0, data_rate / max_data_rate)
        
        # Communication efficiency factor
        comm_factor = 1.0 + normalized_data_rate
        
        # Adjust for allocated bandwidth
        if bandwidth > 0:
            # Better bandwidth allocation improves communication efficiency
            bandwidth_factor = bandwidth * vehicle['channel_gain']
            comm_factor *= (1.0 + 0.2 * bandwidth_factor)
        
        # Combined reward with weights
        reward = (self.c1 * time_component + self.c2 * accuracy_improvement - self.c3 * fairness_penalty) * comm_factor
        
        # Limit minimum reward to prevent extremely negative values
        reward = max(-100, reward)
        
        return reward
    
    def _get_state(self):
        """
        Get the current state of the environment.
        
        Returns:
            State representation as a tensor
        """
        # For backward compatibility, return a tensor representation
        # In practice, the state is the entire environment
        state = torch.tensor([
            self.current_model_performance,
            float(self.current_round) / self.max_round,
            self.elapsed_time / self.sync_limit,
            self.scheduled_count / self.vehicle_count,
            10.0 / self.vehicle_count,  # Target count normalized
            1.0 - self.current_model_performance  # Performance gap
        ], dtype=torch.float32)
        
        return state
