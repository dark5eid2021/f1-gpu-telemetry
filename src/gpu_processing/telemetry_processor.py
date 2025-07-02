"""
GPU-accelerated telemetry processing for F1 data
Uses CUDA kernels for real-time feature engineering
"""

import numpy as np
import cupy as cp
from numba import cuda
import logging
from typing import List, Dict, Any
from dataclasses import dataclass
import time


@dataclass
class TelemetryData:
    """F1 Telemetry data structure"""
    timestamp: float
    driver_id: int
    speed: float
    throttle: float
    brake: float
    gear: int
    drs: bool
    tire_temp_fl: float
    tire_temp_fr: float
    tire_temp_rl: float
    tire_temp_rr: float
    fuel_load: float
    lap_number: int
    sector: int


@cuda.jit
def process_telemetry_kernel(telemetry_data, features_out, batch_size):
    """
    CUDA kernel for real-time telemetry feature engineering
    
    Processes tire degradation, fuel efficiency, and performance metrics
    Each thread processes one telemetry sample
    
    Args:
        telemetry_data: Input telemetry array [batch_size, num_features]
        features_out: Output features array [batch_size, 3]
        batch_size: Number of samples to process
    """
    idx = cuda.grid(1)
    
    if idx < batch_size:
        # Extract telemetry values for this sample
        speed = telemetry_data[idx, 0]
        throttle = telemetry_data[idx, 1]
        brake = telemetry_data[idx, 2]
        
        # Calculate average tire temperature
        tire_temp_avg = (telemetry_data[idx, 4] + telemetry_data[idx, 5] + 
                        telemetry_data[idx, 6] + telemetry_data[idx, 7]) / 4.0
        
        fuel_load = telemetry_data[idx, 8]
        
        # Feature engineering on GPU
        
        # 1. Tire degradation index (0-1, higher = more degraded)
        # Based on tire temperature relative to optimal range (80-120°C)
        optimal_temp = 100.0
        temp_deviation = abs(tire_temp_avg - optimal_temp)
        tire_degradation = min(1.0, max(0.0, temp_deviation / 40.0))
        
        # 2. Fuel efficiency (speed per fuel unit)
        # Higher values indicate better fuel efficiency
        fuel_efficiency = speed / max(fuel_load, 1.0) if fuel_load > 0 else 0.0
        
        # 3. Performance index combining multiple factors
        # Considers speed, throttle application, braking, and tire condition
        throttle_factor = throttle / 100.0  # Normalize to 0-1
        brake_factor = 1.0 - (brake / 100.0)  # Invert brake (less braking = better)
        tire_factor = 1.0 - tire_degradation  # Better tires = higher factor
        
        performance_idx = (speed * throttle_factor * brake_factor * tire_factor) / 300.0
        performance_idx = min(1.0, max(0.0, performance_idx))  # Normalize to 0-1
        
        # Store computed features
        features_out[idx, 0] = tire_degradation
        features_out[idx, 1] = fuel_efficiency
        features_out[idx, 2] = performance_idx


@cuda.jit
def calculate_sector_times_kernel(telemetry_data, sector_times_out, batch_size):
    """
    CUDA kernel for calculating sector performance metrics
    
    Args:
        telemetry_data: Input telemetry array [batch_size, num_features]
        sector_times_out: Output sector metrics [batch_size, 3]
        batch_size: Number of samples to process
    """
    idx = cuda.grid(1)
    
    if idx < batch_size:
        speed = telemetry_data[idx, 0]
        throttle = telemetry_data[idx, 1]
        brake = telemetry_data[idx, 2]
        sector = int(telemetry_data[idx, 9])  # Current sector (1, 2, or 3)
        
        # Calculate sector-specific metrics
        if sector == 1:
            # Sector 1: Focus on acceleration and throttle application
            sector_performance = (speed * throttle) / 100.0
        elif sector == 2:
            # Sector 2: Focus on cornering and balance
            cornering_factor = (100.0 - brake) * throttle / 10000.0
            sector_performance = speed * cornering_factor
        else:  # Sector 3
            # Sector 3: Focus on top speed and DRS usage
            sector_performance = speed * (1.0 + (telemetry_data[idx, 3] * 0.1))  # DRS bonus
        
        # Store in appropriate sector slot
        if 1 <= sector <= 3:
            sector_times_out[idx, sector - 1] = sector_performance


class GPUTelemetryProcessor:
    """GPU-accelerated telemetry processing pipeline"""
    
    def __init__(self, batch_size: int = 1024, device_id: int = 0):
        """
        Initialize GPU telemetry processor
        
        Args:
            batch_size: Number of samples to process in each batch
            device_id: CUDA device ID to use
        """
        self.batch_size = batch_size
        self.device_id = device_id
        self.logger = logging.getLogger(__name__)
        
        # Set CUDA device
        if cp.cuda.is_available():
            cp.cuda.Device(device_id).use()
            self.gpu_memory_pool = cp.get_default_memory_pool()
            self.logger.info(f"✅ GPU Telemetry Processor initialized on device {device_id}")
        else:
            self.logger.warning("⚠️ CUDA not available, falling back to CPU processing")
        
        # Performance metrics
        self.processing_times = []
        self.samples_processed = 0
    
    def process_batch(self, telemetry_batch: List[TelemetryData]) -> Dict[str, cp.ndarray]:
        """
        Process a batch of telemetry data on GPU
        
        Args:
            telemetry_batch: List of TelemetryData objects
            
        Returns:
            Dictionary containing processed features and metrics
        """
        start_time = time.time()
        
        if not telemetry_batch:
            return {}
        
        try:
            # Convert telemetry data to NumPy array
            data_array = self._telemetry_to_array(telemetry_batch)
            
            if not cp.cuda.is_available():
                # CPU fallback
                return self._process_on_cpu(data_array)
            
            # Transfer to GPU
            gpu_data = cp.asarray(data_array, dtype=cp.float32)
            batch_size = len(telemetry_batch)
            
            # Allocate output arrays
            features_out = cp.zeros((batch_size, 3), dtype=cp.float32)
            sector_times_out = cp.zeros((batch_size, 3), dtype=cp.float32)
            
            # Configure CUDA kernel launch parameters
            threads_per_block = 256
            blocks_per_grid = (batch_size + threads_per_block - 1) // threads_per_block
            
            # Launch CUDA kernels
            process_telemetry_kernel[blocks_per_grid, threads_per_block](
                gpu_data, features_out, batch_size
            )
            
            calculate_sector_times_kernel[blocks_per_grid, threads_per_block](
                gpu_data, sector_times_out, batch_size
            )
            
            # Synchronize GPU
            cp.cuda.Stream.null.synchronize()
            
            # Track performance
            processing_time = time.time() - start_time
            self.processing_times.append(processing_time)
            self.samples_processed += batch_size
            
            # Log performance metrics periodically
            if len(self.processing_times) % 100 == 0:
                avg_time = np.mean(self.processing_times[-100:])
                throughput = batch_size / avg_time
                self.logger.info(f"⚡ GPU Processing: {throughput:.0f} samples/sec, "
                               f"Latency: {processing_time*1000:.1f}ms")
            
            return {
                'features': features_out,
                'sector_times': sector_times_out,
                'processing_time': processing_time,
                'batch_size': batch_size
            }
            
        except Exception as e:
            self.logger.error(f"❌ Error processing telemetry batch: {e}")
            return {}
    
    def _telemetry_to_array(self, telemetry_batch: List[TelemetryData]) -> np.ndarray:
        """Convert telemetry data to NumPy array format"""
        
        data_list = []
        for t in telemetry_batch:
            row = [
                t.speed,
                t.throttle,
                t.brake,
                1.0 if t.drs else 0.0,  # Convert DRS boolean to float
                t.tire_temp_fl,
                t.tire_temp_fr,
                t.tire_temp_rl,
                t.tire_temp_rr,
                t.fuel_load,
                t.sector,
                t.gear,
                t.lap_number
            ]
            data_list.append(row)
        
        return np.array(data_list, dtype=np.float32)
    
    def _process_on_cpu(self, data_array: np.ndarray) -> Dict[str, np.ndarray]:
        """CPU fallback processing when GPU is not available"""
        
        batch_size = data_array.shape[0]
        features_out = np.zeros((batch_size, 3), dtype=np.float32)
        sector_times_out = np.zeros((batch_size, 3), dtype=np.float32)
        
        for i in range(batch_size):
            # Basic feature engineering on CPU
            speed = data_array[i, 0]
            throttle = data_array[i, 1]
            brake = data_array[i, 2]
            tire_temp_avg = np.mean(data_array[i, 4:8])
            fuel_load = data_array[i, 8]
            
            # Tire degradation
            tire_degradation = min(1.0, max(0.0, (tire_temp_avg - 80.0) / 40.0))
            
            # Fuel efficiency
            fuel_efficiency = speed / max(fuel_load, 1.0) if fuel_load > 0 else 0.0
            
            # Performance index
            performance_idx = (speed * throttle * (1.0 - brake/100.0) * (1.0 - tire_degradation)) / 100.0
            
            features_out[i] = [tire_degradation, fuel_efficiency, performance_idx]
        
        return {
            'features': features_out,
            'sector_times': sector_times_out,
            'processing_time': 0.001,  # Minimal time for CPU
            'batch_size': batch_size
        }
    
    def get_performance_stats(self) -> Dict[str, float]:
        """Get processing performance statistics"""
        
        if not self.processing_times:
            return {}
        
        return {
            'avg_processing_time': np.mean(self.processing_times),
            'min_processing_time': np.min(self.processing_times),
            'max_processing_time': np.max(self.processing_times),
            'total_samples_processed': self.samples_processed,
            'avg_throughput': self.samples_processed / sum(self.processing_times) if self.processing_times else 0
        }
    
    def reset_stats(self):
        """Reset performance tracking statistics"""
        self.processing_times = []
        self.samples_processed = 0
    
    def __del__(self):
        """Cleanup GPU resources"""
        if hasattr(self, 'gpu_memory_pool') and self.gpu_memory_pool:
            self.gpu_memory_pool.free_all_blocks()