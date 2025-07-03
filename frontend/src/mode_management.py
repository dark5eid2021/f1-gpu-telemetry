# ============================================================================
# src/api/mode_management.py - API endpoints for CPU/GPU mode switching
# ============================================================================

import asyncio
import logging
import os
import torch
from typing import Dict, Optional
from datetime import datetime

from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
import redis

from config.config import get_config
from inference.gpu_pipeline import RealTimeInferencePipeline
from ml.cpu_pipeline import CPUMLPipeline  # We'll need to create this


class ProcessingModeRequest(BaseModel):
    """Request model for changing processing mode"""
    mode: str  # 'cpu' or 'gpu'
    
class ProcessingModeResponse(BaseModel):
    """Response model for processing mode operations"""
    current_mode: str
    previous_mode: Optional[str] = None
    capabilities: Dict
    performance_metrics: Dict
    timestamp: str
    restart_required: bool = False

class SystemCapabilitiesResponse(BaseModel):
    """Response model for system capabilities"""
    gpu_available: bool
    gpu_count: int
    gpu_names: list
    total_gpu_memory_gb: float
    cuda_version: Optional[str]
    torch_version: str
    rapids_available: bool
    recommended_mode: str
    cpu_cores: int
    total_memory_gb: float


# Global state for mode management
current_processing_mode = "cpu"  # Default to CPU mode
processing_pipeline = None
mode_switch_lock = asyncio.Lock()

# Create router
router = APIRouter(prefix="/api/v1/system", tags=["system"])
logger = logging.getLogger(__name__)


class ModeManager:
    """Manages switching between CPU and GPU processing modes"""
    
    def __init__(self):
        self.config = get_config()
        self.redis_client = None
        self.current_mode = "cpu"
        self.pipeline = None
        
        # Initialize Redis connection
        try:
            redis_config = self.config.get_redis_config()
            self.redis_client = redis.Redis(**redis_config)
        except Exception as e:
            logger.warning(f"Redis not available: {e}")
    
    def get_system_capabilities(self) -> Dict:
        """Get comprehensive system capabilities"""
        capabilities = {
            "gpu_available": torch.cuda.is_available(),
            "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "gpu_names": [],
            "total_gpu_memory_gb": 0.0,
            "cuda_version": None,
            "torch_version": torch.__version__,
            "rapids_available": False,
            "cpu_cores": os.cpu_count() or 1,
            "total_memory_gb": 8.0,  # Default fallback
            "recommended_mode": "cpu"
        }
        
        # GPU Information
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                gpu_name = torch.cuda.get_device_name(i)
                capabilities["gpu_names"].append(gpu_name)
                
                # Get GPU memory
                props = torch.cuda.get_device_properties(i)
                gpu_memory_gb = props.total_memory / (1024**3)
                capabilities["total_gpu_memory_gb"] += gpu_memory_gb
            
            # Get CUDA version
            capabilities["cuda_version"] = torch.version.cuda
            capabilities["recommended_mode"] = "gpu"
        
        # Check for RAPIDS availability
        try:
            import cudf
            capabilities["rapids_available"] = True
        except ImportError:
            pass
        
        # Get system memory
        try:
            import psutil
            capabilities["total_memory_gb"] = psutil.virtual_memory().total / (1024**3)
        except ImportError:
            pass
        
        return capabilities
    
    async def switch_mode(self, new_mode: str) -> Dict:
        """Switch processing mode with proper cleanup and initialization"""
        
        if new_mode not in ['cpu', 'gpu']:
            raise ValueError(f"Invalid mode: {new_mode}. Must be 'cpu' or 'gpu'")
        
        if new_mode == self.current_mode:
            return {
                "status": "no_change",
                "message": f"Already in {new_mode} mode"
            }
        
        # Check if GPU mode is available
        if new_mode == 'gpu' and not torch.cuda.is_available():
            raise ValueError("GPU mode requested but CUDA is not available")
        
        logger.info(f"Switching from {self.current_mode} to {new_mode} mode")
        
        previous_mode = self.current_mode
        
        try:
            # Step 1: Stop current pipeline
            if self.pipeline:
                logger.info("Stopping current processing pipeline...")
                await self.pipeline.shutdown()
                self.pipeline = None
            
            # Step 2: Update configuration
            self.current_mode = new_mode
            
            # Update environment variables
            if new_mode == 'gpu':
                os.environ['USE_CPU_ONLY'] = 'false'
                os.environ['GPU_ENABLED'] = 'true'
                os.environ['GPU_BATCH_SIZE'] = str(self.config.GPU_BATCH_SIZE)
            else:
                os.environ['USE_CPU_ONLY'] = 'true'
                os.environ['GPU_ENABLED'] = 'false'
                os.environ['GPU_BATCH_SIZE'] = '64'  # Smaller for CPU
            
            # Step 3: Initialize new pipeline
            logger.info(f"Initializing {new_mode} processing pipeline...")
            
            if new_mode == 'gpu':
                from ml.train_models import GPUMLPipeline
                ml_pipeline = GPUMLPipeline(device='cuda')
                self.pipeline = RealTimeInferencePipeline(
                    ml_pipeline, 
                    self.config.get_redis_config()
                )
            else:
                ml_pipeline = CPUMLPipeline()
                # Use a CPU-optimized inference pipeline
                self.pipeline = CPUInferencePipeline(
                    ml_pipeline,
                    self.config.get_redis_config()
                )
            
            # Step 4: Start new pipeline
            logger.info("Starting new processing pipeline...")
            asyncio.create_task(self.pipeline.process_telemetry_stream())
            
            # Step 5: Update Redis cache
            if self.redis_client:
                mode_info = {
                    "current_mode": new_mode,
                    "previous_mode": previous_mode,
                    "switched_at": datetime.now().isoformat(),
                    "capabilities": self.get_system_capabilities()
                }
                self.redis_client.setex(
                    'processing_mode_info', 
                    3600,  # 1 hour TTL
                    str(mode_info)
                )
            
            logger.info(f"Successfully switched to {new_mode} mode")
            
            return {
                "status": "success",
                "message": f"Successfully switched to {new_mode} mode",
                "previous_mode": previous_mode,
                "new_mode": new_mode,
                "restart_time": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to switch to {new_mode} mode: {e}")
            
            # Attempt to restore previous mode
            try:
                self.current_mode = previous_mode
                logger.warning(f"Restored to {previous_mode} mode after failure")
            except Exception as restore_error:
                logger.error(f"Failed to restore previous mode: {restore_error}")
            
            raise HTTPException(
                status_code=500,
                detail=f"Failed to switch to {new_mode} mode: {str(e)}"
            )
    
    def get_performance_metrics(self) -> Dict:
        """Get current performance metrics"""
        metrics = {
            "mode": self.current_mode,
            "throughput": 0,
            "latency": 0,
            "gpu_utilization": 0,
            "memory_usage": 0,
            "active_connections": 0
        }
        
        if self.pipeline:
            try:
                stats = self.pipeline.get_performance_stats()
                metrics.update({
                    "throughput": stats.get("predictions_per_second", 0),
                    "latency": stats.get("avg_prediction_time", 0) * 1000,  # Convert to ms
                    "total_predictions": stats.get("predictions_generated", 0),
                    "uptime": stats.get("uptime", 0)
                })
            except Exception as e:
                logger.warning(f"Could not get pipeline stats: {e}")
        
        # GPU-specific metrics
        if self.current_mode == 'gpu' and torch.cuda.is_available():
            try:
                metrics["gpu_utilization"] = self._get_gpu_utilization()
                metrics["gpu_memory_used"] = torch.cuda.memory_allocated(0) / (1024**3)  # GB
                metrics["gpu_memory_total"] = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB
            except Exception as e:
                logger.warning(f"Could not get GPU metrics: {e}")
        
        # System memory
        try:
            import psutil
            memory = psutil.virtual_memory()
            metrics["memory_usage"] = memory.percent
            metrics["cpu_usage"] = psutil.cpu_percent(interval=1)
        except ImportError:
            pass
        
        return metrics
    
    def _get_gpu_utilization(self) -> float:
        """Get GPU utilization percentage"""
        try:
            import subprocess
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                return float(result.stdout.strip())
        except Exception:
            pass
        return 0.0


# Initialize global mode manager
mode_manager = ModeManager()


@router.get("/capabilities", response_model=SystemCapabilitiesResponse)
async def get_system_capabilities():
    """Get system hardware capabilities"""
    try:
        capabilities = mode_manager.get_system_capabilities()
        return SystemCapabilitiesResponse(**capabilities)
    except Exception as e:
        logger.error(f"Error getting system capabilities: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/processing-mode", response_model=ProcessingModeResponse)
async def get_current_processing_mode():
    """Get current processing mode and system status"""
    try:
        capabilities = mode_manager.get_system_capabilities()
        performance_metrics = mode_manager.get_performance_metrics()
        
        return ProcessingModeResponse(
            current_mode=mode_manager.current_mode,
            capabilities=capabilities,
            performance_metrics=performance_metrics,
            timestamp=datetime.now().isoformat(),
            restart_required=False
        )
    except Exception as e:
        logger.error(f"Error getting processing mode: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/processing-mode", response_model=ProcessingModeResponse)
async def set_processing_mode(
    request: ProcessingModeRequest,
    background_tasks: BackgroundTasks
):
    """Switch between CPU and GPU processing modes"""
    
    async with mode_switch_lock:
        try:
            # Validate mode
            if request.mode not in ['cpu', 'gpu']:
                raise HTTPException(
                    status_code=400,
                    detail="Mode must be 'cpu' or 'gpu'"
                )
            
            # Check GPU availability for GPU mode
            if request.mode == 'gpu' and not torch.cuda.is_available():
                raise HTTPException(
                    status_code=400,
                    detail="GPU mode requested but CUDA is not available"
                )
            
            previous_mode = mode_manager.current_mode
            
            # Perform mode switch
            switch_result = await mode_manager.switch_mode(request.mode)
            
            # Get updated capabilities and metrics
            capabilities = mode_manager.get_system_capabilities()
            performance_metrics = mode_manager.get_performance_metrics()
            
            return ProcessingModeResponse(
                current_mode=request.mode,
                previous_mode=previous_mode,
                capabilities=capabilities,
                performance_metrics=performance_metrics,
                timestamp=datetime.now().isoformat(),
                restart_required=False
            )
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error switching processing mode: {e}")
            raise HTTPException(status_code=500, detail=str(e))


@router.get("/performance")
async def get_performance_metrics():
    """Get detailed performance metrics for current mode"""
    try:
        metrics = mode_manager.get_performance_metrics()
        
        # Add mode-specific recommendations
        recommendations = []
        
        if metrics["mode"] == "cpu":
            if torch.cuda.is_available():
                recommendations.append({
                    "type": "upgrade",
                    "message": "Switch to GPU mode for better performance",
                    "action": "switch_to_gpu"
                })
            
            if metrics.get("cpu_usage", 0) > 90:
                recommendations.append({
                    "type": "warning",
                    "message": "High CPU usage detected",
                    "action": "reduce_batch_size"
                })
        
        elif metrics["mode"] == "gpu":
            if metrics.get("gpu_utilization", 0) > 95:
                recommendations.append({
                    "type": "warning", 
                    "message": "High GPU utilization detected",
                    "action": "optimize_batch_size"
                })
            
            if metrics.get("gpu_memory_used", 0) / metrics.get("gpu_memory_total", 1) > 0.9:
                recommendations.append({
                    "type": "warning",
                    "message": "High GPU memory usage",
                    "action": "reduce_batch_size"
                })
        
        return {
            "metrics": metrics,
            "recommendations": recommendations,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting performance metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/restart-pipeline")
async def restart_processing_pipeline():
    """Restart the current processing pipeline"""
    
    async with mode_switch_lock:
        try:
            current_mode = mode_manager.current_mode
            logger.info(f"Restarting {current_mode} processing pipeline")
            
            # Restart by switching to the same mode
            switch_result = await mode_manager.switch_mode(current_mode)
            
            return {
                "status": "success",
                "message": f"Processing pipeline restarted in {current_mode} mode",
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error restarting pipeline: {e}")
            raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def get_system_health():
    """Get comprehensive system health check"""
    try:
        capabilities = mode_manager.get_system_capabilities()
        performance_metrics = mode_manager.get_performance_metrics()
        
        # Determine overall health status
        health_status = "healthy"
        issues = []
        
        # Check GPU health if in GPU mode
        if mode_manager.current_mode == "gpu":
            if not capabilities["gpu_available"]:
                health_status = "degraded"
                issues.append("GPU not available but in GPU mode")
            
            gpu_util = performance_metrics.get("gpu_utilization", 0)
            if gpu_util > 95:
                health_status = "warning"
                issues.append(f"High GPU utilization: {gpu_util}%")
        
        # Check CPU health
        cpu_usage = performance_metrics.get("cpu_usage", 0)
        if cpu_usage > 90:
            if health_status == "healthy":
                health_status = "warning"
            issues.append(f"High CPU usage: {cpu_usage}%")
        
        # Check memory
        memory_usage = performance_metrics.get("memory_usage", 0)
        if memory_usage > 90:
            if health_status == "healthy":
                health_status = "warning"
            issues.append(f"High memory usage: {memory_usage}%")
        
        # Check pipeline status
        if not mode_manager.pipeline:
            health_status = "degraded"
            issues.append("Processing pipeline not running")
        
        return {
            "status": health_status,
            "mode": mode_manager.current_mode,
            "issues": issues,
            "capabilities": capabilities,
            "performance": performance_metrics,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting system health: {e}")
        return {
            "status": "error",
            "mode": mode_manager.current_mode,
            "issues": [f"Health check failed: {str(e)}"],
            "timestamp": datetime.now().isoformat()
        }


# ============================================================================
# CPU-specific inference pipeline (simplified version)
# ============================================================================

class CPUInferencePipeline:
    """CPU-optimized inference pipeline"""
    
    def __init__(self, ml_pipeline, redis_config: Dict):
        self.ml_pipeline = ml_pipeline
        self.redis_config = redis_config
        self.redis_client = redis.Redis(**redis_config)
        self.logger = logging.getLogger(__name__)
        
        # CPU-optimized settings
        self.batch_size = 64  # Smaller batch size for CPU
        self.processing_interval = 0.1  # 10Hz instead of 50Hz
        
        # Performance tracking
        self.predictions_generated = 0
        self.total_processing_time = 0
        self.start_time = datetime.now()
    
    async def process_telemetry_stream(self):
        """CPU-optimized telemetry processing"""
        self.logger.info("🖥️ Starting CPU telemetry processing...")
        
        # Simulate CPU processing with mock data for now
        # In a real implementation, this would connect to the actual data source
        while True:
            try:
                # Generate mock telemetry data for CPU mode
                mock_telemetry = self._generate_mock_telemetry()
                
                # Process with CPU pipeline
                predictions = await self._process_batch_cpu(mock_telemetry)
                
                # Cache results
                await self._cache_predictions(predictions)
                
                # Update performance metrics
                self.predictions_generated += 1
                
                # CPU mode runs at lower frequency
                await asyncio.sleep(self.processing_interval)
                
            except asyncio.CancelledError:
                self.logger.info("🛑 CPU telemetry processing cancelled")
                break
            except Exception as e:
                self.logger.error(f"❌ Error in CPU telemetry processing: {e}")
                await asyncio.sleep(1)
    
    def _generate_mock_telemetry(self):
        """Generate mock telemetry data for CPU mode"""
        import random
        
        telemetry_batch = []
        for driver_id in range(1, 21):  # 20 drivers
            telemetry_point = {
                'driver_id': driver_id,
                'speed': random.uniform(200, 350),
                'throttle': random.uniform(0, 100),
                'brake': random.uniform(0, 50),
                'gear': random.randint(1, 8),
                'timestamp': datetime.now().isoformat()
            }
            telemetry_batch.append(telemetry_point)
        
        return telemetry_batch
    
    async def _process_batch_cpu(self, telemetry_batch):
        """Process telemetry batch on CPU"""
        import time
        start_time = time.time()
        
        # Simplified CPU processing
        predictions = {
            'timestamp': datetime.now().isoformat(),
            'batch_size': len(telemetry_batch),
            'mode': 'cpu',
            'race_outcome': [random.uniform(0, 1) for _ in range(20)],
            'lap_times': [
                {
                    'driver_id': i,
                    'predicted_lap_time': random.uniform(75, 85),
                    'confidence': random.uniform(0.7, 0.9)
                }
                for i in range(1, 21)
            ],
            'performance_metrics': {
                'processing_time': time.time() - start_time,
                'throughput': len(telemetry_batch) / (time.time() - start_time),
                'mode': 'cpu'
            }
        }
        
        self.total_processing_time += (time.time() - start_time)
        return predictions
    
    async def _cache_predictions(self, predictions):
        """Cache predictions in Redis"""
        try:
            prediction_json = str(predictions)  # Convert to string for Redis
            self.redis_client.setex('latest_predictions', 60, prediction_json)
        except Exception as e:
            self.logger.error(f"❌ Error caching predictions: {e}")
    
    async def shutdown(self):
        """Shutdown CPU pipeline"""
        self.logger.info("🛑 Shutting down CPU inference pipeline...")
        if self.redis_client:
            self.redis_client.close()
    
    def get_performance_stats(self):
        """Get CPU pipeline performance statistics"""
        uptime = (datetime.now() - self.start_time).total_seconds()
        
        return {
            'predictions_generated': self.predictions_generated,
            'total_processing_time': self.total_processing_time,
            'avg_prediction_time': self.total_processing_time / max(self.predictions_generated, 1),
            'uptime': uptime,
            'predictions_per_second': self.predictions_generated / max(uptime, 1),
            'mode': 'cpu'
        }


# ============================================================================
# CPU ML Pipeline (simplified implementation)
# ============================================================================

class CPUMLPipeline:
    """CPU-only machine learning pipeline"""
    
    def __init__(self):
        self.device = 'cpu'
        self.logger = logging.getLogger(__name__)
        self.logger.info("🖥️ Initializing CPU-only ML pipeline")
        
        # Use simpler models for CPU
        self.models = {
            'race_predictor': None,  # Simplified model
            'lap_time_predictor': None,  # Basic regression
            'pit_strategy': None  # Rule-based system
        }
    
    def predict_race_outcome(self, telemetry_data):
        """Simple CPU-based race prediction"""
        import random
        # Simplified prediction logic for CPU mode
        return [random.uniform(0, 1) for _ in range(20)]
    
    def predict_lap_times(self, telemetry_data):
        """Simple CPU-based lap time prediction"""
        import random
        predictions = []
        for driver_id in range(1, 21):
            predictions.append({
                'driver_id': driver_id,
                'predicted_lap_time': random.uniform(75, 85),
                'confidence': random.uniform(0.7, 0.9)
            })
        return predictions