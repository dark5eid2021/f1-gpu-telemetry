"""
Real-time GPU inference pipeline for F1 predictions
Combines GPU processing with ML models for sub-second predictions
"""

import asyncio
import json
import logging
import time
from typing import Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor

import torch
import cupy as cp
import redis
from kafka import KafkaConsumer

from gpu_processing.telemetry_processor import GPUTelemetryProcessor, TelemetryData
from ml.models import RaceOutcomePredictor, LapTimePredictor, ModelConfig


class RealTimeInferencePipeline:
    """
    Real-time GPU inference pipeline for F1 predictions
    
    Processes streaming telemetry data and generates predictions using
    GPU-accelerated feature engineering and ML models
    """
    
    def __init__(self, ml_pipeline, redis_config: Dict, batch_size: int = 1024):
        """
        Initialize the real-time inference pipeline
        
        Args:
            ml_pipeline: Trained ML models pipeline
            redis_config: Redis configuration for caching
            batch_size: Batch size for GPU processing
        """
        self.ml_pipeline = ml_pipeline
        self.redis_config = redis_config
        self.batch_size = batch_size
        
        # Initialize components
        self.telemetry_processor = GPUTelemetryProcessor(batch_size=batch_size)
        self.redis_client = redis.Redis(**redis_config)
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.logger = logging.getLogger(__name__)
        
        # Buffer for batching telemetry data
        self.telemetry_buffer = []
        self.buffer_lock = asyncio.Lock()
        
        # Performance tracking
        self.predictions_generated = 0
        self.total_processing_time = 0
        self.start_time = time.time()
        
        # Model instances
        self.models = {}
        self._initialize_models()
        
        self.logger.info("✅ Real-time inference pipeline initialized")
    
    def _initialize_models(self):
        """Initialize PyTorch models for inference"""
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        try:
            # Race outcome prediction model
            config = ModelConfig()
            self.models['race_outcome'] = RaceOutcomePredictor(config).to(device)
            self.models['race_outcome'].eval()
            
            # Lap time prediction model  
            self.models['lap_time'] = LapTimePredictor().to(device)
            self.models['lap_time'].eval()
            
            self.logger.info(f"✅ Models initialized on {device}")
            
        except Exception as e:
            self.logger.error(f"❌ Error initializing models: {e}")
    
    async def process_telemetry_stream(self):
        """Main processing loop for real-time telemetry"""
        
        self.logger.info("📡 Starting telemetry stream processing...")
        
        # Initialize Kafka consumer
        consumer = KafkaConsumer(
            'f1-telemetry',
            bootstrap_servers=['localhost:9092'],
            value_deserializer=lambda m: json.loads(m.decode('utf-8')),
            group_id='f1-inference-pipeline',
            auto_offset_reset='latest',
            enable_auto_commit=True,
            consumer_timeout_ms=1000  # Timeout for batch processing
        )
        
        try:
            while True:
                # Collect messages for batch processing
                message_batch = []
                
                # Poll for messages with timeout
                message_pack = consumer.poll(timeout_ms=100, max_records=self.batch_size)
                
                for topic_partition, messages in message_pack.items():
                    for message in messages:
                        telemetry_data = self._parse_telemetry(message.value)
                        if telemetry_data:
                            message_batch.append(telemetry_data)
                
                # Process batch if we have data
                if message_batch:
                    await self._process_batch(message_batch)
                
                # Also process buffer periodically even without new messages
                async with self.buffer_lock:
                    if len(self.telemetry_buffer) >= self.batch_size // 2:
                        batch_to_process = self.telemetry_buffer[:self.batch_size]
                        self.telemetry_buffer = self.telemetry_buffer[self.batch_size:]
                        
                        if batch_to_process:
                            await self._process_batch(batch_to_process)
                
                # Small delay to prevent busy waiting
                await asyncio.sleep(0.001)
                
        except asyncio.CancelledError:
            self.logger.info("🛑 Telemetry processing cancelled")
        except Exception as e:
            self.logger.error(f"❌ Error in telemetry processing: {e}")
        finally:
            consumer.close()
    
    def _parse_telemetry(self, data: Dict) -> Optional[TelemetryData]:
        """Parse raw telemetry data into structured format"""
        
        try:
            return TelemetryData(
                timestamp=data.get('date', time.time()),
                driver_id=data.get('driver_number', 0),
                speed=float(data.get('speed', 0.0)),
                throttle=float(data.get('throttle', 0.0)),
                brake=float(data.get('brake', 0.0)),
                gear=int(data.get('n_gear', 1)),
                drs=bool(data.get('drs', False)),
                tire_temp_fl=float(data.get('tire_temp_fl', 80.0)),
                tire_temp_fr=float(data.get('tire_temp_fr', 80.0)),
                tire_temp_rl=float(data.get('tire_temp_rl', 80.0)),
                tire_temp_rr=float(data.get('tire_temp_rr', 80.0)),
                fuel_load=float(data.get('fuel_load', 100.0)),
                lap_number=int(data.get('lap_number', 1)),
                sector=int(data.get('sector', 1))
            )
        except Exception as e:
            self.logger.warning(f"⚠️ Error parsing telemetry: {e}")
            return None
    
    async def _process_batch(self, telemetry_batch: List[TelemetryData]):
        """Process a batch of telemetry data and generate predictions"""
        
        if not telemetry_batch:
            return
        
        start_time = time.time()
        
        try:
            # GPU feature engineering
            gpu_results = self.telemetry_processor.process_batch(telemetry_batch)
            
            if not gpu_results:
                self.logger.warning("⚠️ GPU processing returned no results")
                return
            
            # Generate ML predictions
            predictions = await self._generate_predictions(gpu_results, telemetry_batch)
            
            # Cache results in Redis
            await self._cache_predictions(predictions)
            
            # Update performance metrics
            processing_time = time.time() - start_time
            self.total_processing_time += processing_time
            self.predictions_generated += 1
            
            # Log performance periodically
            if self.predictions_generated % 100 == 0:
                avg_time = self.total_processing_time / self.predictions_generated
                throughput = len(telemetry_batch) / processing_time
                
                self.logger.info(
                    f"⚡ Processed batch: {len(telemetry_batch)} samples in {processing_time*1000:.1f}ms "
                    f"({throughput:.0f} samples/sec, avg: {avg_time*1000:.1f}ms)"
                )
            
        except Exception as e:
            self.logger.error(f"❌ Error processing batch: {e}")
    
    async def _generate_predictions(self, gpu_results: Dict, 
                                  telemetry_batch: List[TelemetryData]) -> Dict:
        """Generate ML predictions from GPU-processed features"""
        
        predictions = {
            'timestamp': time.time(),
            'batch_size': len(telemetry_batch),
            'race_outcome': None,
            'lap_times': [],
            'pit_strategies': [],
            'performance_metrics': {}
        }
        
        try:
            # Extract GPU features
            features = gpu_results.get('features')
            if features is None:
                return predictions
            
            # Convert to PyTorch tensors
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            
            if hasattr(features, 'get'):  # CuPy array
                torch_features = torch.from_numpy(cp.asnumpy(features)).float().to(device)
            else:  # NumPy array
                torch_features = torch.from_numpy(features).float().to(device)
            
            # Race outcome prediction (if we have enough drivers)
            unique_drivers = len(set(t.driver_id for t in telemetry_batch))
            if unique_drivers >= 10:  # Minimum drivers for meaningful race prediction
                race_prediction = await self._predict_race_outcome(torch_features, telemetry_batch)
                predictions['race_outcome'] = race_prediction
            
            # Lap time predictions
            lap_predictions = await self._predict_lap_times(torch_features, telemetry_batch)
            predictions['lap_times'] = lap_predictions
            
            # Pit strategy recommendations
            pit_predictions = await self._predict_pit_strategies(torch_features, telemetry_batch)
            predictions['pit_strategies'] = pit_predictions
            
            # Performance metrics
            predictions['performance_metrics'] = {
                'avg_tire_degradation': float(torch_features[:, 0].mean()),
                'avg_fuel_efficiency': float(torch_features[:, 1].mean()),
                'avg_performance_index': float(torch_features[:, 2].mean()),
                'gpu_processing_time': gpu_results.get('processing_time', 0)
            }
            
        except Exception as e:
            self.logger.error(f"❌ Error generating predictions: {e}")
        
        return predictions
    
    async def _predict_race_outcome(self, features: torch.Tensor, 
                                   telemetry_batch: List[TelemetryData]) -> Optional[List[float]]:
        """Predict race finishing positions"""
        
        try:
            if 'race_outcome' not in self.models:
                return None
            
            # Prepare input for race outcome model
            # Group by driver and create sequences
            driver_data = {}
            for i, t in enumerate(telemetry_batch):
                if t.driver_id not in driver_data:
                    driver_data[t.driver_id] = []
                driver_data[t.driver_id].append((features[i], t.driver_id))
            
            # Take top drivers with most data
            top_drivers = sorted(driver_data.keys(), 
                               key=lambda d: len(driver_data[d]), reverse=True)[:20]
            
            if len(top_drivers) < 10:
                return None
            
            # Create model input
            batch_features = []
            positions = []
            
            for driver_id in top_drivers:
                if driver_data[driver_id]:
                    # Use latest feature vector for this driver
                    latest_features = driver_data[driver_id][-1][0]
                    batch_features.append(latest_features)
                    positions.append(driver_id)
            
            if not batch_features:
                return None
            
            # Stack features and create dummy sequence dimension
            model_features = torch.stack(batch_features).unsqueeze(1)  # [drivers, 1, features]
            model_positions = torch.tensor(positions, device=features.device).unsqueeze(1)
            
            with torch.no_grad():
                output = self.models['race_outcome'](model_features, model_positions)
                position_probs = output['position_probs'].cpu().numpy()
            
            return position_probs.tolist()
            
        except Exception as e:
            self.logger.error(f"❌ Error predicting race outcome: {e}")
            return None
    
    async def _predict_lap_times(self, features: torch.Tensor, 
                                telemetry_batch: List[TelemetryData]) -> List[Dict]:
        """Predict lap times for each driver"""
        
        predictions = []
        
        try:
            if 'lap_time' not in self.models:
                return predictions
            
            # Group by driver
            driver_features = {}
            for i, t in enumerate(telemetry_batch):
                if t.driver_id not in driver_features:
                    driver_features[t.driver_id] = []
                driver_features[t.driver_id].append(features[i])
            
            # Predict for each driver
            for driver_id, driver_feat_list in driver_features.items():
                if len(driver_feat_list) >= 10:  # Need minimum sequence length
                    # Create sequence
                    sequence = torch.stack(driver_feat_list[-10:]).unsqueeze(0)  # Last 10 samples
                    
                    with torch.no_grad():
                        output = self.models['lap_time'](sequence)
                        predicted_time = float(output['lap_time'].cpu().item())
                    
                    predictions.append({
                        'driver_id': driver_id,
                        'predicted_lap_time': predicted_time,
                        'confidence': 0.85  # Placeholder confidence score
                    })
            
        except Exception as e:
            self.logger.error(f"❌ Error predicting lap times: {e}")
        
        return predictions
    
    async def _predict_pit_strategies(self, features: torch.Tensor, 
                                     telemetry_batch: List[TelemetryData]) -> List[Dict]:
        """Predict optimal pit strategies"""
        
        strategies = []
        
        try:
            # Group by driver and analyze pit timing
            driver_data = {}
            for i, t in enumerate(telemetry_batch):
                if t.driver_id not in driver_data:
                    driver_data[t.driver_id] = {
                        'tire_degradation': [],
                        'fuel_efficiency': [],
                        'lap_number': t.lap_number
                    }
                
                # Extract tire degradation from features
                tire_deg = float(features[i, 0])
                fuel_eff = float(features[i, 1])
                
                driver_data[t.driver_id]['tire_degradation'].append(tire_deg)
                driver_data[t.driver_id]['fuel_efficiency'].append(fuel_eff)
            
            # Generate pit recommendations
            for driver_id, data in driver_data.items():
                if data['tire_degradation']:
                    avg_tire_deg = sum(data['tire_degradation']) / len(data['tire_degradation'])
                    avg_fuel_eff = sum(data['fuel_efficiency']) / len(data['fuel_efficiency'])
                    
                    # Simple pit strategy logic (can be replaced with ML model)
                    should_pit = avg_tire_deg > 0.7 or avg_fuel_eff < 2.0
                    optimal_lap = data['lap_number'] + (3 if should_pit else 10)
                    
                    strategies.append({
                        'driver_id': driver_id,
                        'should_pit': should_pit,
                        'optimal_pit_lap': optimal_lap,
                        'tire_degradation': avg_tire_deg,
                        'fuel_efficiency': avg_fuel_eff,
                        'recommended_tire': 'soft' if should_pit else 'medium'
                    })
            
        except Exception as e:
            self.logger.error(f"❌ Error predicting pit strategies: {e}")
        
        return strategies
    
    async def _cache_predictions(self, predictions: Dict):
        """Cache predictions in Redis for real-time access"""
        
        try:
            # Store latest predictions
            prediction_json = json.dumps(predictions, default=str)
            self.redis_client.setex(
                'latest_predictions',
                60,  # 60 second TTL
                prediction_json
            )
            
            # Store driver-specific predictions
            if predictions.get('lap_times'):
                for lap_pred in predictions['lap_times']:
                    driver_key = f"driver_{lap_pred['driver_id']}_predictions"
                    driver_data = json.dumps(lap_pred, default=str)
                    self.redis_client.setex(driver_key, 30, driver_data)
            
            # Store performance metrics
            if predictions.get('performance_metrics'):
                metrics_json = json.dumps(predictions['performance_metrics'], default=str)
                self.redis_client.setex('performance_metrics', 60, metrics_json)
            
        except Exception as e:
            self.logger.error(f"❌ Error caching predictions: {e}")
    
    async def get_latest_predictions(self) -> Optional[Dict]:
        """Get latest predictions from cache"""
        
        try:
            cached_data = self.redis_client.get('latest_predictions')
            if cached_data:
                return json.loads(cached_data.decode('utf-8'))
        except Exception as e:
            self.logger.error(f"❌ Error retrieving predictions: {e}")
        
        return None
    
    async def get_driver_predictions(self, driver_id: int) -> Optional[Dict]:
        """Get predictions for specific driver"""
        
        try:
            driver_key = f"driver_{driver_id}_predictions"
            cached_data = self.redis_client.get(driver_key)
            if cached_data:
                return json.loads(cached_data.decode('utf-8'))
        except Exception as e:
            self.logger.error(f"❌ Error retrieving driver predictions: {e}")
        
        return None
    
    def get_performance_stats(self) -> Dict:
        """Get inference pipeline performance statistics"""
        
        elapsed_time = time.time() - self.start_time
        gpu_stats = self.telemetry_processor.get_performance_stats()
        
        return {
            'predictions_generated': self.predictions_generated,
            'total_processing_time': self.total_processing_time,
            'avg_prediction_time': self.total_processing_time / max(self.predictions_generated, 1),
            'uptime': elapsed_time,
            'predictions_per_second': self.predictions_generated / max(elapsed_time, 1),
            'gpu_processing_stats': gpu_stats
        }
    
    async def health_check(self) -> Dict[str, bool]:
        """Perform health check on all components"""
        
        health = {
            'redis_connection': False,
            'gpu_available': False,
            'models_loaded': False,
            'overall_healthy': False
        }
        
        try:
            # Check Redis connection
            self.redis_client.ping()
            health['redis_connection'] = True
        except:
            pass
        
        # Check GPU availability
        health['gpu_available'] = torch.cuda.is_available()
        
        # Check models
        health['models_loaded'] = len(self.models) > 0
        
        # Overall health
        health['overall_healthy'] = all([
            health['redis_connection'],
            health['models_loaded']
        ])
        
        return health
    
    async def shutdown(self):
        """Gracefully shutdown the inference pipeline"""
        
        self.logger.info("🛑 Shutting down inference pipeline...")
        
        # Close Redis connection
        if self.redis_client:
            self.redis_client.close()
        
        # Shutdown thread executor
        if self.executor:
            self.executor.shutdown(wait=True)
        
        self.logger.info("✅ Inference pipeline shutdown complete")


# Utility functions for testing and monitoring
async def test_inference_pipeline():
    """Test the inference pipeline with mock data"""
    
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Mock configuration
    redis_config = {'host': 'localhost', 'port': 6379, 'db': 0}
    
    # Create pipeline
    pipeline = RealTimeInferencePipeline(None, redis_config)
    
    # Create mock telemetry data
    mock_telemetry = []
    for i in range(100):
        telemetry = TelemetryData(
            timestamp=time.time(),
            driver_id=i % 20 + 1,
            speed=250.0 + (i % 50),
            throttle=80.0 + (i % 20),
            brake=10.0 + (i % 15),
            gear=6,
            drs=i % 2 == 0,
            tire_temp_fl=85.0 + (i % 10),
            tire_temp_fr=87.0 + (i % 10),
            tire_temp_rl=89.0 + (i % 10),
            tire_temp_rr=88.0 + (i % 10),
            fuel_load=100.0 - (i % 30),
            lap_number=i // 20 + 1,
            sector=i % 3 + 1
        )
        mock_telemetry.append(telemetry)
    
    # Process mock data
    await pipeline._process_batch(mock_telemetry)
    
    # Get results
    predictions = await pipeline.get_latest_predictions()
    if predictions:
        logger.info(f"✅ Test successful: Generated predictions for {predictions.get('batch_size', 0)} samples")
    else:
        logger.error("❌ Test failed: No predictions generated")
    
    # Health check
    health = await pipeline.health_check()
    logger.info(f"Health status: {health}")
    
    # Performance stats
    stats = pipeline.get_performance_stats()
    logger.info(f"Performance: {stats}")
    
    await pipeline.shutdown()


if __name__ == "__main__":
    asyncio.run(test_inference_pipeline())