"""
FastAPI backend for F1 GPU Telemetry System
Provides REST API and WebSocket endpoints for real-time data access
"""

import asyncio
import json
import logging
import time
from datetime import datetime
from typing import Dict, List, Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import redis
import torch

from config.config import get_config

# Initialize FastAPI app
app = FastAPI(
    title="F1 GPU Telemetry API",
    description="Real-time F1 telemetry processing and race prediction API",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global instances
redis_client: Optional[redis.Redis] = None
config = get_config()
logger = logging.getLogger(__name__)

# WebSocket connection manager
class WebSocketManager:
    """Manages WebSocket connections for real-time streaming"""
    
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.connection_count = 0
    
    async def connect(self, websocket: WebSocket):
        """Accept new WebSocket connection"""
        await websocket.accept()
        self.active_connections.append(websocket)
        self.connection_count += 1
        logger.info(f"📡 WebSocket connected. Total connections: {len(self.active_connections)}")
    
    def disconnect(self, websocket: WebSocket):
        """Remove WebSocket connection"""
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        logger.info(f"🔌 WebSocket disconnected. Total connections: {len(self.active_connections)}")
    
    async def broadcast(self, message: str):
        """Broadcast message to all connected clients"""
        if not self.active_connections:
            return
        
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except Exception:
                disconnected.append(connection)
        
        # Remove disconnected clients
        for connection in disconnected:
            self.disconnect(connection)

# Initialize WebSocket manager
websocket_manager = WebSocketManager()


@app.on_event("startup")
async def startup_event():
    """Initialize application on startup"""
    global redis_client
    
    logger.info("🚀 Starting F1 GPU Telemetry API...")
    
    # Initialize Redis connection
    redis_config = config.get_redis_config()
    redis_client = redis.Redis(**redis_config)
    
    try:
        redis_client.ping()
        logger.info("✅ Redis connection established")
    except Exception as e:
        logger.error(f"❌ Redis connection failed: {e}")
    
    # Start background task for WebSocket broadcasting
    asyncio.create_task(broadcast_updates())
    
    logger.info("✅ F1 GPU Telemetry API started successfully")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on application shutdown"""
    global redis_client
    
    logger.info("🛑 Shutting down F1 GPU Telemetry API...")
    
    if redis_client:
        redis_client.close()
    
    logger.info("✅ F1 GPU Telemetry API shutdown complete")


async def get_redis_client() -> redis.Redis:
    """Dependency to get Redis client"""
    if redis_client is None:
        raise HTTPException(status_code=500, detail="Redis connection not available")
    return redis_client


async def broadcast_updates():
    """Background task to broadcast real-time updates to WebSocket clients"""
    
    while True:
        try:
            if redis_client and websocket_manager.active_connections:
                # Get latest predictions from Redis
                predictions = redis_client.get('latest_predictions')
                if predictions:
                    await websocket_manager.broadcast(predictions.decode('utf-8'))
            
            await asyncio.sleep(0.1)  # 10Hz update rate
            
        except Exception as e:
            logger.error(f"❌ Error in broadcast updates: {e}")
            await asyncio.sleep(1)


# Health Check Endpoints
@app.get("/api/v1/health")
async def health_check():
    """System health check endpoint"""
    
    health_status = {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0",
        "gpu_available": torch.cuda.is_available(),
        "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "redis_connected": False,
        "active_websockets": len(websocket_manager.active_connections)
    }
    
    # Check Redis connection
    try:
        if redis_client:
            redis_client.ping()
            health_status["redis_connected"] = True
    except Exception:
        health_status["status"] = "degraded"
    
    # Add GPU information if available
    if torch.cuda.is_available():
        health_status["gpu_info"] = {
            "device_name": torch.cuda.get_device_name(0),
            "memory_allocated": torch.cuda.memory_allocated(0),
            "memory_reserved": torch.cuda.memory_reserved(0)
        }
    
    return health_status


@app.get("/api/v1/status")
async def system_status():
    """Detailed system status endpoint"""
    
    status = {
        "system": {
            "uptime": time.time(),
            "python_version": "3.10+",
            "api_version": "1.0.0"
        },
        "redis": {
            "connected": False,
            "memory_usage": None
        },
        "gpu": {
            "available": torch.cuda.is_available(),
            "count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "devices": []
        },
        "websockets": {
            "active_connections": len(websocket_manager.active_connections),
            "total_connections": websocket_manager.connection_count
        }
    }
    
    # Redis status
    try:
        if redis_client:
            redis_client.ping()
            status["redis"]["connected"] = True
            info = redis_client.info()
            status["redis"]["memory_usage"] = info.get("used_memory_human", "unknown")
    except Exception:
        pass
    
    # GPU status
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            device_info = {
                "id": i,
                "name": torch.cuda.get_device_name(i),
                "memory_allocated": torch.cuda.memory_allocated(i),
                "memory_reserved": torch.cuda.memory_reserved(i),
                "utilization": "unknown"  # Would need nvidia-ml-py for real utilization
            }
            status["gpu"]["devices"].append(device_info)
    
    return status


# Telemetry Endpoints
@app.get("/api/v1/telemetry/live")
async def get_live_telemetry(client: redis.Redis = Depends(get_redis_client)):
    """Get latest live telemetry data"""
    
    try:
        # Get latest telemetry from Redis
        telemetry_data = client.get('latest_telemetry')
        
        if telemetry_data:
            return json.loads(telemetry_data.decode('utf-8'))
        else:
            return {
                "message": "No live telemetry data available",
                "timestamp": datetime.now().isoformat(),
                "data": []
            }
    
    except Exception as e:
        logger.error(f"❌ Error retrieving telemetry: {e}")
        raise HTTPException(status_code=500, detail="Error retrieving telemetry data")


@app.get("/api/v1/telemetry/driver/{driver_id}")
async def get_driver_telemetry(driver_id: int, client: redis.Redis = Depends(get_redis_client)):
    """Get telemetry data for specific driver"""
    
    try:
        driver_key = f"driver_{driver_id}_telemetry"
        telemetry_data = client.get(driver_key)
        
        if telemetry_data:
            return json.loads(telemetry_data.decode('utf-8'))
        else:
            raise HTTPException(status_code=404, detail=f"No telemetry data found for driver {driver_id}")
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error retrieving driver telemetry: {e}")
        raise HTTPException(status_code=500, detail="Error retrieving driver telemetry")


# Prediction Endpoints
@app.get("/api/v1/predictions/race")
async def get_race_predictions(client: redis.Redis = Depends(get_redis_client)):
    """Get current race outcome predictions"""
    
    try:
        predictions_data = client.get('latest_predictions')
        
        if predictions_data:
            predictions = json.loads(predictions_data.decode('utf-8'))
            
            return {
                "timestamp": predictions.get('timestamp'),
                "race_outcome": predictions.get('race_outcome'),
                "confidence": 0.85,  # Placeholder confidence
                "num_drivers": len(predictions.get('race_outcome', [])) if predictions.get('race_outcome') else 0
            }
        else:
            raise HTTPException(status_code=404, detail="No race predictions available")
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error retrieving race predictions: {e}")
        raise HTTPException(status_code=500, detail="Error retrieving race predictions")


@app.get("/api/v1/predictions/laptime")
async def get_laptime_predictions(client: redis.Redis = Depends(get_redis_client)):
    """Get lap time predictions for all drivers"""
    
    try:
        predictions_data = client.get('latest_predictions')
        
        if predictions_data:
            predictions = json.loads(predictions_data.decode('utf-8'))
            lap_times = predictions.get('lap_times', [])
            
            return {
                "timestamp": predictions.get('timestamp'),
                "predictions": lap_times,
                "count": len(lap_times)
            }
        else:
            raise HTTPException(status_code=404, detail="No lap time predictions available")
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error retrieving lap time predictions: {e}")
        raise HTTPException(status_code=500, detail="Error retrieving lap time predictions")


@app.get("/api/v1/predictions/pit-strategy")
async def get_pit_strategy_predictions(client: redis.Redis = Depends(get_redis_client)):
    """Get pit strategy recommendations"""
    
    try:
        predictions_data = client.get('latest_predictions')
        
        if predictions_data:
            predictions = json.loads(predictions_data.decode('utf-8'))
            pit_strategies = predictions.get('pit_strategies', [])
            
            return {
                "timestamp": predictions.get('timestamp'),
                "strategies": pit_strategies,
                "count": len(pit_strategies)
            }
        else:
            raise HTTPException(status_code=404, detail="No pit strategy predictions available")
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error retrieving pit strategy predictions: {e}")
        raise HTTPException(status_code=500, detail="Error retrieving pit strategy predictions")


@app.get("/api/v1/predictions/driver/{driver_id}")
async def get_driver_predictions(driver_id: int, client: redis.Redis = Depends(get_redis_client)):
    """Get all predictions for specific driver"""
    
    try:
        driver_key = f"driver_{driver_id}_predictions"
        predictions_data = client.get(driver_key)
        
        if predictions_data:
            return json.loads(predictions_data.decode('utf-8'))
        else:
            raise HTTPException(status_code=404, detail=f"No predictions found for driver {driver_id}")
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error retrieving driver predictions: {e}")
        raise HTTPException(status_code=500, detail="Error retrieving driver predictions")


# Analytics Endpoints
@app.get("/api/v1/analytics/performance")
async def get_performance_analytics(client: redis.Redis = Depends(get_redis_client)):
    """Get system performance analytics"""
    
    try:
        metrics_data = client.get('performance_metrics')
        
        if metrics_data:
            return json.loads(metrics_data.decode('utf-8'))
        else:
            return {
                "message": "No performance metrics available",
                "timestamp": datetime.now().isoformat()
            }
    
    except Exception as e:
        logger.error(f"❌ Error retrieving performance analytics: {e}")
        raise HTTPException(status_code=500, detail="Error retrieving performance analytics")


@app.get("/api/v1/analytics/driver/{driver_id}")
async def get_driver_analytics(driver_id: int, client: redis.Redis = Depends(get_redis_client)):
    """Get performance analytics for specific driver"""
    
    try:
        # This would typically aggregate historical data
        # For now, return current performance metrics
        driver_key = f"driver_{driver_id}_analytics"
        analytics_data = client.get(driver_key)
        
        if analytics_data:
            return json.loads(analytics_data.decode('utf-8'))
        else:
            # Generate basic analytics from current predictions
            predictions_key = f"driver_{driver_id}_predictions"
            predictions_data = client.get(predictions_key)
            
            if predictions_data:
                predictions = json.loads(predictions_data.decode('utf-8'))
                
                return {
                    "driver_id": driver_id,
                    "current_performance": predictions,
                    "analytics": {
                        "avg_lap_time": predictions.get('predicted_lap_time', 0),
                        "consistency_rating": 0.85,  # Placeholder
                        "tire_management": "good",    # Placeholder
                        "fuel_efficiency": "medium"  # Placeholder
                    },
                    "timestamp": datetime.now().isoformat()
                }
            else:
                raise HTTPException(status_code=404, detail=f"No analytics data for driver {driver_id}")
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error retrieving driver analytics: {e}")
        raise HTTPException(status_code=500, detail="Error retrieving driver analytics")


# WebSocket Endpoints
@app.websocket("/ws/telemetry")
async def websocket_telemetry(websocket: WebSocket):
    """WebSocket endpoint for real-time telemetry streaming"""
    
    await websocket_manager.connect(websocket)
    
    try:
        while True:
            # Keep connection alive and handle incoming messages if needed
            message = await websocket.receive_text()
            
            # Handle client messages (e.g., subscription preferences)
            try:
                data = json.loads(message)
                command = data.get('command')
                
                if command == 'ping':
                    await websocket.send_text(json.dumps({'type': 'pong', 'timestamp': time.time()}))
                elif command == 'subscribe':
                    # Handle subscription to specific data types
                    await websocket.send_text(json.dumps({'type': 'subscribed', 'data': data.get('data_types', [])}))
                
            except json.JSONDecodeError:
                # Invalid JSON, ignore
                pass
    
    except WebSocketDisconnect:
        websocket_manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"❌ WebSocket error: {e}")
        websocket_manager.disconnect(websocket)


@app.websocket("/ws/predictions")
async def websocket_predictions(websocket: WebSocket):
    """WebSocket endpoint for real-time predictions streaming"""
    
    await websocket_manager.connect(websocket)
    
    try:
        while True:
            # Send predictions periodically
            if redis_client:
                predictions_data = redis_client.get('latest_predictions')
                if predictions_data:
                    await websocket.send_text(predictions_data.decode('utf-8'))
            
            await asyncio.sleep(1)  # 1Hz for predictions
    
    except WebSocketDisconnect:
        websocket_manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"❌ WebSocket predictions error: {e}")
        websocket_manager.disconnect(websocket)


# Model Management Endpoints
@app.post("/api/v1/models/retrain")
async def trigger_model_retrain():
    """Trigger model retraining (placeholder for future implementation)"""
    
    # This would trigger background model retraining
    return {
        "message": "Model retraining triggered",
        "timestamp": datetime.now().isoformat(),
        "status": "queued"
    }


@app.get("/api/v1/models/status")
async def get_model_status():
    """Get status of ML models"""
    
    return {
        "models": {
            "race_outcome": {
                "status": "loaded",
                "last_trained": "2024-01-01T00:00:00Z",
                "accuracy": 0.85
            },
            "lap_time": {
                "status": "loaded", 
                "last_trained": "2024-01-01T00:00:00Z",
                "rmse": 0.2
            },
            "pit_strategy": {
                "status": "loaded",
                "last_trained": "2024-01-01T00:00:00Z",
                "precision": 0.78
            }
        },
        "timestamp": datetime.now().isoformat()
    }


# Error handlers
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler"""
    logger.error(f"❌ Unhandled exception: {exc}")
    
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": "An unexpected error occurred",
            "timestamp": datetime.now().isoformat()
        }
    )


if __name__ == "__main__":
    import uvicorn
    
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Run the API server
    uvicorn.run(
        "main:app",
        host=config.API_HOST,
        port=config.API_PORT,
        reload=config.DEBUG,
        log_level=config.LOG_LEVEL.lower()
    )