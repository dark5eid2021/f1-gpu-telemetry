#!/usr/bin/env python3
"""
F1 Real-Time GPU Telemetry & Race Prediction System
Main Application Entry Point

This is the main entry point that starts the entire F1 GPU telemetry system.
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add src to Python path
sys.path.append(str(Path(__file__).parent))

import torch
from config.config import get_config
from api.main import app
from inference.gpu_pipeline import RealTimeInferencePipeline
from ml.train_models import GPUMLPipeline

import uvicorn


async def main():
    """Main application entry point"""
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    logger.info("🏎️ Starting F1 GPU Telemetry System...")
    
    # Load configuration
    config = get_config()
    
    # Validate configuration
    if not config.validate_required_secrets():
        logger.error("❌ Configuration validation failed!")
        logger.error("Please check your .env file and ensure all required values are set")
        sys.exit(1)
    
    # Check GPU availability
    if not torch.cuda.is_available():
        logger.warning("⚠️ CUDA GPU not available! Running in CPU mode.")
        logger.warning("For optimal performance, please ensure NVIDIA GPU and drivers are installed")
    else:
        gpu_count = torch.cuda.device_count()
        gpu_name = torch.cuda.get_device_name(0)
        logger.info(f"✅ Found {gpu_count} GPU(s): {gpu_name}")
    
    try:
        # Initialize ML pipeline
        logger.info("🧠 Initializing ML pipeline...")
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        ml_pipeline = GPUMLPipeline(device=device)
        
        # Initialize real-time inference pipeline
        logger.info("⚡ Setting up real-time inference pipeline...")
        redis_config = config.get_redis_config()
        inference_pipeline = RealTimeInferencePipeline(ml_pipeline, redis_config)
        
        # Start telemetry processing in background
        logger.info("📡 Starting telemetry processing...")
        asyncio.create_task(inference_pipeline.process_telemetry_stream())
        
        # Start the FastAPI server
        logger.info("🌐 Starting API server...")
        server_config = uvicorn.Config(
            app, 
            host=config.API_HOST, 
            port=config.API_PORT, 
            log_level=config.LOG_LEVEL.lower()
        )
        server = uvicorn.Server(server_config)
        
        logger.info(f"🚀 F1 GPU Telemetry System running at http://{config.API_HOST}:{config.API_PORT}")
        logger.info(f"📊 API Documentation: http://{config.API_HOST}:{config.API_PORT}/docs")
        
        await server.serve()
        
    except KeyboardInterrupt:
        logger.info("🛑 Shutting down F1 GPU Telemetry System...")
    except Exception as e:
        logger.error(f"❌ Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())