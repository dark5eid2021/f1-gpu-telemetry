"""
OpenF1 API client for real-time F1 data ingestion
Handles streaming telemetry data from the OpenF1 API
"""

import asyncio
import aiohttp
import json
import logging
from typing import Dict, List, Optional, AsyncGenerator, Callable
from datetime import datetime, timedelta
import time

from kafka import KafkaProducer
from config.config import get_config


class OpenF1Client:
    """Client for OpenF1 API real-time data"""
    
    def __init__(self, api_key: str, base_url: str = "https://api.openf1.org/v1"):
        """
        Initialize OpenF1 client
        
        Args:
            api_key: OpenF1 API key
            base_url: OpenF1 API base URL
        """
        self.api_key = api_key
        self.base_url = base_url
        self.session: Optional[aiohttp.ClientSession] = None
        self.logger = logging.getLogger(__name__)
        
        # Rate limiting
        self.last_request_time = 0
        self.min_request_interval = 0.02  # 50Hz max
        
        # Session tracking
        self.current_session_key: Optional[str] = None
        self.session_info: Dict = {}
    
    async def __aenter__(self):
        """Async context manager entry"""
        self.session = aiohttp.ClientSession(
            headers={'Authorization': f'Bearer {self.api_key}'},
            timeout=aiohttp.ClientTimeout(total=30)
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
    
    async def get_latest_session(self) -> Optional[Dict]:
        """Get the most recent F1 session"""
        try:
            async with self.session.get(f"{self.base_url}/sessions") as response:
                if response.status == 200:
                    sessions = await response.json()
                    if sessions:
                        # Get the most recent session
                        latest_session = max(sessions, key=lambda x: x.get('date_start', ''))
                        self.current_session_key = latest_session.get('session_key')
                        self.session_info = latest_session
                        
                        self.logger.info(f"📡 Latest session: {latest_session.get('session_name', 'Unknown')} "
                                       f"at {latest_session.get('location', 'Unknown')}")
                        return latest_session
                else:
                    self.logger.error(f"❌ Failed to fetch sessions: {response.status}")
        except Exception as e:
            self.logger.error(f"❌ Error fetching latest session: {e}")
        
        return None
    
    async def get_car_data(self, session_key: Optional[str] = None, 
                          driver_number: Optional[int] = None) -> List[Dict]:
        """
        Get real-time car telemetry data
        
        Args:
            session_key: Session identifier (uses current if None)
            driver_number: Specific driver number (all drivers if None)
            
        Returns:
            List of telemetry data points
        """
        await self._rate_limit()
        
        session_key = session_key or self.current_session_key
        if not session_key:
            self.logger.warning("⚠️ No session key available")
            return []
        
        params = {'session_key': session_key}
        if driver_number:
            params['driver_number'] = driver_number
        
        try:
            async with self.session.get(f"{self.base_url}/car_data", params=params) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    self.logger.warning(f"⚠️ Car data request failed: {response.status}")
                    return []
        except Exception as e:
            self.logger.error(f"❌ Error fetching car data: {e}")
            return []
    
    async def get_position_data(self, session_key: Optional[str] = None) -> List[Dict]:
        """Get real-time position data for all drivers"""
        await self._rate_limit()
        
        session_key = session_key or self.current_session_key
        if not session_key:
            return []
        
        try:
            async with self.session.get(f"{self.base_url}/position", 
                                      params={'session_key': session_key}) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    return []
        except Exception as e:
            self.logger.error(f"❌ Error fetching position data: {e}")
            return []
    
    async def get_lap_data(self, session_key: Optional[str] = None) -> List[Dict]:
        """Get lap timing data"""
        await self._rate_limit()
        
        session_key = session_key or self.current_session_key
        if not session_key:
            return []
        
        try:
            async with self.session.get(f"{self.base_url}/laps", 
                                      params={'session_key': session_key}) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    return []
        except Exception as e:
            self.logger.error(f"❌ Error fetching lap data: {e}")
            return []
    
    async def stream_telemetry(self, callback: Callable[[Dict], None], 
                              session_key: Optional[str] = None) -> None:
        """
        Stream real-time telemetry data
        
        Args:
            callback: Function to call with each telemetry update
            session_key: Session to stream (uses current if None)
        """
        session_key = session_key or self.current_session_key
        if not session_key:
            self.logger.error("❌ No session key for streaming")
            return
        
        self.logger.info(f"📡 Starting telemetry stream for session {session_key}")
        
        while True:
            try:
                # Get latest telemetry data
                car_data = await self.get_car_data(session_key)
                position_data = await self.get_position_data(session_key)
                
                # Combine and process data
                for car_point in car_data:
                    # Enrich with position data
                    driver_number = car_point.get('driver_number')
                    position_point = next(
                        (p for p in position_data if p.get('driver_number') == driver_number),
                        {}
                    )
                    
                    # Merge data
                    telemetry_point = {**car_point, **position_point}
                    
                    # Add timestamp
                    telemetry_point['ingestion_timestamp'] = time.time()
                    
                    # Call the callback function
                    callback(telemetry_point)
                
                # Wait before next poll (50Hz rate)
                await asyncio.sleep(0.02)
                
            except asyncio.CancelledError:
                self.logger.info("🛑 Telemetry streaming cancelled")
                break
            except Exception as e:
                self.logger.error(f"❌ Error in telemetry stream: {e}")
                await asyncio.sleep(1)  # Wait before retrying
    
    async def _rate_limit(self):
        """Implement rate limiting to avoid API limits"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.min_request_interval:
            await asyncio.sleep(self.min_request_interval - time_since_last)
        
        self.last_request_time = time.time()


class F1DataIngester:
    """Real-time F1 data ingestion pipeline with Kafka integration"""
    
    def __init__(self, kafka_config: Dict, openf1_config: Dict):
        """
        Initialize F1 data ingester
        
        Args:
            kafka_config: Kafka configuration
            openf1_config: OpenF1 API configuration
        """
        self.kafka_config = kafka_config
        self.openf1_config = openf1_config
        self.logger = logging.getLogger(__name__)
        
        # Initialize Kafka producer
        self.producer = KafkaProducer(
            bootstrap_servers=kafka_config['bootstrap_servers'],
            value_serializer=lambda v: json.dumps(v, default=str).encode('utf-8'),
            key_serializer=lambda k: str(k).encode('utf-8') if k else None,
            acks='all',  # Wait for all replicas
            retries=3,
            batch_size=16384,  # Batch for efficiency
            linger_ms=10,  # Small delay for batching
        )
        
        # Topics
        self.telemetry_topic = 'f1-telemetry'
        self.positions_topic = 'f1-positions'
        self.laps_topic = 'f1-laps'
        
        # Statistics
        self.messages_sent = 0
        self.start_time = time.time()
    
    async def start_ingestion(self, session_key: Optional[str] = None):
        """Start the data ingestion pipeline"""
        
        self.logger.info("🚀 Starting F1 data ingestion pipeline...")
        
        async with OpenF1Client(
            api_key=self.openf1_config['api_key'],
            base_url=self.openf1_config.get('base_url', 'https://api.openf1.org/v1')
        ) as client:
            
            # Get current session if not provided
            if not session_key:
                session = await client.get_latest_session()
                if session:
                    session_key = session.get('session_key')
                else:
                    self.logger.error("❌ No active F1 session found")
                    return
            
            # Start telemetry streaming
            await client.stream_telemetry(
                callback=self._handle_telemetry_data,
                session_key=session_key
            )
    
    def _handle_telemetry_data(self, telemetry_point: Dict):
        """Handle incoming telemetry data point"""
        
        try:
            # Extract driver number for partitioning
            driver_number = telemetry_point.get('driver_number')
            
            # Send to Kafka with driver number as key for partitioning
            future = self.producer.send(
                self.telemetry_topic,
                key=driver_number,
                value=telemetry_point
            )
            
            # Add callback for delivery confirmation
            future.add_callback(self._on_send_success)
            future.add_errback(self._on_send_error)
            
            self.messages_sent += 1
            
            # Log progress periodically
            if self.messages_sent % 1000 == 0:
                elapsed = time.time() - self.start_time
                rate = self.messages_sent / elapsed
                self.logger.info(f"📊 Ingested {self.messages_sent} messages "
                               f"({rate:.1f} msg/sec)")
        
        except Exception as e:
            self.logger.error(f"❌ Error handling telemetry data: {e}")
    
    def _on_send_success(self, record_metadata):
        """Callback for successful message delivery"""
        pass  # Could log detailed delivery info if needed
    
    def _on_send_error(self, exception):
        """Callback for message delivery errors"""
        self.logger.error(f"❌ Kafka send error: {exception}")
    
    async def ingest_historical_data(self, file_path: str):
        """Ingest historical data from file (for testing/replay)"""
        
        self.logger.info(f"📂 Ingesting historical data from {file_path}")
        
        try:
            import pandas as pd
            
            # Read historical data
            if file_path.endswith('.parquet'):
                df = pd.read_parquet(file_path)
            elif file_path.endswith('.csv'):
                df = pd.read_csv(file_path)
            else:
                raise ValueError(f"Unsupported file format: {file_path}")
            
            # Send each row as a telemetry point
            for _, row in df.iterrows():
                telemetry_point = row.to_dict()
                telemetry_point['ingestion_timestamp'] = time.time()
                
                self._handle_telemetry_data(telemetry_point)
                
                # Small delay to simulate real-time
                await asyncio.sleep(0.02)  # 50Hz
            
            self.logger.info(f"✅ Historical data ingestion complete: {len(df)} records")
            
        except Exception as e:
            self.logger.error(f"❌ Error ingesting historical data: {e}")
    
    def get_stats(self) -> Dict[str, float]:
        """Get ingestion statistics"""
        elapsed = time.time() - self.start_time
        return {
            'messages_sent': self.messages_sent,
            'elapsed_time': elapsed,
            'messages_per_second': self.messages_sent / elapsed if elapsed > 0 else 0
        }
    
    def close(self):
        """Close the ingester and cleanup resources"""
        if self.producer:
            self.producer.flush()
            self.producer.close()
        self.logger.info("🛑 F1 data ingester closed")


# Example usage and testing
async def main():
    """Example usage of F1DataIngester"""
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Load configuration
    config = get_config()
    
    # Kafka configuration
    kafka_config = config.get_kafka_config()
    
    # OpenF1 configuration
    openf1_config = {
        'api_key': config.OPENF1_API_KEY,
        'base_url': config.OPENF1_BASE_URL
    }
    
    # Create and start ingester
    ingester = F1