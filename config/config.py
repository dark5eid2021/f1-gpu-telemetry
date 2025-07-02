# ============================================================================
# config/config.py - Main Configuration Management
# ============================================================================

import os
from typing import Optional, Dict, Any
from pathlib import Path
from dotenv import load_dotenv
import logging

# Load environment variables from .env file
env_file = Path(__file__).parent.parent / '.env'
if env_file.exists():
    load_dotenv(env_file)

class Config:
    """Base configuration class that loads from environment variables"""
    
    # ========================================================================
    # F1 API Configuration
    # ========================================================================
    OPENF1_API_KEY: str = os.getenv('OPENF1_API_KEY', '')
    OPENF1_BASE_URL: str = os.getenv('OPENF1_BASE_URL', 'https://api.openf1.org/v1')
    
    # Weather API (Optional)
    WEATHER_API_KEY: str = os.getenv('WEATHER_API_KEY', '')
    WEATHER_API_URL: str = os.getenv('WEATHER_API_URL', 'https://api.openweathermap.org/data/2.5')
    
    # ========================================================================
    # Database Configuration
    # ========================================================================
    DATABASE_URL: str = os.getenv('DATABASE_URL', 'postgresql://f1user:password@localhost:5432/f1_telemetry')
    POSTGRES_USER: str = os.getenv('POSTGRES_USER', 'f1user')
    POSTGRES_PASSWORD: str = os.getenv('POSTGRES_PASSWORD', 'password')
    POSTGRES_DB: str = os.getenv('POSTGRES_DB', 'f1_telemetry')
    POSTGRES_HOST: str = os.getenv('POSTGRES_HOST', 'localhost')
    POSTGRES_PORT: int = int(os.getenv('POSTGRES_PORT', '5432'))
    
    # ========================================================================
    # Redis Configuration
    # ========================================================================
    REDIS_HOST: str = os.getenv('REDIS_HOST', 'localhost')
    REDIS_PORT: int = int(os.getenv('REDIS_PORT', '6379'))
    REDIS_PASSWORD: Optional[str] = os.getenv('REDIS_PASSWORD')
    REDIS_DB: int = int(os.getenv('REDIS_DB', '0'))
    REDIS_SSL: bool = os.getenv('REDIS_SSL', 'false').lower() == 'true'
    
    # ========================================================================
    # Kafka Configuration
    # ========================================================================
    KAFKA_BOOTSTRAP_SERVERS: str = os.getenv('KAFKA_BOOTSTRAP_SERVERS', 'localhost:9092')
    KAFKA_SECURITY_PROTOCOL: str = os.getenv('KAFKA_SECURITY_PROTOCOL', 'PLAINTEXT')
    KAFKA_SASL_USERNAME: Optional[str] = os.getenv('KAFKA_SASL_USERNAME')
    KAFKA_SASL_PASSWORD: Optional[str] = os.getenv('KAFKA_SASL_PASSWORD')
    KAFKA_SSL_CA_LOCATION: Optional[str] = os.getenv('KAFKA_SSL_CA_LOCATION')
    
    # Kafka Topics
    KAFKA_TELEMETRY_TOPIC: str = os.getenv('KAFKA_TELEMETRY_TOPIC', 'f1-telemetry')
    KAFKA_PREDICTIONS_TOPIC: str = os.getenv('KAFKA_PREDICTIONS_TOPIC', 'f1-predictions')
    KAFKA_POSITIONS_TOPIC: str = os.getenv('KAFKA_POSITIONS_TOPIC', 'f1-positions')
    
    # ========================================================================
    # GPU Configuration
    # ========================================================================
    CUDA_VISIBLE_DEVICES: str = os.getenv('CUDA_VISIBLE_DEVICES', '0')
    GPU_BATCH_SIZE: int = int(os.getenv('GPU_BATCH_SIZE', '1024'))
    GPU_MEMORY_FRACTION: float = float(os.getenv('GPU_MEMORY_FRACTION', '0.8'))
    GPU_ALLOW_GROWTH: bool = os.getenv('GPU_ALLOW_GROWTH', 'true').lower() == 'true'
    
    # ========================================================================
    # Application Configuration
    # ========================================================================
    API_HOST: str = os.getenv('API_HOST', '0.0.0.0')
    API_PORT: int = int(os.getenv('API_PORT', '8000'))
    DEBUG: bool = os.getenv('DEBUG', 'false').lower() == 'true'
    LOG_LEVEL: str = os.getenv('LOG_LEVEL', 'INFO')
    ENVIRONMENT: str = os.getenv('ENVIRONMENT', 'development')
    
    # Application metadata
    APP_NAME: str = "F1 GPU Telemetry System"
    APP_VERSION: str = "1.0.0"
    APP_DESCRIPTION: str = "Real-time F1 telemetry processing and race prediction"
    
    # ========================================================================
    # Security Configuration
    # ========================================================================
    JWT_SECRET_KEY: str = os.getenv('JWT_SECRET_KEY', 'your-secret-key-change-this')
    JWT_ALGORITHM: str = os.getenv('JWT_ALGORITHM', 'HS256')
    JWT_EXPIRATION_HOURS: int = int(os.getenv('JWT_EXPIRATION_HOURS', '24'))
    
    API_SECRET_KEY: str = os.getenv('API_SECRET_KEY', 'your-api-secret-change-this')
    ENCRYPTION_KEY: str = os.getenv('ENCRYPTION_KEY', 'your-32-char-encryption-key-here')
    
    # CORS settings
    CORS_ORIGINS: list = os.getenv('CORS_ORIGINS', '*').split(',')
    CORS_ALLOW_CREDENTIALS: bool = os.getenv('CORS_ALLOW_CREDENTIALS', 'true').lower() == 'true'
    
    # ========================================================================
    # Monitoring Configuration
    # ========================================================================
    PROMETHEUS_ENABLED: bool = os.getenv('PROMETHEUS_ENABLED', 'true').lower() == 'true'
    PROMETHEUS_PORT: int = int(os.getenv('PROMETHEUS_PORT', '8001'))
    
    GRAFANA_ADMIN_PASSWORD: str = os.getenv('GRAFANA_ADMIN_PASSWORD', 'admin123')
    
    # Logging configuration
    LOG_FORMAT: str = os.getenv('LOG_FORMAT', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    LOG_FILE: Optional[str] = os.getenv('LOG_FILE')
    LOG_MAX_SIZE: int = int(os.getenv('LOG_MAX_SIZE', '10485760'))  # 10MB
    LOG_BACKUP_COUNT: int = int(os.getenv('LOG_BACKUP_COUNT', '5'))
    
    # ========================================================================
    # Cloud Provider Configuration
    # ========================================================================
    # AWS
    AWS_ACCESS_KEY_ID: Optional[str] = os.getenv('AWS_ACCESS_KEY_ID')
    AWS_SECRET_ACCESS_KEY: Optional[str] = os.getenv('AWS_SECRET_ACCESS_KEY')
    AWS_REGION: str = os.getenv('AWS_REGION', 'us-west-2')
    AWS_S3_BUCKET: Optional[str] = os.getenv('AWS_S3_BUCKET')
    
    # GCP
    GCP_SERVICE_ACCOUNT_KEY_PATH: Optional[str] = os.getenv('GCP_SERVICE_ACCOUNT_KEY_PATH')
    GCP_PROJECT_ID: Optional[str] = os.getenv('GCP_PROJECT_ID')
    GCP_STORAGE_BUCKET: Optional[str] = os.getenv('GCP_STORAGE_BUCKET')
    
    # ========================================================================
    # Data Processing Configuration
    # ========================================================================
    DATA_DIR: str = os.getenv('DATA_DIR', 'data')
    CACHE_DIR: str = os.getenv('CACHE_DIR', 'data/cache')
    MODEL_DIR: str = os.getenv('MODEL_DIR', 'data/models')
    LOG_DIR: str = os.getenv('LOG_DIR', 'logs')
    
    # Data update settings
    DATA_UPDATE_INTERVAL: str = os.getenv('DATA_UPDATE_INTERVAL', 'weekly')
    DATA_RETENTION_DAYS: int = int(os.getenv('DATA_RETENTION_DAYS', '1095'))  # 3 years
    
    # FastF1 settings
    FASTF1_CACHE_ENABLED: bool = os.getenv('FASTF1_CACHE_ENABLED', 'true').lower() == 'true'
    
    # ========================================================================
    # Performance Configuration
    # ========================================================================
    MAX_WORKERS: int = int(os.getenv('MAX_WORKERS', '4'))
    TELEMETRY_BUFFER_SIZE: int = int(os.getenv('TELEMETRY_BUFFER_SIZE', '10000'))
    PREDICTION_CACHE_TTL: int = int(os.getenv('PREDICTION_CACHE_TTL', '60'))  # seconds
    
    # Rate limiting
    RATE_LIMIT_REQUESTS: int = int(os.getenv('RATE_LIMIT_REQUESTS', '100'))
    RATE_LIMIT_WINDOW: int = int(os.getenv('RATE_LIMIT_WINDOW', '60'))  # seconds
    
    # ========================================================================
    # Container & Orchestration
    # ========================================================================
    KUBERNETES_NAMESPACE: str = os.getenv('KUBERNETES_NAMESPACE', 'f1-system')
    DOCKER_REGISTRY: str = os.getenv('DOCKER_REGISTRY', 'docker.io')
    DOCKER_USERNAME: Optional[str] = os.getenv('DOCKER_USERNAME')
    DOCKER_PASSWORD: Optional[str] = os.getenv('DOCKER_PASSWORD')
    
    @classmethod
    def validate_required_secrets(cls) -> bool:
        """Validate that all required secrets are present"""
        required_secrets = [
            cls.OPENF1_API_KEY,
            cls.POSTGRES_PASSWORD,
            cls.JWT_SECRET_KEY,
            cls.API_SECRET_KEY,
            cls.ENCRYPTION_KEY
        ]
        
        missing_secrets = []
        placeholder_secrets = []
        
        for secret in required_secrets:
            if not secret:
                missing_secrets.append(secret)
            elif any(placeholder in secret.lower() for placeholder in ['your_', 'change', 'here', 'placeholder']):
                placeholder_secrets.append(secret)
        
        if missing_secrets or placeholder_secrets:
            logger = logging.getLogger(__name__)
            if missing_secrets:
                logger.error(f"❌ Missing required secrets: {len(missing_secrets)} secrets not configured")
            if placeholder_secrets:
                logger.error(f"❌ Placeholder values found: {len(placeholder_secrets)} secrets need real values")
            logger.error("Please check your .env file and ensure all required values are set")
            return False
        
        return True
    
    @classmethod
    def get_redis_config(cls) -> Dict[str, Any]:
        """Get Redis connection configuration"""
        config = {
            'host': cls.REDIS_HOST,
            'port': cls.REDIS_PORT,
            'db': cls.REDIS_DB,
            'decode_responses': True,
            'socket_connect_timeout': 5,
            'socket_timeout': 5,
            'retry_on_timeout': True
        }
        
        if cls.REDIS_PASSWORD:
            config['password'] = cls.REDIS_PASSWORD
        
        if cls.REDIS_SSL:
            config['ssl'] = True
        
        return config
    
    @classmethod
    def get_kafka_config(cls) -> Dict[str, Any]:
        """Get Kafka connection configuration"""
        config = {
            'bootstrap_servers': cls.KAFKA_BOOTSTRAP_SERVERS.split(','),
            'security_protocol': cls.KAFKA_SECURITY_PROTOCOL,
            'client_id': 'f1-gpu-telemetry',
            'api_version': (2, 0, 2)
        }
        
        if cls.KAFKA_SASL_USERNAME and cls.KAFKA_SASL_PASSWORD:
            config.update({
                'sasl_mechanism': 'PLAIN',
                'sasl_plain_username': cls.KAFKA_SASL_USERNAME,
                'sasl_plain_password': cls.KAFKA_SASL_PASSWORD
            })
        
        if cls.KAFKA_SSL_CA_LOCATION:
            config['ssl_ca_location'] = cls.KAFKA_SSL_CA_LOCATION
        
        return config
    
    @classmethod
    def get_database_url(cls) -> str:
        """Get complete database URL"""
        if cls.DATABASE_URL:
            return cls.DATABASE_URL
        
        return f"postgresql://{cls.POSTGRES_USER}:{cls.POSTGRES_PASSWORD}@{cls.POSTGRES_HOST}:{cls.POSTGRES_PORT}/{cls.POSTGRES_DB}"
    
    @classmethod
    def setup_logging(cls):
        """Setup application logging"""
        log_level = getattr(logging, cls.LOG_LEVEL.upper(), logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter(cls.LOG_FORMAT)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        
        # File handler (if specified)
        handlers = [console_handler]
        if cls.LOG_FILE:
            from logging.handlers import RotatingFileHandler
            file_handler = RotatingFileHandler(
                cls.LOG_FILE,
                maxBytes=cls.LOG_MAX_SIZE,
                backupCount=cls.LOG_BACKUP_COUNT
            )
            file_handler.setFormatter(formatter)
            handlers.append(file_handler)
        
        # Configure root logger
        logging.basicConfig(
            level=log_level,
            handlers=handlers,
            force=True
        )


class DevelopmentConfig(Config):
    """Development configuration with debug settings"""
    DEBUG = True
    LOG_LEVEL = 'DEBUG'
    ENVIRONMENT = 'development'
    
    # More permissive CORS for development
    CORS_ORIGINS = ['http://localhost:3000', 'http://127.0.0.1:3000']
    
    # Smaller batch sizes for development
    GPU_BATCH_SIZE = 512
    TELEMETRY_BUFFER_SIZE = 1000


class ProductionConfig(Config):
    """Production configuration with security settings"""
    DEBUG = False
    LOG_LEVEL = 'INFO'
    ENVIRONMENT = 'production'
    
    # Stricter settings for production
    JWT_EXPIRATION_HOURS = 8
    RATE_LIMIT_REQUESTS = 1000
    RATE_LIMIT_WINDOW = 3600  # 1 hour
    
    # Larger batch sizes for production
    GPU_BATCH_SIZE = 2048
    TELEMETRY_BUFFER_SIZE = 50000


class TestingConfig(Config):
    """Testing configuration"""
    DEBUG = True
    LOG_LEVEL = 'DEBUG'
    ENVIRONMENT = 'testing'
    
    # Test database
    POSTGRES_DB = 'f1_telemetry_test'
    DATABASE_URL = 'postgresql://test:test@localhost:5432/f1_telemetry_test'
    
    # Disable external services in tests
    PROMETHEUS_ENABLED = False
    FASTF1_CACHE_ENABLED = False
    
    # Small batch sizes for fast tests
    GPU_BATCH_SIZE = 32
    TELEMETRY_BUFFER_SIZE = 100


# Configuration factory
def get_config() -> Config:
    """Get configuration based on environment"""
    env = os.getenv('ENVIRONMENT', 'development').lower()
    
    if env == 'production':
        return ProductionConfig()
    elif env == 'testing':
        return TestingConfig()
    else:
        return DevelopmentConfig()
