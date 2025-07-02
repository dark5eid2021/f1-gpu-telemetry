# ============================================================================
# config/models.py - Model Configuration
# ============================================================================

from dataclasses import dataclass
from typing import Dict, Any, Optional
import os

@dataclass
class ModelConfig:
    """Configuration for ML models"""
    
    # Model architecture
    input_dim: int = 64
    hidden_dim: int = 256
    num_drivers: int = 20
    num_layers: int = 6
    num_heads: int = 8
    dropout: float = 0.1
    sequence_length: int = 50
    
    # Training parameters
    learning_rate: float = 0.001
    batch_size: int = 32
    num_epochs: int = 100
    early_stopping_patience: int = 10
    
    # Model-specific configurations
    race_outcome_config: Dict[str, Any] = None
    lap_time_config: Dict[str, Any] = None
    pit_strategy_config: Dict[str, Any] = None
    
    def __post_init__(self):
        """Initialize model-specific configurations"""
        if self.race_outcome_config is None:
            self.race_outcome_config = {
                'model_type': 'transformer',
                'hidden_dim': self.hidden_dim,
                'num_layers': self.num_layers,
                'num_heads': self.num_heads,
                'dropout': self.dropout,
                'output_dim': self.num_drivers
            }
        
        if self.lap_time_config is None:
            self.lap_time_config = {
                'model_type': 'lstm',
                'hidden_dim': 128,
                'num_layers': 3,
                'sequence_length': self.sequence_length,
                'bidirectional': True
            }
        
        if self.pit_strategy_config is None:
            self.pit_strategy_config = {
                'model_type': 'dqn',
                'state_dim': 32,
                'action_dim': 10,
                'hidden_dim': 256,
                'num_layers': 3
            }


def get_model_config() -> ModelConfig:
    """Get model configuration from environment or defaults"""
    
    config = ModelConfig()
    
    # Override with environment variables if present
    config.input_dim = int(os.getenv('MODEL_INPUT_DIM', config.input_dim))
    config.hidden_dim = int(os.getenv('MODEL_HIDDEN_DIM', config.hidden_dim))
    config.num_drivers = int(os.getenv('MODEL_NUM_DRIVERS', config.num_drivers))
    config.learning_rate = float(os.getenv('MODEL_LEARNING_RATE', config.learning_rate))
    config.batch_size = int(os.getenv('MODEL_BATCH_SIZE', config.batch_size))
    
    return config