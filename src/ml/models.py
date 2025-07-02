"""
Machine Learning Models for F1 Race Prediction
Contains PyTorch and RAPIDS models for various prediction tasks
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

try:
    import cuml
    from cuml.ensemble import RandomForestRegressor
    from cuml.linear_model import LogisticRegression
    RAPIDS_AVAILABLE = True
except ImportError:
    RAPIDS_AVAILABLE = False
    logging.warning("⚠️ RAPIDS not available, using CPU fallbacks")


@dataclass
class ModelConfig:
    """Configuration for ML models"""
    input_dim: int = 64
    hidden_dim: int = 256
    num_drivers: int = 20
    num_layers: int = 6
    num_heads: int = 8
    dropout: float = 0.1
    sequence_length: int = 50


class RaceOutcomePredictor(nn.Module):
    """
    Transformer-based race outcome prediction model
    Predicts finishing position probabilities for all drivers
    """
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.input_dim = config.input_dim
        self.hidden_dim = config.hidden_dim
        self.num_drivers = config.num_drivers
        
        # Embedding layers
        self.position_embed = nn.Embedding(config.num_drivers, config.hidden_dim)
        self.feature_proj = nn.Linear(config.input_dim, config.hidden_dim)
        
        # Positional encoding for sequence data
        self.pos_encoding = self._create_positional_encoding(
            config.sequence_length, config.hidden_dim
        )
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_dim,
            nhead=config.num_heads,
            dim_feedforward=config.hidden_dim * 4,
            dropout=config.dropout,
            batch_first=True,
            activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=config.num_layers)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(config.hidden_dim)
        
        # Output heads for different predictions
        self.position_head = nn.Linear(config.hidden_dim, config.num_drivers)  # Finishing positions
        self.time_head = nn.Linear(config.hidden_dim, 1)  # Lap time prediction
        self.pit_head = nn.Linear(config.hidden_dim, 2)  # Pit stop probability
        
        # Dropout for regularization
        self.dropout = nn.Dropout(config.dropout)
        
    def _create_positional_encoding(self, max_len: int, d_model: int) -> torch.Tensor:
        """Create sinusoidal positional encoding"""
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        return pe.unsqueeze(0)  # Add batch dimension
    
    def forward(self, features: torch.Tensor, positions: torch.Tensor, 
                mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass for race prediction
        
        Args:
            features: Input features [batch_size, seq_len, input_dim]
            positions: Current track positions [batch_size, seq_len]
            mask: Attention mask for padding [batch_size, seq_len]
            
        Returns:
            Dictionary containing different predictions
        """
        batch_size, seq_len = features.shape[:2]
        device = features.device
        
        # Project features to hidden dimension
        feat_proj = self.feature_proj(features)  # [batch, seq, hidden]
        
        # Add position embeddings
        pos_embed = self.position_embed(positions)  # [batch, seq, hidden]
        
        # Add positional encoding
        pos_encoding = self.pos_encoding[:, :seq_len, :].to(device)
        
        # Combine all embeddings
        x = feat_proj + pos_embed + pos_encoding
        x = self.layer_norm(x)
        x = self.dropout(x)
        
        # Apply transformer encoder
        if mask is not None:
            # Convert mask to attention mask format
            mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        
        encoded = self.transformer(x, src_key_padding_mask=mask)
        
        # Global average pooling over sequence dimension
        if mask is not None:
            # Masked average pooling
            mask_expanded = (~mask).unsqueeze(-1).float()
            pooled = (encoded * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1)
        else:
            pooled = encoded.mean(dim=1)  # [batch, hidden]
        
        # Generate predictions
        position_logits = self.position_head(pooled)  # [batch, num_drivers]
        lap_time_pred = self.time_head(pooled)  # [batch, 1]
        pit_logits = self.pit_head(pooled)  # [batch, 2]
        
        return {
            'position_probs': F.softmax(position_logits, dim=-1),
            'position_logits': position_logits,
            'lap_time': lap_time_pred.squeeze(-1),
            'pit_probs': F.softmax(pit_logits, dim=-1),
            'pit_logits': pit_logits,
            'encoded_features': pooled
        }


class LapTimePredictor(nn.Module):
    """
    LSTM-based lap time prediction model
    Predicts next lap time based on current telemetry sequence
    """
    
    def __init__(self, input_dim: int = 32, hidden_dim: int = 128, 
                 num_layers: int = 3, dropout: float = 0.1):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=True
        )
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim * 2,  # Bidirectional
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )
        
        # Output layers
        self.output_projection = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
        
    def forward(self, x: torch.Tensor, 
                hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass for lap time prediction
        
        Args:
            x: Input telemetry sequence [batch_size, seq_len, input_dim]
            hidden: Initial hidden state for LSTM
            
        Returns:
            Dictionary containing lap time prediction and attention weights
        """
        batch_size, seq_len = x.shape[:2]
        
        # LSTM forward pass
        lstm_out, (h_n, c_n) = self.lstm(x, hidden)  # [batch, seq, hidden*2]
        
        # Apply attention
        attended, attention_weights = self.attention(lstm_out, lstm_out, lstm_out)
        
        # Use the last timestep for prediction
        last_output = attended[:, -1, :]  # [batch, hidden*2]
        
        # Predict lap time
        lap_time = self.output_projection(last_output)  # [batch, 1]
        
        return {
            'lap_time': lap_time.squeeze(-1),
            'attention_weights': attention_weights,
            'lstm_output': lstm_out,
            'hidden_state': (h_n, c_n)
        }


class PitStrategyOptimizer(nn.Module):
    """
    Deep Q-Network for pit stop strategy optimization
    Learns optimal pit timing based on race state
    """
    
    def __init__(self, state_dim: int = 32, action_dim: int = 10, 
                 hidden_dim: int = 256, num_layers: int = 3):
        super().__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Build DQN network
        layers = []
        in_dim = state_dim
        
        for i in range(num_layers):
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            in_dim = hidden_dim
        
        # Output layer for Q-values
        layers.append(nn.Linear(hidden_dim, action_dim))
        
        self.network = nn.Sequential(*layers)
        
        # Dueling DQN architecture
        self.value_head = nn.Linear(hidden_dim, 1)
        self.advantage_head = nn.Linear(hidden_dim, action_dim)
        
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for Q-value prediction
        
        Args:
            state: Current race state [batch_size, state_dim]
            
        Returns:
            Q-values for each action [batch_size, action_dim]
        """
        # Get features from main network
        features = self.network[:-1](state)  # Remove last layer
        
        # Dueling architecture
        value = self.value_head(features)  # [batch, 1]
        advantage = self.advantage_head(features)  # [batch, action_dim]
        
        # Combine value and advantage
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        
        return q_values


class TirePerformanceModel(nn.Module):
    """
    Model for predicting tire performance and degradation
    """
    
    def __init__(self, input_dim: int = 16, hidden_dim: int = 64):
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.1),
            
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.1),
            
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(hidden_dim // 2, 4)  # [degradation, grip, temperature, wear_rate]
        )
        
    def forward(self, tire_data: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Predict tire performance metrics
        
        Args:
            tire_data: Tire telemetry [batch_size, input_dim]
            
        Returns:
            Dictionary of tire performance predictions
        """
        output = self.network(tire_data)
        
        return {
            'degradation': torch.sigmoid(output[:, 0]),  # 0-1 scale
            'grip': torch.sigmoid(output[:, 1]),         # 0-1 scale
            'temperature': output[:, 2],                 # Raw temperature
            'wear_rate': torch.sigmoid(output[:, 3])     # 0-1 scale
        }


class RAIDSMLPipeline:
    """
    RAPIDS-based ML pipeline for fast data processing
    Uses GPU-accelerated algorithms when available
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        if not RAPIDS_AVAILABLE:
            self.logger.warning("⚠️ RAPIDS not available, using CPU fallbacks")
        
        # Initialize models
        self.models = {}
        self._setup_models()
    
    def _setup_models(self):
        """Setup RAPIDS ML models"""
        
        if RAPIDS_AVAILABLE:
            # Lap time prediction (Random Forest)
            self.models['lap_time'] = RandomForestRegressor(
                n_estimators=100,
                max_depth=20,
                random_state=42,
                n_streams=1  # Use single stream for consistency
            )
            
            # Pit stop strategy (Logistic Regression)  
            self.models['pit_strategy'] = LogisticRegression(
                penalty='l2',
                tol=1e-4,
                max_iter=1000
            )
            
            self.logger.info("✅ RAPIDS models initialized")
        else:
            # CPU fallbacks
            from sklearn.ensemble import RandomForestRegressor as SKRandomForest
            from sklearn.linear_model import LogisticRegression as SKLogisticRegression
            
            self.models['lap_time'] = SKRandomForest(
                n_estimators=100,
                max_depth=20,
                random_state=42,
                n_jobs=-1
            )
            
            self.models['pit_strategy'] = SKLogisticRegression(
                penalty='l2',
                tol=1e-4,
                max_iter=1000
            )
            
            self.logger.info("✅ CPU fallback models initialized")
    
    def train_lap_time_model(self, telemetry_df):
        """Train lap time prediction model"""
        
        # Feature engineering
        feature_columns = ['speed', 'throttle', 'brake', 'tire_temp_avg', 'fuel_load']
        target_column = 'lap_time'
        
        if RAPIDS_AVAILABLE:
            import cudf
            if not isinstance(telemetry_df, cudf.DataFrame):
                telemetry_df = cudf.from_pandas(telemetry_df)
        
        # Prepare features and target
        X = telemetry_df[feature_columns]
        y = telemetry_df[target_column]
        
        # Train model
        self.models['lap_time'].fit(X, y)
        
        self.logger.info("✅ Lap time model trained successfully")
        
        return self.models['lap_time']
    
    def predict_lap_time(self, telemetry_features):
        """Predict lap time from telemetry features"""
        
        if 'lap_time' not in self.models:
            raise ValueError("Lap time model not trained")
        
        return self.models['lap_time'].predict(telemetry_features)
    
    def train_pit_strategy_model(self, strategy_df):
        """Train pit stop strategy model"""
        
        feature_columns = ['lap_number', 'tire_age', 'fuel_remaining', 'position', 'gap_to_leader']
        target_column = 'should_pit'
        
        if RAPIDS_AVAILABLE:
            import cudf
            if not isinstance(strategy_df, cudf.DataFrame):
                strategy_df = cudf.from_pandas(strategy_df)
        
        X = strategy_df[feature_columns]
        y = strategy_df[target_column]
        
        self.models['pit_strategy'].fit(X, y)
        
        self.logger.info("✅ Pit strategy model trained successfully")
        
        return self.models['pit_strategy']
    
    def predict_pit_strategy(self, race_state):
        """Predict optimal pit strategy"""
        
        if 'pit_strategy' not in self.models:
            raise ValueError("Pit strategy model not trained")
        
        return self.models['pit_strategy'].predict_proba(race_state)


# Model factory and utilities
def create_race_outcome_model(config: Optional[ModelConfig] = None) -> RaceOutcomePredictor:
    """Create and initialize race outcome prediction model"""
    if config is None:
        config = ModelConfig()
    
    model = RaceOutcomePredictor(config)
    
    # Initialize weights
    def init_weights(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            torch.nn.init.normal_(m.weight, 0, 0.1)
    
    model.apply(init_weights)
    
    return model


def create_lap_time_model(input_dim: int = 32) -> LapTimePredictor:
    """Create and initialize lap time prediction model"""
    return LapTimePredictor(input_dim=input_dim)


def create_pit_strategy_model(state_dim: int = 32, action_dim: int = 10) -> PitStrategyOptimizer:
    """Create and initialize pit strategy optimization model"""
    return PitStrategyOptimizer(state_dim=state_dim, action_dim=action_dim)