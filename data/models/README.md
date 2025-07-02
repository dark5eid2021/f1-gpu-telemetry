# Models Directory

This directory contains trained machine learning models, checkpoints, and related artifacts for the F1 GPU Telemetry System.

## Directory Structure

```
models/
├── README.md                   # This file
├── production/                 # Production-ready models
│   ├── race_predictor_v1.2.pth
│   ├── lap_time_predictor_v2.1.pth
│   ├── pit_strategy_v1.0.pth
│   └── model_registry.json
├── checkpoints/               # Training checkpoints
│   ├── race_predictor/
│   ├── lap_time_predictor/
│   └── pit_strategy/
├── experiments/               # Experimental models
│   ├── transformer_v3/
│   ├── lstm_attention/
│   └── ensemble_models/
└── metadata/                  # Model metadata and configs
    ├── training_configs/
    ├── performance_metrics/
    └── model_cards/
```

## Model Types

### 1. Race Outcome Predictor
**File:** `production/race_predictor_v1.2.pth`
**Type:** Transformer-based sequence model
**Purpose:** Predicts finishing position probabilities

**Architecture:**
- Input dimension: 64 features
- Hidden dimension: 256
- Layers: 6 transformer encoder layers
- Attention heads: 8
- Output: 20 driver position probabilities

**Performance:**
- Top-3 accuracy: 85.3%
- Top-5 accuracy: 92.1%
- Mean reciprocal rank: 0.74

### 2. Lap Time Predictor
**File:** `production/lap_time_predictor_v2.1.pth`
**Type:** LSTM with attention mechanism
**Purpose:** Predicts next lap time based on telemetry

**Architecture:**
- Input features: 32 telemetry channels
- LSTM layers: 3 bidirectional
- Hidden dimension: 128
- Attention mechanism: Multi-head (8 heads)
- Output: Single lap time prediction

**Performance:**
- RMSE: 0.21 seconds
- MAE: 0.15 seconds
- R²: 0.92

### 3. Pit Strategy Optimizer
**File:** `production/pit_strategy_v1.0.pth`
**Type:** Deep Q-Network (DQN)
**Purpose:** Optimal pit stop timing decisions

**Architecture:**
- State dimension: 32 race state features
- Action dimension: 10 pit timing options
- Hidden layers: 3 layers (256 neurons each)
- Dueling architecture: Value + advantage streams

**Performance:**
- Strategy improvement: 12% better than baseline
- Win rate: 67% in simulated races
- Average position gain: +1.3 positions

## Model Versioning

### Naming Convention
```
ModelType_vMajor.Minor.pth
```

**Examples:**
- `race_predictor_v1.2.pth` - Race predictor, version 1.2
- `lap_time_predictor_v2.1.pth` - Lap time predictor, version 2.1

### Version History
```json
{
  "race_predictor": {
    "v1.0": "Initial transformer model",
    "v1.1": "Added tire degradation features",
    "v1.2": "Improved attention mechanism"
  },
  "lap_time_predictor": {
    "v1.0": "Basic LSTM model",
    "v2.0": "Added bidirectional LSTM",
    "v2.1": "Attention mechanism integration"
  }
}
```

## Model Registry

### production/model_registry.json
Central registry tracking all production models:

```json
{
  "models": {
    "race_predictor": {
      "version": "1.2",
      "file": "race_predictor_v1.2.pth",
      "trained_date": "2024-01-15",
      "training_data": "2018-2023 seasons",
      "accuracy": 0.853,
      "status": "active"
    },
    "lap_time_predictor": {
      "version": "2.1", 
      "file": "lap_time_predictor_v2.1.pth",
      "trained_date": "2024-01-20",
      "training_data": "2020-2023 telemetry",
      "rmse": 0.21,
      "status": "active"
    }
  }
}
```

## Checkpoints

### Training Checkpoints
Stored in `checkpoints/ModelName/` directories:

```
checkpoints/race_predictor/
├── epoch_010.pth
├── epoch_020.pth
├── epoch_030.pth
├── best_model.pth
└── final_model.pth
```

### Checkpoint Metadata
```json
{
  "epoch": 30,
  "train_loss": 0.245,
  "val_loss": 0.289,
  "accuracy": 0.851,
  "learning_rate": 0.0001,
  "timestamp": "2024-01-15T14:30:00Z"
}
```

## Experimental Models

### experiments/ Directory
Contains research and experimental models:

- **transformer_v3/**: Next-generation transformer architecture
- **lstm_attention/**: LSTM with advanced attention mechanisms  
- **ensemble_models/**: Combined model approaches
- **federated_learning/**: Distributed training experiments

### Experiment Tracking
Each experiment includes:
- Model code and configuration
- Training logs and metrics
- Hyperparameter search results
- Performance comparisons

## Model Cards

### metadata/model_cards/
Documentation for each model including:

```markdown
# Race Predictor v1.2 Model Card

## Model Description
Transformer-based model for predicting F1 race finishing positions.

## Training Data
- **Sources**: FastF1, OpenF1 APIs
- **Timeframe**: 2018-2023 F1 seasons
- **Samples**: 450,000 race instances
- **Features**: 64 telemetry and race state features

## Performance Metrics
- **Top-3 Accuracy**: 85.3%
- **Calibration Error**: 0.034
- **Inference Time**: 15ms average

## Limitations
- Performance degrades in wet conditions
- Limited training data for new circuits
- Sensitive to driver changes mid-season

## Ethical Considerations
- No personal driver data used
- Fair representation across all teams
- Bias testing completed
```

## Usage Examples

### Loading Production Models
```python
import torch
from src.ml.models import RaceOutcomePredictor

# Load production model
model = RaceOutcomePredictor()
model.load_state_dict(torch.load('data/models/production/race_predictor_v1.2.pth'))
model.eval()

# Make prediction
with torch.no_grad():
    prediction = model(features, positions)
```

### Model Registry Access
```python
import json

def load_latest_model(model_name):
    with open('data/models/production/model_registry.json', 'r') as f:
        registry = json.load(f)
    
    model_info = registry['models'][model_name]
    model_file = f"data/models/production/{model_info['file']}"
    
    return torch.load(model_file), model_info
```

### Checkpoint Management
```python
def save_checkpoint(model, optimizer, epoch, metrics, filepath):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics,
        'timestamp': datetime.now().isoformat()
    }
    torch.save(checkpoint, filepath)

def load_checkpoint(filepath):
    return torch.load(filepath)
```

## Training Pipeline

### Model Training Process
1. **Data Preparation**: Load and preprocess training data
2. **Model Initialization**: Create model architecture
3. **Training Loop**: Train with validation monitoring
4. **Checkpoint Saving**: Save progress periodically
5. **Model Evaluation**: Test on held-out data
6. **Production Deployment**: Move best model to production

### Training Configuration
```python
training_config = {
    "model_type": "race_predictor",
    "batch_size": 32,
    "learning_rate": 0.001,
    "epochs": 100,
    "validation_split": 0.2,
    "early_stopping_patience": 10,
    "checkpoint_frequency": 10
}
```

## Performance Monitoring

### Model Drift Detection
- **Data Drift**: Monitor input feature distributions
- **Concept Drift**: Track prediction accuracy over time
- **Performance Degradation**: Alert when metrics drop below thresholds

### Retraining Triggers
- Accuracy drops below 80%
- New season data available
- Significant rule changes in F1
- Monthly scheduled retraining

### A/B Testing
- Compare new models against production models
- Gradual rollout of model updates
- Performance metric monitoring
- Rollback capabilities

## Deployment

### Model Serving
```python
class ModelServer:
    def __init__(self):
        self.models = self.load_production_models()
    
    def predict_race_outcome(self, features):
        model = self.models['race_predictor']
        return model(features)
    
    def predict_lap_time(self, telemetry):
        model = self.models['lap_time_predictor'] 
        return model(telemetry)
```

### Model Updates
1. Train new model in experiments/
2. Validate performance against production
3. Update model registry
4. Deploy to production/
5. Monitor performance
6. Rollback if issues detected

## Backup and Recovery

### Model Backup Strategy
- **Daily**: Production model backups
- **Weekly**: Checkpoint backups
- **Monthly**: Complete model archive
- **Quarterly**: Cold storage archival

### Recovery Procedures
- Automatic fallback to previous model version
- Checkpoint recovery for interrupted training
- Model reconstruction from source code and data
- Emergency model deployment procedures