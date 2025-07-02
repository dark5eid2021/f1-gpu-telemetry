# F1 GPU Telemetry System - Data Directory

This directory contains all data files for the F1 GPU Telemetry System. The structure is organized for optimal performance and easy maintenance.

## Directory Structure

```
data/
├── README.md                    # This file
├── .gitkeep                     # Ensures directory is tracked in git
├── historical/                  # Historical F1 data
│   ├── README.md
│   ├── 2024/                   # Data by year
│   ├── 2023/
│   └── metadata.json          # Data catalog metadata
├── cache/                      # FastF1 and processing cache
│   ├── README.md
│   └── .gitkeep
├── models/                     # Trained ML models
│   ├── README.md
│   ├── checkpoints/           # Model checkpoints
│   ├── production/            # Production models
│   └── experiments/           # Experimental models
├── real-time/                 # Real-time data buffer
│   ├── README.md
│   └── .gitkeep
└── backup/                    # Data backups
    ├── README.md
    └── .gitkeep
```

## Data Types

### Historical Data
- **F1 Telemetry:** Speed, throttle, brake, tire temperatures
- **Lap Times:** Sector times, lap records, race results
- **Weather Data:** Track conditions, temperature, humidity
- **Session Data:** Practice, qualifying, race sessions

### Real-time Data
- **Live Telemetry:** Streaming car data during races
- **Position Data:** Real-time car positions and gaps
- **Prediction Results:** ML model outputs and confidence scores

### Model Data
- **Trained Models:** PyTorch and scikit-learn models
- **Model Metadata:** Training metrics, hyperparameters
- **Checkpoints:** Training snapshots for recovery

## Storage Guidelines

### File Formats
- **Parquet:** Primary format for structured data (efficient, columnar)
- **JSON:** Configuration and metadata files
- **HDF5:** Large numerical datasets
- **CSV:** Human-readable exports and imports

### Naming Conventions
```
# Historical data
YYYY_RR_EventName_DataType.parquet
# Example: 2024_01_Bahrain_telemetry.parquet

# Real-time data
YYYYMMDD_HHMMSS_session_key_DataType.parquet
# Example: 20241201_143000_9158_telemetry.parquet

# Models
ModelType_YYYYMMDD_HHMMSS_version.pth
# Example: RacePredictor_20241201_143000_v1.2.pth
```

## Data Retention Policy

### Historical Data
- **Keep:** Current year + 2 previous years
- **Archive:** Data older than 3 years to cold storage
- **Delete:** Data older than 5 years (unless historically significant)

### Real-time Data
- **Buffer:** Keep 24 hours in real-time directory
- **Process:** Move to historical after processing
- **Cleanup:** Daily cleanup of processed files

### Cache Data
- **FastF1 Cache:** Keep 30 days, auto-cleanup
- **Processing Cache:** Keep 7 days, manual cleanup
- **Temporary Files:** Delete after 24 hours

## Usage Examples

### Loading Historical Data
```python
import pandas as pd

# Load specific race telemetry
df = pd.read_parquet('data/historical/2024/2024_01_Bahrain_telemetry.parquet')

# Load multiple races
import glob
files = glob.glob('data/historical/2024/*_telemetry.parquet')
df_all = pd.concat([pd.read_parquet(f) for f in files])
```

### Working with Models
```python
import torch

# Load trained model
model = torch.load('data/models/production/RacePredictor_latest.pth')

# Save model checkpoint
torch.save(model.state_dict(), 'data/models/checkpoints/checkpoint_epoch_100.pth')
```

## Security Notes

- **No Credentials:** Never store API keys or passwords in data files
- **Anonymization:** Personal data should be anonymized where applicable
- **Encryption:** Sensitive data should be encrypted at rest
- **Access Control:** Implement proper file permissions

## Monitoring

- **Disk Usage:** Monitor directory sizes for storage planning
- **File Counts:** Track number of files for performance
- **Access Patterns:** Monitor which data is accessed frequently
- **Data Quality:** Regular validation of data integrity

## Backup Strategy

- **Daily Backups:** Critical model and configuration data
- **Weekly Backups:** Historical race data
- **Monthly Archives:** Complete system snapshots
- **Offsite Storage:** Cloud backup for disaster recovery

## Data Sources

- **OpenF1 API:** Real-time and historical F1 data
- **FastF1:** Official F1 timing and telemetry data
- **Weather APIs:** Track condition data
- **Internal Systems:** Processed and enriched data

For more information about data processing pipelines, see the main project documentation.