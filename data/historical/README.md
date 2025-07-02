# Historical F1 Data Directory

This directory contains historical Formula 1 data organized by year and race. Data is primarily in Parquet format for optimal performance with the GPU processing pipeline.

## Data Organization

### By Year Structure
```
historical/
├── 2024/
│   ├── round_01_Bahrain/
│   ├── round_02_Saudi_Arabia/
│   └── ...
├── 2023/
└── 2022/
```

### Race Directory Structure
```
round_XX_EventName/
├── session_info.parquet          # Race weekend metadata
├── lap_times.parquet             # All driver lap times
├── results.parquet               # Final race results
├── telemetry_VER.parquet         # Driver-specific telemetry
├── telemetry_LEC.parquet
└── ...
```

## Data Files Description

### session_info.parquet
Race weekend metadata and session information
```
- year: int64
- round: int64  
- event_name: string
- circuit: string
- date: timestamp
- weather_conditions: string
- track_temperature: float64
- air_temperature: float64
```

### lap_times.parquet
Complete lap timing data for all drivers
```
- driver: string (3-letter code)
- lap_number: int64
- lap_time: timedelta64
- sector_1_time: timedelta64
- sector_2_time: timedelta64  
- sector_3_time: timedelta64
- is_personal_best: bool
- compound: string (tire compound)
- tyre_life: int64
- track_status: string
```

### telemetry_XXX.parquet
High-frequency telemetry data (50Hz) for individual drivers
```
- time: timedelta64 (time in session)
- speed: float64 (km/h)
- throttle: float64 (0-100%)
- brake: float64 (0-100%)
- gear: int64
- drs: bool
- x: float64 (track position)
- y: float64 (track position)
- z: float64 (track position)
```

### results.parquet
Final race results and standings
```
- position: int64
- driver: string
- team: string
- points: int64
- laps: int64
- time: timedelta64
- fastest_lap: timedelta64
- fastest_lap_rank: int64
- status: string
```

## Data Quality

### Validation Checks
- **Completeness:** All expected files present for each race
- **Schema Consistency:** Column types and names match expected format
- **Time Continuity:** No gaps in telemetry time series
- **Value Ranges:** Speed, throttle, brake within expected ranges

### Known Issues
- **Missing Telemetry:** Some sessions may have incomplete telemetry data
- **Time Sync:** Different data sources may have slight timing differences
- **Driver Changes:** Mid-season driver changes affect data consistency

## Usage Examples

### Load Race Weekend Data
```python
import pandas as pd
from pathlib import Path

# Load specific race
race_dir = Path('data/historical/2024/round_01_Bahrain')
lap_times = pd.read_parquet(race_dir / 'lap_times.parquet')
results = pd.read_parquet(race_dir / 'results.parquet')

# Load driver telemetry
verstappen_telemetry = pd.read_parquet(race_dir / 'telemetry_VER.parquet')
```

### Analyze Season Performance
```python
import glob

# Load all lap times for 2024 season
lap_files = glob.glob('data/historical/2024/*/lap_times.parquet')
season_laps = pd.concat([pd.read_parquet(f) for f in lap_files])

# Calculate driver averages
driver_avg = season_laps.groupby('driver')['lap_time'].mean()
```

## Data Sources

- **FastF1:** Primary source for official F1 timing data
- **OpenF1:** Real-time API data for recent sessions
- **FIA:** Official race results and regulations
- **Weather Services:** Track condition data

## Update Schedule

- **Live Sessions:** Real-time during race weekends
- **Post-Session:** Complete data within 30 minutes after session end
- **Historical Backfill:** Older seasons updated as data becomes available
- **Quality Checks:** Weekly validation and cleanup

## File Size Guidelines

### Typical File Sizes
- **session_info.parquet:** 1-5 KB
- **lap_times.parquet:** 50-200 KB per race
- **results.parquet:** 1-5 KB
- **telemetry_XXX.parquet:** 5-50 MB per driver per session

### Storage Estimates
- **Single Race Weekend:** 500 MB - 2 GB
- **Full Season (24 races):** 12-48 GB
- **3 Year Retention:** 36-144 GB

## Performance Tips

### Loading Large Datasets
```python
# Use column selection for large files
columns = ['time', 'speed', 'throttle', 'brake']
df = pd.read_parquet('telemetry_VER.parquet', columns=columns)

# Use time-based filtering
df = pd.read_parquet('telemetry_VER.parquet', 
                    filters=[('time', '>=', pd.Timedelta(minutes=10))])
```

### Memory Optimization
```python
# Use appropriate data types
df['gear'] = df['gear'].astype('int8')
df['drs'] = df['drs'].astype('bool')
df['speed'] = df['speed'].astype('float32')
```