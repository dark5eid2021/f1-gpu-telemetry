# F1 GPU Telemetry System - validate.py Documentation

The `validate.py` script is a comprehensive system validation tool that checks all components of the F1 GPU Telemetry System to ensure proper installation, configuration, and functionality.

## 📋 Overview

The validation script performs automated testing of your system environment, dependencies, configuration, and optional components to verify readiness for running the F1 GPU Telemetry System. It provides detailed reports with actionable recommendations for resolving any issues.

## 🚀 Quick Start

### Basic Usage
```bash
# Run full validation
python scripts/validate.py

# CPU-only validation (skip GPU tests)
python scripts/validate.py --cpu-only

# Quick validation (essential tests only)
python scripts/validate.py --quick

# Quiet mode (minimal output)
python scripts/validate.py --quiet
```

### Installation
Place the script in your `scripts/` directory:
```bash
# Make executable
chmod +x scripts/validate.py

# Run from project root
python scripts/validate.py
```

## 🔧 Command Line Options

| Option | Description | Use Case |
|--------|-------------|----------|
| `--quick` | Run essential tests only | Fast environment check |
| `--cpu-only` | Skip GPU-specific tests | Systems without NVIDIA GPU |
| `--no-network` | Skip network connectivity tests | Offline environments |
| `--quiet` | Reduce output verbosity | Automated scripts |
| `--output FILE` | Save report to custom file | CI/CD integration |
| `--help` | Show help message | Reference |

### Examples
```bash
# Validate CPU-only setup for development
python scripts/validate.py --cpu-only --quiet

# Quick check before deployment
python scripts/validate.py --quick

# Offline validation
python scripts/validate.py --no-network

# CI/CD integration
python scripts/validate.py --quiet --output ci_report.json
```

## 🧪 Test Categories

### 1. Python Environment Tests
**Purpose**: Verify Python version and core dependencies

**Tests Performed**:
- ✅ Python version (3.10+ required)
- ✅ Core package imports (pandas, numpy)
- ✅ Optional package availability (torch, fastapi, uvicorn)
- ✅ GPU packages (cupy, rapids) if available

**Sample Output**:
```
✅ PASS Python Version: Python 3.10.12
✅ PASS Pandas Import: Pandas available
✅ PASS PyTorch Import: PyTorch available
⚠️ WARN CuPy Import: CuPy not available (optional)
```

### 2. GPU Availability Tests
**Purpose**: Check GPU hardware and CUDA setup

**Tests Performed**:
- ✅ CUDA availability via PyTorch
- ✅ GPU memory information
- ✅ CuPy functionality testing
- ✅ NVIDIA-SMI command availability
- ✅ GPU device detection

**Sample Output**:
```
✅ PASS CUDA Availability: 1 GPU(s) detected: NVIDIA RTX 4090
✅ PASS GPU Memory: Total: 24GB, Reserved: 0MB, Allocated: 0MB
✅ PASS CuPy Functionality: CuPy working correctly
✅ PASS NVIDIA SMI: GPU detected: NVIDIA RTX 4090, 24564 MiB
```

**CPU-Only Mode**:
```
⚠️ WARN CUDA Availability: CUDA not available (will run in CPU mode)
⚠️ WARN NVIDIA SMI: nvidia-smi not available (GPU features disabled)
```

### 3. Docker Environment Tests
**Purpose**: Validate Docker installation and GPU support

**Tests Performed**:
- ✅ Docker installation and version
- ✅ NVIDIA Container Runtime availability
- ✅ GPU passthrough functionality
- ✅ Container execution capability

**Sample Output**:
```
✅ PASS Docker Availability: Docker version 24.0.6, build ed223bc
✅ PASS Docker GPU Support: NVIDIA Container Runtime working
```

### 4. Configuration Tests
**Purpose**: Verify application configuration files

**Tests Performed**:
- ✅ `.env` file existence
- ✅ Placeholder value detection
- ✅ Required configuration validation
- ✅ Configuration parsing

**Sample Output**:
```
✅ PASS Configuration File: .env file exists
❌ FAIL Configuration Values: Placeholder values found: OPENF1_API_KEY, JWT_SECRET_KEY
    Details: Please update .env with actual values
```

### 5. Data Availability Tests
**Purpose**: Check data directories and sample data

**Tests Performed**:
- ✅ Data directory structure
- ✅ Subdirectory existence (historical, models, cache)
- ✅ Sample data file detection
- ✅ File count reporting

**Sample Output**:
```
✅ PASS Data Directory: Data directory exists
✅ PASS Data/historical: historical directory exists (156 files)
⚠️ WARN Data/models: models directory missing (will be created)
⚠️ WARN Sample Data: No sample data found (run download script)
```

### 6. Network Connectivity Tests
**Purpose**: Test external API access and local services

**Tests Performed**:
- ✅ OpenF1 API accessibility
- ✅ Local service detection (API server, frontend)
- ✅ Network timeout handling
- ✅ HTTP response validation

**Sample Output**:
```
✅ PASS OpenF1 API: API accessible (247 sessions available)
⚠️ WARN F1 API Server Connectivity: F1 API Server not running (expected if not started)
⚠️ WARN Frontend Connectivity: Frontend not running (expected if not started)
```

### 7. Database & Cache Tests
**Purpose**: Verify database and Redis connectivity

**Tests Performed**:
- ✅ PostgreSQL connection testing
- ✅ Redis connection and operations
- ✅ Database client availability
- ✅ Connection timeout handling

**Sample Output**:
```
✅ PASS PostgreSQL Connection: Database accessible
✅ PASS Redis Connection: Redis accessible
✅ PASS Redis Operations: Redis read/write working
```

### 8. Model Loading Tests
**Purpose**: Test ML model functionality

**Tests Performed**:
- ✅ PyTorch model creation
- ✅ GPU model execution (if available)
- ✅ Pre-trained model detection
- ✅ Model file loading

**Sample Output**:
```
✅ PASS PyTorch Model Creation: PyTorch models working
✅ PASS GPU Model Execution: GPU models working
⚠️ WARN Pre-trained Models: No pre-trained models found (expected for new installation)
```

### 9. Scripts Availability Tests
**Purpose**: Check for required scripts and permissions

**Tests Performed**:
- ✅ Script file existence
- ✅ Execute permissions
- ✅ Critical script availability

**Sample Output**:
```
✅ PASS Script start-local.sh: Available and executable
⚠️ WARN Script deploy-k8s.sh: Available but not executable (run: chmod +x)
❌ FAIL Script missing-script.sh: Script not found
```

### 10. Performance Benchmarks
**Purpose**: Basic performance testing

**Tests Performed**:
- ✅ CPU benchmark timing
- ✅ GPU benchmark (if available)
- ✅ Memory usage monitoring
- ✅ Performance comparison

**Sample Output**:
```
✅ PASS CPU Performance: CPU benchmark completed in 0.152s
✅ PASS GPU Performance: GPU benchmark completed in 0.008s (speedup: 19.0x)
✅ PASS Memory Usage: Current usage: 342.1MB
```

### 11. Kubernetes Tests
**Purpose**: Validate Kubernetes environment (optional)

**Tests Performed**:
- ✅ kubectl client availability
- ✅ Cluster connectivity
- ✅ GPU node detection
- ✅ Namespace accessibility

**Sample Output**:
```
✅ PASS kubectl Client: kubectl available
✅ PASS Kubernetes Cluster: Connected to cluster
✅ PASS GPU Nodes: 2 GPU nodes available
```

## 📊 Report Generation

### Console Output
The script provides real-time feedback with colored status indicators:
- ✅ **PASS** - Test succeeded
- ❌ **FAIL** - Test failed (action required)
- ⚠️ **WARN** - Warning (optional feature)

### JSON Report
Detailed results are saved to `validation_report.json`:

```json
{
  "timestamp": 1704123456.789,
  "system_status": "READY_WITH_WARNINGS",
  "summary": {
    "total_tests": 45,
    "passed": 38,
    "failed": 2,
    "warnings": 5,
    "success_rate": 84.4
  },
  "details": [
    {
      "test": "Python Version",
      "status": "✅ PASS",
      "message": "Python 3.10.12",
      "details": null
    },
    {
      "test": "Configuration Values",
      "status": "❌ FAIL",
      "message": "Placeholder values found: OPENF1_API_KEY",
      "details": "Please update .env with actual values"
    }
  ]
}
```

### System Status Levels
- **READY** - All tests passed, system ready for deployment
- **READY_WITH_WARNINGS** - Minor issues, system functional with limitations
- **NOT_READY** - Critical issues, resolve before deployment

## 🛠️ Troubleshooting

### Common Issues and Solutions

#### 1. Python Dependencies Missing
**Error**: `❌ FAIL PyTorch Import: PyTorch not installed`

**Solution**:
```bash
# Install missing dependencies
pip install torch torchvision
# Or for CPU-only
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

#### 2. Configuration Problems
**Error**: `❌ FAIL Configuration Values: Placeholder values found`

**Solution**:
```bash
# Copy and edit configuration
cp .env.example .env
# Edit .env with actual values
vim .env
```

#### 3. GPU Not Available
**Warning**: `⚠️ WARN CUDA Availability: CUDA not available`

**Solutions**:
```bash
# Option 1: Install NVIDIA drivers and CUDA
# Option 2: Use CPU-only mode
python scripts/validate.py --cpu-only
```

#### 4. Docker GPU Support Missing
**Warning**: `⚠️ WARN Docker GPU Support: GPU support not configured`

**Solution**:
```bash
# Install NVIDIA Container Runtime
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update && sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker
```

#### 5. Network Connectivity Issues
**Error**: `❌ FAIL OpenF1 API: Connection failed`

**Solutions**:
```bash
# Check internet connection
curl -I https://api.openf1.org/v1/sessions

# Skip network tests if offline
python scripts/validate.py --no-network
```

## 🔄 Integration with Other Tools

### CI/CD Integration
```yaml
# GitHub Actions example
- name: Validate F1 System
  run: |
    python scripts/validate.py --quiet --cpu-only
    if [ $? -ne 0 ]; then
      echo "Validation failed"
      exit 1
    fi
```

### Pre-deployment Check
```bash
#!/bin/bash
# pre-deploy.sh
echo "Running pre-deployment validation..."
python scripts/validate.py --quick

if [ $? -eq 0 ]; then
    echo "✅ System validated - proceeding with deployment"
    ./scripts/start.sh --mode docker
else
    echo "❌ Validation failed - please resolve issues"
    exit 1
fi
```

### Development Workflow
```bash
# Morning routine for developers
python scripts/validate.py --cpu-only --quiet && echo "Ready to code! 🚀"
```

## 📈 Performance and Optimization

### Execution Time
- **Full validation**: ~30-60 seconds
- **Quick validation**: ~10-20 seconds
- **CPU-only**: ~15-30 seconds

### Memory Usage
- **Peak memory**: ~50-100MB
- **GPU memory**: Minimal test allocation only
- **Cleanup**: Automatic resource cleanup

### Network Usage
- **OpenF1 API**: Single lightweight request
- **Local services**: Quick health checks only
- **Offline mode**: Available with `--no-network`

## 🔒 Security Considerations

### Safe Operation
- ✅ **No data modification** - Read-only operations
- ✅ **Minimal permissions** - Standard user access only
- ✅ **Temporary resources** - Auto-cleanup of test data
- ✅ **No credential exposure** - Configuration validation only

### What Gets Tested
- ✅ Configuration file existence (not contents)
- ✅ API connectivity (not authentication)
- ✅ Service availability (not data access)
- ✅ System capabilities (not sensitive information)

## 📚 Advanced Usage

### Custom Test Selection
```python
# Custom validation script
from scripts.validate import SystemValidator

validator = SystemValidator()

# Run specific tests only
validator.test_python_environment()
validator.test_gpu_availability()
validator.generate_report()
```

### Integration with Monitoring
```python
# Health check endpoint integration
import json
from pathlib import Path

def get_system_health():
    report_file = Path('validation_report.json')
    if report_file.exists():
        with open(report_file) as f:
            return json.load(f)
    return {"status": "unknown"}
```

## 🎯 Best Practices

### When to Run Validation
- ✅ **Before first setup** - Verify system compatibility
- ✅ **After environment changes** - Confirm functionality
- ✅ **Before deployment** - Ensure readiness
- ✅ **During troubleshooting** - Identify issues
- ✅ **In CI/CD pipelines** - Automated verification

### Recommended Flags by Environment
- **Development**: `--cpu-only` (if no GPU)
- **CI/CD**: `--quiet --no-network`
- **Production**: Full validation (no flags)
- **Troubleshooting**: Default (full output)
- **Quick check**: `--quick`

This validation script is your first line of defense against configuration issues and environmental problems. Run it early and often to ensure your F1 GPU Telemetry System is always ready to perform at its best! 🏎️