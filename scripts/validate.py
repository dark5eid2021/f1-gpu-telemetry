#!/usr/bin/env python3
"""
F1 GPU Telemetry System - Validation Script
Comprehensive testing and validation of system components
"""

import asyncio
import json
import logging
import os
import sys
import time
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Handle optional imports gracefully
try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

try:
    import psycopg2
    POSTGRES_AVAILABLE = True
except ImportError:
    POSTGRES_AVAILABLE = False


class SystemValidator:
    """Comprehensive system validation for F1 GPU Telemetry"""
    
    def __init__(self):
        self.setup_logging()
        self.logger = logging.getLogger(__name__)
        self.results = {
            'tests_run': 0,
            'tests_passed': 0,
            'tests_failed': 0,
            'warnings': 0,
            'details': []
        }
        
    def setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
    
    def add_result(self, test_name: str, passed: bool, message: str, 
                   warning: bool = False, details: Optional[str] = None):
        """Add test result"""
        self.results['tests_run'] += 1
        
        if passed:
            self.results['tests_passed'] += 1
            status = "✅ PASS"
        elif warning:
            self.results['warnings'] += 1
            status = "⚠️ WARN"
        else:
            self.results['tests_failed'] += 1
            status = "❌ FAIL"
        
        result = {
            'test': test_name,
            'status': status,
            'message': message,
            'details': details
        }
        
        self.results['details'].append(result)
        self.logger.info(f"{status} {test_name}: {message}")
        
        if details:
            self.logger.debug(f"Details: {details}")
    
    def run_command(self, command: str) -> Tuple[bool, str]:
        """Run shell command and return success status and output"""
        try:
            result = subprocess.run(
                command.split(),
                capture_output=True,
                text=True,
                timeout=30
            )
            return result.returncode == 0, result.stdout + result.stderr
        except subprocess.TimeoutExpired:
            return False, "Command timed out"
        except Exception as e:
            return False, str(e)
    
    def test_python_environment(self):
        """Test Python environment and dependencies"""
        self.logger.info("🐍 Testing Python Environment...")
        
        # Python version
        python_version = sys.version_info
        if python_version >= (3, 10):
            self.add_result(
                "Python Version",
                True,
                f"Python {python_version.major}.{python_version.minor}.{python_version.micro}"
            )
        else:
            self.add_result(
                "Python Version",
                False,
                f"Python {python_version.major}.{python_version.minor} (3.10+ required)"
            )
        
        # Required packages
        required_packages = [
            ('pandas', 'Pandas'),
            ('numpy', 'NumPy'),
        ]
        
        # Add optional packages based on availability
        if TORCH_AVAILABLE:
            required_packages.append(('torch', 'PyTorch'))
        if REQUESTS_AVAILABLE:
            required_packages.extend([
                ('fastapi', 'FastAPI'),
                ('uvicorn', 'Uvicorn'),
            ])
        
        for package, name in required_packages:
            try:
                __import__(package)
                self.add_result(f"{name} Import", True, f"{name} available")
            except ImportError:
                self.add_result(f"{name} Import", False, f"{name} not installed")
        
        # Optional packages
        optional_packages = [
            ('cupy', 'CuPy', CUPY_AVAILABLE),
            ('redis', 'Redis', REDIS_AVAILABLE),
            ('psycopg2', 'PostgreSQL', POSTGRES_AVAILABLE),
            ('requests', 'Requests', REQUESTS_AVAILABLE),
        ]
        
        for package, name, available in optional_packages:
            if available:
                self.add_result(f"{name} Import", True, f"{name} available")
            else:
                self.add_result(
                    f"{name} Import", 
                    True, 
                    f"{name} not available (optional)", 
                    warning=True
                )
    
    def test_gpu_availability(self):
        """Test GPU availability and CUDA setup"""
        self.logger.info("🔥 Testing GPU Availability...")
        
        if not TORCH_AVAILABLE:
            self.add_result(
                "PyTorch Availability",
                True,
                "PyTorch not available (GPU tests skipped)",
                warning=True
            )
            return
        
        # CUDA availability
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            self.add_result(
                "CUDA Availability",
                True,
                f"{gpu_count} GPU(s) detected: {gpu_name}"
            )
            
            # GPU memory
            if gpu_count > 0:
                try:
                    memory_allocated = torch.cuda.memory_allocated(0)
                    memory_reserved = torch.cuda.memory_reserved(0)
                    total_memory = torch.cuda.get_device_properties(0).total_memory
                    
                    self.add_result(
                        "GPU Memory",
                        True,
                        f"Total: {total_memory//1024//1024//1024}GB, "
                        f"Reserved: {memory_reserved//1024//1024}MB, "
                        f"Allocated: {memory_allocated//1024//1024}MB"
                    )
                except Exception as e:
                    self.add_result("GPU Memory", False, f"Error checking GPU memory: {e}")
            
            # CuPy availability
            if CUPY_AVAILABLE:
                try:
                    # Test CuPy functionality
                    a = cp.array([1, 2, 3])
                    b = cp.sum(a)
                    self.add_result("CuPy Functionality", True, "CuPy working correctly")
                except Exception as e:
                    self.add_result("CuPy Functionality", False, f"CuPy error: {e}")
            
        else:
            self.add_result(
                "CUDA Availability",
                True,
                "CUDA not available (will run in CPU mode)",
                warning=True
            )
        
        # NVIDIA-SMI
        nvidia_success, nvidia_output = self.run_command("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader")
        if nvidia_success:
            gpu_lines = [line.strip() for line in nvidia_output.strip().split('\n') if line.strip()]
            if gpu_lines:
                gpu_info = gpu_lines[0]
                self.add_result("NVIDIA SMI", True, f"GPU detected: {gpu_info}")
            else:
                self.add_result("NVIDIA SMI", False, "nvidia-smi returned no GPU information")
        else:
            self.add_result(
                "NVIDIA SMI",
                True,
                "nvidia-smi not available (GPU features disabled)",
                warning=True
            )
    
    def test_docker_environment(self):
        """Test Docker availability and configuration"""
        self.logger.info("🐳 Testing Docker Environment...")
        
        # Docker availability
        docker_success, docker_output = self.run_command("docker --version")
        if docker_success:
            version_line = docker_output.split('\n')[0] if docker_output else "Docker installed"
            self.add_result("Docker Availability", True, version_line)
            
            # Docker GPU support (only test if nvidia-smi is available)
            nvidia_available, _ = self.run_command("nvidia-smi")
            if nvidia_available:
                gpu_test_cmd = "docker run --rm --gpus all nvidia/cuda:12.0-base-ubuntu20.04 nvidia-smi"
                gpu_success, gpu_output = self.run_command(gpu_test_cmd)
                
                if gpu_success:
                    self.add_result("Docker GPU Support", True, "NVIDIA Container Runtime working")
                else:
                    self.add_result(
                        "Docker GPU Support",
                        True,
                        "GPU support not configured (optional)",
                        warning=True,
                        details=gpu_output
                    )
            else:
                self.add_result(
                    "Docker GPU Support",
                    True,
                    "No GPU available for Docker testing",
                    warning=True
                )
        else:
            self.add_result("Docker Availability", False, "Docker not available")
    
    def test_configuration(self):
        """Test application configuration"""
        self.logger.info("⚙️ Testing Configuration...")
        
        # .env file existence
        env_file = Path('.env')
        if env_file.exists():
            self.add_result("Configuration File", True, ".env file exists")
            
            # Load and validate environment variables
            try:
                with open('.env', 'r') as f:
                    env_content = f.read()
                
                # Check for placeholder values
                placeholder_patterns = ['your_', 'placeholder', 'change_this', 'example']
                placeholders_found = []
                
                for line in env_content.split('\n'):
                    if '=' in line and not line.strip().startswith('#'):
                        key, value = line.split('=', 1)
                        for pattern in placeholder_patterns:
                            if pattern in value.lower():
                                placeholders_found.append(key.strip())
                
                if placeholders_found:
                    self.add_result(
                        "Configuration Values",
                        False,
                        f"Placeholder values found: {', '.join(placeholders_found)}",
                        details="Please update .env with actual values"
                    )
                else:
                    self.add_result("Configuration Values", True, "No placeholder values detected")
                
            except Exception as e:
                self.add_result("Configuration Parsing", False, f"Error reading .env: {e}")
        else:
            self.add_result("Configuration File", False, ".env file not found")
    
    def test_data_availability(self):
        """Test data directory and sample data"""
        self.logger.info("📊 Testing Data Availability...")
        
        # Data directory structure
        data_dir = Path('data')
        if data_dir.exists():
            self.add_result("Data Directory", True, "Data directory exists")
            
            # Check subdirectories
            subdirs = ['historical', 'models', 'cache']
            for subdir in subdirs:
                subdir_path = data_dir / subdir
                if subdir_path.exists():
                    file_count = len(list(subdir_path.rglob('*')))
                    self.add_result(
                        f"Data/{subdir}",
                        True,
                        f"{subdir} directory exists ({file_count} files)"
                    )
                else:
                    self.add_result(
                        f"Data/{subdir}",
                        True,
                        f"{subdir} directory missing (will be created)",
                        warning=True
                    )
            
            # Check for sample data
            sample_files = list(data_dir.rglob('*sample*'))
            if sample_files:
                self.add_result(
                    "Sample Data",
                    True,
                    f"Found {len(sample_files)} sample data files"
                )
            else:
                self.add_result(
                    "Sample Data",
                    True,
                    "No sample data found (run download script)",
                    warning=True
                )
        else:
            self.add_result("Data Directory", False, "Data directory missing")
    
    def test_network_connectivity(self):
        """Test network connectivity and external APIs"""
        self.logger.info("🌐 Testing Network Connectivity...")
        
        if not REQUESTS_AVAILABLE:
            self.add_result(
                "Network Tests",
                True,
                "requests library not available (network tests skipped)",
                warning=True
            )
            return
        
        # Test OpenF1 API
        try:
            response = requests.get(
                "https://api.openf1.org/v1/sessions",
                timeout=10
            )
            if response.status_code == 200:
                data = response.json()
                self.add_result(
                    "OpenF1 API",
                    True,
                    f"API accessible ({len(data)} sessions available)"
                )
            else:
                self.add_result(
                    "OpenF1 API",
                    False,
                    f"API returned status {response.status_code}"
                )
        except requests.RequestException as e:
            self.add_result("OpenF1 API", False, f"Connection failed: {e}")
        
        # Test local services (if running)
        local_services = [
            ("http://localhost:8000/api/v1/health", "F1 API Server"),
            ("http://localhost:3000", "Frontend"),
        ]
        
        for url, service_name in local_services:
            try:
                response = requests.get(url, timeout=2)
                self.add_result(
                    f"{service_name} Connectivity",
                    True,
                    f"{service_name} accessible"
                )
            except requests.RequestException:
                self.add_result(
                    f"{service_name} Connectivity",
                    True,
                    f"{service_name} not running (expected if not started)",
                    warning=True
                )
    
    def test_api_functionality(self):
        """Test API functionality if server is running"""
        self.logger.info("🌐 Testing API Functionality...")
        
        if not REQUESTS_AVAILABLE:
            self.add_result(
                "API Tests",
                True,
                "requests library not available (API tests skipped)",
                warning=True
            )
            return
        
        base_url = "http://localhost:8000"
        
        try:
            # Health check
            response = requests.get(f"{base_url}/api/v1/health", timeout=5)
            if response.status_code == 200:
                health_data = response.json()
                self.add_result(
                    "API Health Check",
                    True,
                    f"API healthy (status: {health_data.get('status', 'unknown')})"
                )
                
                # Test additional endpoints
                endpoints = [
                    "/api/v1/status",
                    "/api/v1/telemetry/live",
                    "/api/v1/predictions/race"
                ]
                
                for endpoint in endpoints:
                    try:
                        resp = requests.get(f"{base_url}{endpoint}", timeout=3)
                        self.add_result(
                            f"API {endpoint}",
                            resp.status_code < 500,
                            f"Status: {resp.status_code}"
                        )
                    except requests.RequestException as e:
                        self.add_result(f"API {endpoint}", False, f"Error: {e}")
            else:
                self.add_result(
                    "API Health Check",
                    False,
                    f"API returned status {response.status_code}"
                )
        
        except requests.RequestException:
            self.add_result(
                "API Health Check",
                True,
                "API server not running (expected if not started)",
                warning=True
            )
    
    async def test_database_connectivity(self):
        """Test database connectivity if available"""
        self.logger.info("🗄️ Testing Database Connectivity...")
        
        if not POSTGRES_AVAILABLE:
            self.add_result(
                "PostgreSQL Client",
                True,
                "PostgreSQL client not available (optional)",
                warning=True
            )
            return
        
        # Try to connect to local PostgreSQL
        try:
            conn = psycopg2.connect(
                host="localhost",
                port=5432,
                database="f1_telemetry",
                user="f1user",
                password="password",  # Default test password
                connect_timeout=3
            )
            conn.close()
            self.add_result("PostgreSQL Connection", True, "Database accessible")
            
        except psycopg2.Error as e:
            self.add_result(
                "PostgreSQL Connection",
                True,
                f"Database not accessible (expected if not started): {e}",
                warning=True
            )
    
    async def test_redis_connectivity(self):
        """Test Redis connectivity if available"""
        self.logger.info("📦 Testing Redis Connectivity...")
        
        if not REDIS_AVAILABLE:
            self.add_result(
                "Redis Client",
                True,
                "Redis client not available (optional)",
                warning=True
            )
            return
        
        try:
            r = redis.Redis(host='localhost', port=6379, db=0, socket_timeout=3)
            r.ping()
            self.add_result("Redis Connection", True, "Redis accessible")
            
            # Test Redis operations in a separate try block
            try:
                test_key = "f1_test_key"
                test_value = "test_value"
                r.set(test_key, test_value, ex=10)  # 10 second expiry
                retrieved = r.get(test_key)
                
                if retrieved and retrieved.decode('utf-8') == test_value:
                    self.add_result("Redis Operations", True, "Redis read/write working")
                else:
                    self.add_result("Redis Operations", False, "Redis read/write failed")
                
                r.delete(test_key)
                
            except redis.RedisError as e:
                self.add_result("Redis Operations", False, f"Redis operations failed: {e}")
            
        except redis.RedisError as e:
            self.add_result(
                "Redis Connection",
                True,
                f"Redis not accessible (expected if not started): {e}",
                warning=True
            )
    
    def test_model_loading(self):
        """Test model loading capabilities"""
        self.logger.info("🧠 Testing Model Loading...")
        
        if not TORCH_AVAILABLE:
            self.add_result(
                "PyTorch Models",
                True,
                "PyTorch not available (model tests skipped)",
                warning=True
            )
            return
        
        # Test PyTorch model creation
        try:
            # Simple test model
            model = torch.nn.Linear(10, 1)
            test_input = torch.randn(1, 10)
            output = model(test_input)
            
            self.add_result("PyTorch Model Creation", True, "PyTorch models working")
            
            # Test GPU model if available
            if torch.cuda.is_available():
                try:
                    model_gpu = model.cuda()
                    test_input_gpu = test_input.cuda()
                    output_gpu = model_gpu(test_input_gpu)
                    self.add_result("GPU Model Execution", True, "GPU models working")
                except Exception as e:
                    self.add_result("GPU Model Execution", False, f"GPU model error: {e}")
            
        except Exception as e:
            self.add_result("PyTorch Model Creation", False, f"Model creation failed: {e}")
        
        # Check for pre-trained models
        models_dir = Path('data/models')
        if models_dir.exists():
            model_files = list(models_dir.glob('*.pth')) + list(models_dir.glob('*.pt'))
            if model_files:
                self.add_result(
                    "Pre-trained Models",
                    True,
                    f"Found {len(model_files)} model files"
                )
                
                # Try loading one model
                try:
                    model_file = model_files[0]
                    torch.load(model_file, map_location='cpu')
                    self.add_result("Model Loading", True, f"Successfully loaded {model_file.name}")
                except Exception as e:
                    self.add_result("Model Loading", False, f"Failed to load {model_file.name}: {e}")
            else:
                self.add_result(
                    "Pre-trained Models",
                    True,
                    "No pre-trained models found (expected for new installation)",
                    warning=True
                )
        else:
            self.add_result(
                "Pre-trained Models",
                True,
                "Models directory not found (expected for new installation)",
                warning=True
            )
    
    def test_scripts_availability(self):
        """Test availability and executability of scripts"""
        self.logger.info("📜 Testing Scripts Availability...")
        
        scripts_to_test = [
            'scripts/start-local.sh',
            'scripts/deploy-k8s.sh',
            'scripts/check-system.sh',
            'scripts/update-data.sh',
            'scripts/download_historical_data.py'
        ]
        
        for script in scripts_to_test:
            script_path = Path(script)
            if script_path.exists():
                # Check if executable
                if os.access(script_path, os.X_OK):
                    self.add_result(f"Script {script_path.name}", True, "Available and executable")
                else:
                    self.add_result(
                        f"Script {script_path.name}",
                        True,
                        "Available but not executable (run: chmod +x)",
                        warning=True
                    )
            else:
                self.add_result(f"Script {script_path.name}", False, "Script not found")
    
    def test_performance_benchmarks(self):
        """Run basic performance benchmarks"""
        self.logger.info("⚡ Testing Performance Benchmarks...")
        
        # CPU performance test
        start_time = time.time()
        
        # Simple CPU benchmark
        result = sum(i * i for i in range(100000))
        cpu_time = time.time() - start_time
        
        self.add_result(
            "CPU Performance",
            True,
            f"CPU benchmark completed in {cpu_time:.3f}s"
        )
        
        # GPU performance test if available
        if TORCH_AVAILABLE and torch.cuda.is_available():
            try:
                torch.cuda.synchronize()
                start_time = time.time()
                
                # Simple GPU benchmark
                a = torch.randn(1000, 1000, device='cuda')
                b = torch.randn(1000, 1000, device='cuda')
                c = torch.matmul(a, b)
                
                torch.cuda.synchronize()
                gpu_time = time.time() - start_time
                
                speedup = cpu_time / gpu_time if gpu_time > 0 else 0
                self.add_result(
                    "GPU Performance",
                    True,
                    f"GPU benchmark completed in {gpu_time:.3f}s (speedup: {speedup:.1f}x)"
                )
                
            except Exception as e:
                self.add_result("GPU Performance", False, f"GPU benchmark failed: {e}")
        
        # Memory usage test
        try:
            import psutil
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024
            
            if memory_mb < 1000:  # Less than 1GB
                self.add_result("Memory Usage", True, f"Current usage: {memory_mb:.1f}MB")
            else:
                self.add_result(
                    "Memory Usage",
                    True,
                    f"High memory usage: {memory_mb:.1f}MB",
                    warning=True
                )
                
        except ImportError:
            self.add_result(
                "Memory Usage",
                True,
                "psutil not available for memory monitoring",
                warning=True
            )
    
    def test_kubernetes_availability(self):
        """Test Kubernetes availability"""
        self.logger.info("☸️ Testing Kubernetes Availability...")
        
        # kubectl availability
        kubectl_success, kubectl_output = self.run_command("kubectl version --client")
        if kubectl_success:
            self.add_result("kubectl Client", True, "kubectl available")
            
            # Cluster connectivity
            cluster_success, cluster_output = self.run_command("kubectl cluster-info")
            if cluster_success:
                self.add_result("Kubernetes Cluster", True, "Connected to cluster")
                
                # Check for GPU nodes
                gpu_nodes_success, gpu_nodes_output = self.run_command(
                    "kubectl get nodes -l nvidia.com/gpu.present=true"
                )
                if gpu_nodes_success and "nvidia.com/gpu.present=true" in gpu_nodes_output:
                    node_lines = [line for line in gpu_nodes_output.split('\n') 
                                 if line and not line.startswith('NAME')]
                    node_count = len(node_lines)
                    self.add_result("GPU Nodes", True, f"{node_count} GPU nodes available")
                else:
                    self.add_result(
                        "GPU Nodes",
                        True,
                        "No GPU nodes found (CPU-only cluster)",
                        warning=True
                    )
            else:
                self.add_result(
                    "Kubernetes Cluster",
                    True,
                    "Not connected to cluster (optional)",
                    warning=True
                )
        else:
            self.add_result(
                "kubectl Client",
                True,
                "kubectl not available (optional)",
                warning=True
            )
    
    async def run_all_tests(self):
        """Run all validation tests"""
        self.logger.info("🚀 Starting F1 GPU Telemetry System Validation...")
        self.logger.info("=" * 60)
        
        # Run all tests
        self.test_python_environment()
        self.test_gpu_availability()
        self.test_docker_environment()
        self.test_kubernetes_availability()
        self.test_configuration()
        self.test_data_availability()
        self.test_scripts_availability()
        self.test_network_connectivity()
        self.test_api_functionality()
        await self.test_database_connectivity()
        await self.test_redis_connectivity()
        self.test_model_loading()
        self.test_performance_benchmarks()
        
        # Generate report
        return self.generate_report()
    
    def generate_report(self):
        """Generate validation report"""
        self.logger.info("=" * 60)
        
        total_tests = self.results['tests_run']
        passed = self.results['tests_passed']
        failed = self.results['tests_failed']
        warnings = self.results['warnings']
        
        success_rate = (passed / total_tests * 100) if total_tests > 0 else 0
        
        print(f"\n📈 Summary:")
        print(f"  Total Tests: {total_tests}")
        print(f"  Passed: {passed} ({success_rate:.1f}%)")
        print(f"  Failed: {failed}")
        print(f"  Warnings: {warnings}")
        
        if failed == 0:
            print(f"\n🎉 All tests passed! System ready for deployment.")
            system_status = "READY"
        elif failed <= 2:
            print(f"\n⚠️ Minor issues detected. System should work with limitations.")
            system_status = "READY_WITH_WARNINGS"
        else:
            print(f"\n❌ Major issues detected. Please resolve before deployment.")
            system_status = "NOT_READY"
        
        print(f"\n🔍 Detailed Results:")
        for result in self.results['details']:
            print(f"  {result['status']} {result['test']}: {result['message']}")
            if result['details']:
                print(f"    Details: {result['details']}")
        
        # Save report to file
        report_file = Path('validation_report.json')
        report_data = {
            'timestamp': time.time(),
            'system_status': system_status,
            'summary': {
                'total_tests': total_tests,
                'passed': passed,
                'failed': failed,
                'warnings': warnings,
                'success_rate': success_rate
            },
            'details': self.results['details']
        }
        
        try:
            with open(report_file, 'w') as f:
                json.dump(report_data, f, indent=2)
            print(f"\n📄 Full report saved to: {report_file}")
        except Exception as e:
            print(f"\n⚠️ Could not save report file: {e}")
        
        # Recommendations
        self.generate_recommendations()
        
        return system_status == "READY" or system_status == "READY_WITH_WARNINGS"
    
    def generate_recommendations(self):
        """Generate recommendations based on test results"""
        print(f"\n💡 Recommendations:")
        
        failed_tests = [r for r in self.results['details'] if '❌' in r['status']]
        warning_tests = [r for r in self.results['details'] if '⚠️' in r['status']]
        
        if failed_tests:
            print("  Critical Issues to Resolve:")
            for test in failed_tests:
                print(f"    - {test['test']}: {test['message']}")
        
        if warning_tests:
            print("  Optional Improvements:")
            for test in warning_tests:
                print(f"    - {test['test']}: {test['message']}")
        
        # Specific recommendations
        gpu_available = any('GPU' in r['test'] and '✅' in r['status'] for r in self.results['details'])
        config_ready = any('Configuration' in r['test'] and '✅' in r['status'] for r in self.results['details'])
        
        print("\n🎯 Next Steps:")
        
        if not config_ready:
            print("  1. Set up configuration: cp .env.example .env && edit .env")
        
        if not gpu_available:
            print("  2. For GPU acceleration: Install NVIDIA drivers and CUDA")
            print("     For CPU-only mode: Use --cpu-only flag when starting")
        
        if not any('Sample Data' in r['test'] and '✅' in r['status'] for r in self.results['details']):
            print("  3. Download sample data: python scripts/download_historical_data.py --sample")
        
        print("  4. Start the system:")
        print("     - Docker: ./scripts/start.sh --mode docker")
        print("     - CPU-only: ./scripts/start.sh --mode docker --cpu-only")
        print("     - Local dev: ./scripts/start.sh --mode local")
        print("  5. Access dashboard: http://localhost:3000")


def main():
    """Main validation function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='F1 GPU Telemetry System Validator')
    parser.add_argument('--quick', action='store_true', help='Run quick validation only')
    parser.add_argument('--no-network', action='store_true', help='Skip network tests')
    parser.add_argument('--cpu-only', action='store_true', help='Skip GPU-specific tests')
    parser.add_argument('--output', help='Output report file')
    parser.add_argument('--quiet', action='store_true', help='Reduce output verbosity')
    
    args = parser.parse_args()
    
    # Set logging level based on quiet flag
    if args.quiet:
        logging.getLogger().setLevel(logging.WARNING)
    
    validator = SystemValidator()
    
    try:
        # Run validation
        result = asyncio.run(validator.run_all_tests())
        
        # Exit with appropriate code
        sys.exit(0 if result else 1)
        
    except KeyboardInterrupt:
        print("\n\n⚠️ Validation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Validation failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()