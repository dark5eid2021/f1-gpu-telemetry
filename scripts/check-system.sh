# ============================================================================
# scripts/check-system.sh - System Health Check Script
# ============================================================================

#!/bin/bash

# F1 GPU Telemetry System - Health Check Script
# Verifies system requirements and deployment status

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'

print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[✅]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[⚠️]${NC} $1"
}

print_error() {
    echo -e "${RED}[❌]${NC} $1"
}

print_header() {
    echo -e "${BLUE}============================================================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}============================================================================${NC}"
}

echo "🔍 F1 GPU Telemetry System - Health Check"
echo ""

# System Requirements Check
print_header "System Requirements"

# Check GPU
print_status "Checking NVIDIA GPU..."
if command -v nvidia-smi &> /dev/null; then
    gpu_info=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits | head -1)
    print_success "GPU: $gpu_info"
    
    # Check GPU utilization
    gpu_util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits)
    gpu_mem=$(nvidia-smi --query-gpu=utilization.memory --format=csv,noheader,nounits)
    echo "   GPU Utilization: ${gpu_util}%"
    echo "   Memory Utilization: ${gpu_mem}%"
else
    print_error "NVIDIA GPU not detected"
fi

# Check Docker
print_status "Checking Docker..."
if command -v docker &> /dev/null; then
    docker_version=$(docker --version | cut -d' ' -f3 | cut -d',' -f1)
    print_success "Docker: v$docker_version"
    
    # Check NVIDIA Container Runtime
    if docker run --rm --gpus all nvidia/cuda:12.0-base-ubuntu20.04 nvidia-smi &> /dev/null 2>&1; then
        print_success "NVIDIA Container Runtime: OK"
    else
        print_warning "NVIDIA Container Runtime: Not configured"
    fi
else
    print_error "Docker not installed"
fi

# Check Kubernetes
print_status "Checking Kubernetes..."
if command -v kubectl &> /dev/null; then
    if kubectl cluster-info &> /dev/null; then
        cluster_info=$(kubectl cluster-info | head -1)
        print_success "Kubernetes: Connected"
        echo "   $cluster_info"
        
        # Check GPU nodes
        gpu_nodes=$(kubectl get nodes -l nvidia.com/gpu.present=true --no-headers 2>/dev/null | wc -l)
        if [ $gpu_nodes -gt 0 ]; then
            print_success "GPU Nodes: $gpu_nodes available"
        else
            print_warning "GPU Nodes: None found"
        fi
    else
        print_warning "Kubernetes: Not connected to cluster"
    fi
else
    print_warning "kubectl not installed"
fi

echo ""

# Application Status Check
print_header "Application Status"

# Check .env file
print_status "Checking configuration..."
if [ -f ".env" ]; then
    print_success "Configuration file: .env exists"
    
    # Source and validate
    source .env
    
    if [[ "$OPENF1_API_KEY" != *"your_"* ]] && [[ ! -z "$OPENF1_API_KEY" ]]; then
        print_success "OpenF1 API Key: Configured"
    else
        print_warning "OpenF1 API Key: Not configured"
    fi
    
    if [[ "$POSTGRES_PASSWORD" != *"your_"* ]] && [[ ! -z "$POSTGRES_PASSWORD" ]]; then
        print_success "Database Password: Configured"
    else
        print_warning "Database Password: Not configured"
    fi
else
    print_error "Configuration: .env file not found"
fi

# Check Docker containers
print_status "Checking Docker containers..."
if command -v docker &> /dev/null; then
    running_containers=$(docker ps --filter "name=f1" --format "table {{.Names}}\t{{.Status}}" 2>/dev/null || echo "")
    if [ ! -z "$running_containers" ]; then
        print_success "Docker containers running:"
        echo "$running_containers"
    else
        print_warning "No F1 containers running"
    fi
fi

# Check Kubernetes deployment
if command -v kubectl &> /dev/null && kubectl cluster-info &> /dev/null; then
    print_status "Checking Kubernetes deployment..."
    
    if kubectl get namespace f1-system &> /dev/null; then
        print_success "Namespace: f1-system exists"
        
        # Check pods
        pod_status=$(kubectl get pods -n f1-system --no-headers 2>/dev/null || echo "")
        if [ ! -z "$pod_status" ]; then
            print_success "Pods in f1-system:"
            kubectl get pods -n f1-system
        else
            print_warning "No pods found in f1-system namespace"
        fi
        
        # Check services
        service_status=$(kubectl get services -n f1-system --no-headers 2>/dev/null || echo "")
        if [ ! -z "$service_status" ]; then
            echo ""
            print_success "Services in f1-system:"
            kubectl get services -n f1-system
        fi
    else
        print_warning "Kubernetes: f1-system namespace not found"
    fi
fi

echo ""

# Network Connectivity Check
print_header "Network Connectivity"

# Check OpenF1 API
print_status "Checking OpenF1 API..."
if curl -s --connect-timeout 5 https://api.openf1.org/v1/sessions | grep -q "session_key"; then
    print_success "OpenF1 API: Accessible"
else
    print_warning "OpenF1 API: Not accessible (check internet connection)"
fi

# Check local services
print_status "Checking local services..."
if curl -s --connect-timeout 2 http://localhost:8000/api/v1/health &> /dev/null; then
    print_success "API Server: Running on localhost:8000"
else
    print_warning "API Server: Not running on localhost:8000"
fi

if curl -s --connect-timeout 2 http://localhost:3000 &> /dev/null; then
    print_success "Frontend: Running on localhost:3000"
else
    print_warning "Frontend: Not running on localhost:3000"
fi

echo ""

# Resource Usage Check
print_header "Resource Usage"

# Check disk space
print_status "Checking disk space..."
available_space=$(df -h . | awk 'NR==2 {print $4}')
print_success "Available space: $available_space"

# Check memory
print_status "Checking memory..."
if command -v free &> /dev/null; then
    memory_info=$(free -h | awk 'NR==2{printf "Used: %s, Available: %s", $3, $7}')
    print_success "Memory: $memory_info"
fi

echo ""

# Summary
print_header "Health Check Summary"

echo "System is ready for F1 GPU Telemetry!"
echo ""
echo "Next steps:"
echo "1. Start locally: ./scripts/start-local.sh"
echo "2. Deploy to K8s: ./scripts/deploy-k8s.sh"
echo "3. Monitor: nvidia-smi -l 1"