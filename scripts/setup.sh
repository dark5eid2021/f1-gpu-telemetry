# F1 GPU Telemetry System Setup Script

echo "🏎️ Setting up F1 GPU Telemetry System..."

# Check prerequisites
check_prerequisites() {
    echo "Checking prerequisites..."
    
    # Check for NVIDIA GPU
    if ! nvidia-smi &> /dev/null; then
        echo "❌ NVIDIA GPU not detected or drivers not installed"
        exit 1
    fi
    
    # Check for Docker
    if ! docker --version &> /dev/null; then
        echo "❌ Docker not installed"
        exit 1
    fi
    
    # Check for NVIDIA Container Runtime
    if ! docker run --rm --gpus all nvidia/cuda:11.0-base-ubuntu20.04 nvidia-smi &> /dev/null; then
        echo "❌ NVIDIA Container Runtime not configured"
        exit 1
    fi
    
    # Check for kubectl
    if ! kubectl version --client &> /dev/null; then
        echo "❌ kubectl not installed"
        exit 1
    fi
    
    echo "✅ Prerequisites check passed"
}

# Setup Python environment
setup_python_env() {
    echo "Setting up Python environment..."
    
    # Create conda environment
    conda create -n f1-gpu python=3.10 -y
    conda activate f1-gpu
    
    # Install RAPIDS
    conda install -c rapidsai -c nvidia rapids=24.02 python=3.10 cuda-version=12.0 -y
    
    # Install additional dependencies
    pip install -r requirements.txt
    
    echo "✅ Python environment setup complete"
}

# Download sample data
download_data() {
    echo "Downloading F1 historical data..."
    
    mkdir -p data/historical
    python scripts/download_historical_data.py --years 2018-2024 --output data/historical/
    
    echo "✅ Data download complete"
}

# Build Docker images
build_images() {
    echo "Building Docker images..."
    
    docker build -t f1-gpu-processor:latest -f docker/gpu-processor.Dockerfile .
    docker build -t f1-web-app:latest -f docker/web-app.Dockerfile .
    
    echo "✅ Docker images built"
}

# Deploy to Kubernetes
deploy_k8s() {
    echo "Deploying to Kubernetes..."
    
    # Create namespace
    kubectl apply -f k8s/namespace.yaml
    
    # Apply configurations
    kubectl apply -f k8s/configmaps.yaml
    kubectl apply -f k8s/secrets.yaml
    kubectl apply -f k8s/storage.yaml
    
    # Deploy services
    kubectl apply -f k8s/timescaledb.yaml
    kubectl apply -f k8s/kafka.yaml
    kubectl apply -f k8s/redis.yaml
    kubectl apply -f k8s/gpu-processor.yaml
    kubectl apply -f k8s/web-app.yaml
    
    # Wait for deployments
    kubectl rollout status deployment/f1-gpu-processor -n f1-system
    kubectl rollout status deployment/f1-web-app -n f1-system
    
    echo "✅ Kubernetes deployment complete"
}

# Main setup flow
main() {
    check_prerequisites
    setup_python_env
    download_data
    build_images
    deploy_k8s
    
    echo "🎉 F1 GPU Telemetry System setup complete!"
    echo ""
    echo "Access the dashboard at: http://localhost:3000"
    echo "API endpoint: http://localhost:8000"
    echo ""
    echo "To monitor GPU usage: nvidia-smi -l 1"
    echo "To check pod status: kubectl get pods -n f1-system"
}

main "$@"
