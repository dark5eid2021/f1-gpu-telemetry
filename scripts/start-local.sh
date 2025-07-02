# ============================================================================
# scripts/start-local.sh - Local Development Startup Script
# ============================================================================

#!/bin/bash

# F1 GPU Telemetry System - Local Development Startup
# Starts the system locally with Docker Compose

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
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

echo "🏎️ Starting F1 GPU Telemetry System locally..."

# Check prerequisites
if ! command -v docker &> /dev/null; then
    print_error "Docker not found. Please install Docker first."
    exit 1
fi

if ! docker run --rm --gpus all nvidia/cuda:12.0-base-ubuntu20.04 nvidia-smi &> /dev/null; then
    print_warning "GPU not available or NVIDIA Container Runtime not configured"
    print_status "System will run in CPU-only mode"
fi

# Check if .env exists
if [ ! -f ".env" ]; then
    print_error ".env file not found"
    print_status "Please run: cp .env.example .env"
    print_status "Then edit .env with your API keys"
    exit 1
fi

# Source environment variables
source .env

# Validate required secrets
if [[ "$OPENF1_API_KEY" == *"your_"* ]] || [[ -z "$OPENF1_API_KEY" ]]; then
    print_warning "OpenF1 API key not configured - real-time data will not work"
fi

# Start with Docker Compose
if [ -f "docker-compose.gpu.yml" ]; then
    print_status "Starting services with Docker Compose..."
    docker-compose -f docker-compose.gpu.yml up -d
    
    print_status "Waiting for services to start..."
    sleep 10
    
    # Check service health
    print_status "Checking service health..."
    
    # Check if services are running
    if docker-compose -f docker-compose.gpu.yml ps | grep -q "Up"; then
        print_success "Services started successfully!"
    else
        print_error "Some services failed to start"
        docker-compose -f docker-compose.gpu.yml ps
        exit 1
    fi
    
else
    print_error "docker-compose.gpu.yml not found"
    print_status "Starting services manually..."
    
    # Start individual containers
    docker run -d --name f1-redis -p 6379:6379 redis:7-alpine
    docker run -d --name f1-kafka -p 9092:9092 -e KAFKA_ZOOKEEPER_CONNECT=localhost:2181 confluentinc/cp-kafka:latest
    
    print_status "Starting main application..."
    python src/main.py &
fi

# Show access information
echo ""
print_success "🎉 F1 GPU Telemetry System is running!"
echo ""
echo "📊 Access URLs:"
echo "   Dashboard: http://localhost:3000"
echo "   API: http://localhost:8000"
echo "   API Docs: http://localhost:8000/docs"
echo ""
echo "🔧 Monitoring:"
echo "   GPU Usage: nvidia-smi -l 1"
echo "   Logs: docker-compose -f docker-compose.gpu.yml logs -f"
echo ""
echo "🛑 To stop: docker-compose -f docker-compose.gpu.yml down"
