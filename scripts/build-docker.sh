# ============================================================================
# scripts/build-docker.sh - Docker Build Script
# ============================================================================

#!/bin/bash

# F1 GPU Telemetry System - Docker Build Script
# Builds all Docker images for the F1 system

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
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

echo "🐳 Building F1 GPU Telemetry Docker Images..."

# Check if Docker is available
if ! command -v docker &> /dev/null; then
    echo "❌ Docker not found. Please install Docker first."
    exit 1
fi

# Get version tag (default to 'latest')
VERSION_TAG=${1:-latest}
REGISTRY=${DOCKER_REGISTRY:-"f1-gpu"}

print_status "Building images with tag: $VERSION_TAG"

# Build GPU processor image
if [ -f "docker/gpu-processor.Dockerfile" ]; then
    print_status "Building GPU processor image..."
    docker build \
        -t $REGISTRY/f1-gpu-processor:$VERSION_TAG \
        -f docker/gpu-processor.Dockerfile \
        .
    print_success "GPU processor image built: $REGISTRY/f1-gpu-processor:$VERSION_TAG"
else
    print_warning "docker/gpu-processor.Dockerfile not found, skipping..."
fi

# Build web app image
if [ -f "docker/web-app.Dockerfile" ]; then
    print_status "Building web app image..."
    docker build \
        -t $REGISTRY/f1-web-app:$VERSION_TAG \
        -f docker/web-app.Dockerfile \
        .
    print_success "Web app image built: $REGISTRY/f1-web-app:$VERSION_TAG"
else
    print_warning "docker/web-app.Dockerfile not found, skipping..."
fi

# Show built images
print_status "Built images:"
docker images | grep f1-

print_success "🎉 Docker build complete!"

# Optional: Push to registry
if [ "$2" = "--push" ] && [ ! -z "$DOCKER_REGISTRY" ]; then
    print_status "Pushing images to registry..."
    docker push $REGISTRY/f1-gpu-processor:$VERSION_TAG
    docker push $REGISTRY/f1-web-app:$VERSION_TAG
    print_success "Images pushed to registry"
fi

