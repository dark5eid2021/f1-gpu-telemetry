# ============================================================================
# scripts/deploy-k8s.sh - Kubernetes Deployment Script
# ============================================================================

#!/bin/bash

# F1 GPU Telemetry System - Kubernetes Deployment Script
# Deploys the complete F1 system to Kubernetes

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

echo "☸️ Deploying F1 GPU Telemetry System to Kubernetes..."

# Check if kubectl is available
if ! command -v kubectl &> /dev/null; then
    print_error "kubectl not found. Please install kubectl first."
    exit 1
fi

# Check if cluster is accessible
if ! kubectl cluster-info &> /dev/null; then
    print_error "Cannot connect to Kubernetes cluster"
    exit 1
fi

print_status "Connected to Kubernetes cluster"
kubectl cluster-info

# Deployment order matters - dependencies first
DEPLOYMENT_ORDER=(
    "k8s/namespace.yaml"
    "k8s/configmaps.yaml"
    "k8s/storage.yaml"
    "k8s/timescaledb.yaml"
    "k8s/redis.yaml"
    "k8s/kafka.yaml"
    "k8s/gpu-processor.yaml"
    "k8s/web-app.yaml"
    "k8s/ingress.yaml"
    "k8s/monitoring.yaml"
    "k8s/hpa.yaml"
)

# Deploy each component
for manifest in "${DEPLOYMENT_ORDER[@]}"; do
    if [ -f "$manifest" ]; then
        print_status "Applying $manifest..."
        kubectl apply -f "$manifest"
    else
        print_warning "$manifest not found, skipping..."
    fi
done

# Wait for deployments to be ready
print_status "Waiting for deployments to be ready..."

DEPLOYMENTS=(
    "timescaledb"
    "redis" 
    "kafka"
    "f1-gpu-processor"
    "f1-web-app"
)

for deployment in "${DEPLOYMENTS[@]}"; do
    print_status "Waiting for $deployment..."
    kubectl rollout status deployment/$deployment -n f1-system --timeout=300s || {
        print_warning "Timeout waiting for $deployment"
    }
done

# Show deployment status
print_status "Deployment status:"
kubectl get pods -n f1-system
echo ""
kubectl get services -n f1-system

print_success "🎉 Kubernetes deployment complete!"

# Show access information
echo ""
echo "📊 Access Information:"
echo ""

# Get service endpoints
WEB_APP_IP=$(kubectl get service f1-web-app-lb -n f1-system -o jsonpath='{.status.loadBalancer.ingress[0].ip}' 2>/dev/null || echo "pending")
API_IP=$(kubectl get service f1-gpu-processor-lb -n f1-system -o jsonpath='{.status.loadBalancer.ingress[0].ip}' 2>/dev/null || echo "pending")

if [ "$WEB_APP_IP" != "pending" ] && [ ! -z "$WEB_APP_IP" ]; then
    echo "🌐 Dashboard: http://$WEB_APP_IP"
else
    echo "🌐 Dashboard: kubectl port-forward svc/f1-web-app 3000:80 -n f1-system"
fi

if [ "$API_IP" != "pending" ] && [ ! -z "$API_IP" ]; then
    echo "🚀 API: http://$API_IP:8000"
else
    echo "🚀 API: kubectl port-forward svc/f1-gpu-processor 8000:8000 -n f1-system"
fi

echo "📈 Monitoring:"
echo "   Grafana: kubectl port-forward svc/grafana 3000:3000 -n f1-system"
echo "   Prometheus: kubectl port-forward svc/prometheus 9090:9090 -n f1-system"
