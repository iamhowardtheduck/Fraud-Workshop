#!/bin/bash

# SAR Workshop Quick Setup Script
# This script sets up the complete SAR system for the workshop environment
# Run from: /workspace/workshop

set -e

echo "=== SAR Workshop Quick Setup ==="
echo "Setting up SAR Management System for workshop environment..."
echo ""

# Verify we're in the right directory
if [[ ! "$(pwd)" == *"/workspace/workshop"* ]]; then
    echo "Warning: This script should be run from /workspace/workshop"
    echo "Current directory: $(pwd)"
    read -p "Continue anyway? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Check if Elasticsearch is accessible
echo "Testing Elasticsearch connectivity..."
if curl -s -u fraud:hunter http://kubernetes-vm:30920/_cluster/health > /dev/null; then
    echo "✓ Elasticsearch is accessible at http://kubernetes-vm:30920"
else
    echo "✗ Cannot reach Elasticsearch at http://kubernetes-vm:30920"
    echo "Please ensure the Kubernetes cluster and Elasticsearch service are running."
    exit 1
fi

# Run the installation with sample data
echo ""
echo "Installing SAR system with sample data..."
chmod +x install_sar_system.sh
./install_sar_system.sh --sample-data

echo ""
echo "=== Setup Complete! ==="
echo ""
echo "Your SAR Management System is ready:"
echo "  📂 Installation path: /workspace/workshop/sar-system"
echo "  🔗 Elasticsearch: http://kubernetes-vm:30920"
echo "  👤 Username: fraud"
echo "  📊 Sample data: Loaded"
echo ""
echo "Workshop credentials:"
echo "  Elasticsearch URL: http://kubernetes-vm:30920"
echo "  Username: fraud"
echo "  Password: hunter"
echo ""

# Automatically start the application
cd /workspace/workshop/sar-system
echo "Starting SAR Management System..."
echo "Access the application at: http://localhost:3000"
echo "Press Ctrl+C to stop the application"
echo ""
npm start
