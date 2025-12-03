#!/bin/bash

# SAR (Suspicious Activity Report) Web System Installation Script
# This script installs and configures a web server to display SAR data from Elasticsearch
# Workshop Environment Configuration

set -e

# Configuration for Workshop Environment
APP_DIR="/workspace/workshop/sar-system"
LOAD_SAMPLE_DATA=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --sample-data)
      LOAD_SAMPLE_DATA=true
      shift
      ;;
    --help|-h)
      echo "SAR Web System Installation - Workshop Environment"
      echo "Usage: $0 [options]"
      echo ""
      echo "Options:"
      echo "  --sample-data    Load sample SAR data into Elasticsearch"
      echo "  --help, -h       Show this help message"
      echo ""
      echo "Workshop Configuration:"
      echo "  Elasticsearch URL: http://kubernetes-vm:30920"
      echo "  Username: fraud"
      echo "  Password: hunter"
      echo "  Installation path: $APP_DIR"
      exit 0
      ;;
    *)
      echo "Unknown option $1"
      exit 1
      ;;
  esac
done

echo "=== SAR Web System Installation - Workshop Environment ==="
echo "Installing web server and dependencies..."

# Create application directory in workspace
echo "Setting up application directory at $APP_DIR..."
mkdir -p $APP_DIR
cd $APP_DIR

echo "Current working directory: $(pwd)"

# Check if Node.js is available
if ! command -v node &> /dev/null; then
    echo "Installing Node.js..."
    curl -fsSL https://deb.nodesource.com/setup_18.x | sudo -E bash -
    sudo apt-get install -y nodejs
else
    echo "Node.js already available: $(node --version)"
fi

# Install required system packages if not available
if ! command -v jq &> /dev/null; then
    echo "Installing jq..."
    sudo apt update && sudo apt install -y jq
fi

# Create directory structure
echo "Setting up application structure..."
mkdir -p public/css public/js views

# Copy application files (if they exist in current directory)
if [ -f "../server.js" ] || [ -f "server.js" ]; then
    echo "Application files found locally"
elif [ -f "/workspace/workshop/server.js" ]; then
    echo "Copying application files from workshop directory..."
    cp /workspace/workshop/*.js .
    cp /workspace/workshop/*.json .
    cp /workspace/workshop/*.md .
    cp /workspace/workshop/.env.example .
    cp -r /workspace/workshop/views/* views/ 2>/dev/null || true
    cp -r /workspace/workshop/public/* public/ 2>/dev/null || true
else
    echo "Warning: Application files not found. Please ensure all files are in the correct location."
fi

# Create package.json if it doesn't exist
if [ ! -f "package.json" ]; then
    echo "Creating package.json..."
    cat > package.json << EOF
{
  "name": "sar-management-system",
  "version": "1.0.0",
  "description": "Suspicious Activity Report Management System with Elasticsearch integration",
  "main": "server.js",
  "scripts": {
    "start": "node server.js",
    "dev": "nodemon server.js"
  },
  "dependencies": {
    "@elastic/elasticsearch": "^8.12.0",
    "body-parser": "^1.20.2",
    "compression": "^1.7.4",
    "cors": "^2.8.5",
    "ejs": "^3.1.9",
    "express": "^4.18.2",
    "express-rate-limit": "^7.1.5",
    "helmet": "^7.1.0",
    "morgan": "^1.10.0"
  }
}
EOF
fi

# Install Node.js dependencies
echo "Installing Node.js dependencies..."
npm install

# Create .env file with workshop configuration
echo "Creating .env configuration for workshop environment..."
cat > .env << EOF
# SAR Management System Configuration - Workshop Environment
PORT=3000
NODE_ENV=development

# Elasticsearch Configuration - Workshop Environment
ELASTICSEARCH_URL=http://kubernetes-vm:30920
ELASTICSEARCH_USERNAME=fraud
ELASTICSEARCH_PASSWORD=hunter
ELASTICSEARCH_INDEX=sar-reports

# Security Configuration
SESSION_SECRET=workshop-demo-secret-key
EOF

# Make scripts executable if they exist
chmod +x *.sh 2>/dev/null || true

echo "=== Installation Complete ==="
echo "Application directory: $APP_DIR"
echo ""
echo "Workshop Configuration Applied:"
echo "  Elasticsearch URL: http://kubernetes-vm:30920"
echo "  Username: fraud"
echo "  Password: hunter"
echo "  Index: sar-reports"
echo ""

# Load sample data if requested
if [ "$LOAD_SAMPLE_DATA" = true ]; then
    echo "=== Loading Sample Data ==="
    if [ -f "load-sample-data.sh" ]; then
        echo "Loading sample SAR data into Elasticsearch..."
        export ELASTICSEARCH_URL="http://kubernetes-vm:30920"
        export ELASTICSEARCH_USERNAME="fraud"
        export ELASTICSEARCH_PASSWORD="hunter"
        export ELASTICSEARCH_INDEX="sar-reports"
        
        ./load-sample-data.sh
    else
        echo "Sample data script not found. Skipping data load."
    fi
fi

echo ""
echo "=== Next Steps ==="
echo "1. Test Elasticsearch connectivity:"
echo "   curl -u fraud:hunter http://kubernetes-vm:30920/_cluster/health"
echo ""
echo "2. Start the application:"
echo "   cd $APP_DIR"
echo "   npm start"
echo ""
echo "3. Access the web interface:"
echo "   http://localhost:3000"
echo ""
echo "4. (Optional) Load sample data:"
echo "   ./load-sample-data.sh"
echo ""

# Test Elasticsearch connectivity
echo "=== Workshop Connectivity Test ==="
echo "Testing Elasticsearch connection to kubernetes-vm:30920..."
if curl -s -u fraud:hunter http://kubernetes-vm:30920/_cluster/health &> /dev/null; then
    echo "✓ Successfully connected to Elasticsearch cluster"
else
    echo "⚠ Cannot connect to Elasticsearch at kubernetes-vm:30920"
    echo "  Please verify the cluster is running and accessible"
fi
