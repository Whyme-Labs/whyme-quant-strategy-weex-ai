#!/bin/bash
set -e

# WEEX AI Strategy Engine - Deployment Script
# Usage: ./scripts/deploy.sh [server_ip]

SERVER_IP="${1:-209.182.237.49}"
SERVER_USER="${SERVER_USER:-root}"
APP_DIR="/opt/weex-ai-strategy"
REPO_URL="https://github.com/user/whyme-quant-strategy-weex-ai.git"

echo "=========================================="
echo "WEEX AI Strategy Engine - Deployment"
echo "=========================================="
echo "Server: $SERVER_USER@$SERVER_IP"
echo "App Dir: $APP_DIR"
echo ""

# Check if .env exists locally
if [ ! -f ".env" ]; then
    echo "Error: .env file not found locally"
    echo "Please create .env from .env.example and configure your credentials"
    exit 1
fi

echo "Step 1: Syncing code to server..."
rsync -avz --exclude '.git' --exclude '__pycache__' --exclude '*.pyc' --exclude '.env' \
    ./ "$SERVER_USER@$SERVER_IP:$APP_DIR/"

echo "Step 2: Syncing .env file..."
scp .env "$SERVER_USER@$SERVER_IP:$APP_DIR/.env"

echo "Step 3: Building and starting containers on server..."
ssh "$SERVER_USER@$SERVER_IP" << 'ENDSSH'
cd /opt/weex-ai-strategy

# Stop existing containers
echo "Stopping existing containers..."
docker compose down || true

# Build and start
echo "Building Docker image..."
docker compose build --no-cache

echo "Starting containers..."
docker compose up -d

# Check status
echo ""
echo "Container status:"
docker compose ps

echo ""
echo "Waiting 5 seconds for startup..."
sleep 5

echo ""
echo "Recent logs:"
docker compose logs --tail=50 strategy-engine
ENDSSH

echo ""
echo "=========================================="
echo "Deployment complete!"
echo "=========================================="
echo ""
echo "Useful commands:"
echo "  View logs:    ssh $SERVER_USER@$SERVER_IP 'docker compose -f $APP_DIR/docker-compose.yml logs -f'"
echo "  Restart:      ssh $SERVER_USER@$SERVER_IP 'docker compose -f $APP_DIR/docker-compose.yml restart'"
echo "  Stop:         ssh $SERVER_USER@$SERVER_IP 'docker compose -f $APP_DIR/docker-compose.yml down'"
