#!/bin/bash
# ==============================================================================
# Parcel Intelligence - EC2 One-Click Docker Deployment Script
# Compatible with Ubuntu 22.04 / 24.04 LTS and Amazon Linux 2023
# ==============================================================================
set -e

echo "🚀 [1/4] Checking and Installing Docker & Docker Compose..."

if ! command -v docker &> /dev/null; then
    echo "Installing Docker..."
    if [ -f /etc/os-release ]; then
        . /etc/os-release
        if [[ "$ID" == "ubuntu" || "$ID" == "debian" ]]; then
            sudo apt-get update
            sudo apt-get install -y ca-certificates curl gnupg
            sudo install -m 0755 -d /etc/apt/keyrings
            curl -fsSL https://download.docker.com/linux/$ID/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
            sudo chmod a+r /etc/apt/keyrings/docker.gpg
            echo \
              "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/$ID \
              $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
              sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
            sudo apt-get update
            sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
        elif [[ "$ID" == "amzn" ]]; then
            sudo dnf update -y
            sudo dnf install -y docker
            sudo systemctl enable --now docker
            # Install compose plugin
            DOCKER_CONFIG=${DOCKER_CONFIG:-$HOME/.docker}
            mkdir -p $DOCKER_CONFIG/cli-plugins
            curl -SL https://github.com/docker/compose/releases/latest/download/docker-compose-linux-$(uname -m) -o $DOCKER_CONFIG/cli-plugins/docker-compose
            chmod +x $DOCKER_CONFIG/cli-plugins/docker-compose
        fi
    fi
    sudo usermod -aG docker $USER || true
    sudo systemctl restart docker || true
fi

echo "✅ Docker is installed: $(docker --version)"

echo "📋 [2/4] Verifying Environment Configuration..."
if [ ! -f .env ]; then
    if [ -f .env.example ]; then
        echo "Creating .env from .env.example..."
        cp .env.example .env
        echo "⚠️  PLEASE EDIT .env TO CONFIGURE YOUR S3 BUCKET AND CREDENTIALS BEFORE CONTINUING!"
    else
        echo "❌ Error: Neither .env nor .env.example was found!"
        exit 1
    fi
fi

echo "📦 [3/4] Building and starting containers with Docker Compose..."
if docker compose version &> /dev/null; then
    docker compose up -d --build
else
    docker-compose up -d --build
fi

echo "⏳ [4/4] Waiting for services to initialize..."
sleep 10

if command -v curl &> /dev/null; then
    for i in {1..12}; do
        if curl -s http://localhost:8000/api/health | grep -q "ready"; then
            echo "🎉 [SUCCESS] Backend service is healthy and ready!"
            break
        fi
        echo "Waiting for health check (attempt $i/12)..."
        sleep 5
    done
fi

echo "================================================================================"
echo " Parcel Intelligence is running!"
echo " Access the Web Dashboard at: http://<YOUR-EC2-PUBLIC-IP>"
echo " Direct Backend API docs at: http://<YOUR-EC2-PUBLIC-IP>:8000/docs"
echo "================================================================================"
