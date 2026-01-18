#!/bin/bash

echo "=========================================="
echo "  FocalAI - AI Photography Assistant"
echo "=========================================="

# Create persistent storage directory
mkdir -p /mnt/workspace/user_styles/images 2>/dev/null || mkdir -p /home/user/app/user_styles/images

# Start FastAPI backend (python main.py)
echo "Starting FastAPI backend on port 8000..."
cd /home/user/app
python main.py &
BACKEND_PID=$!

# Wait for backend to be ready
echo "Waiting for backend to start..."
sleep 5

for i in {1..10}; do
    if curl -s http://127.0.0.1:8000/ > /dev/null 2>&1; then
        echo "Backend is ready!"
        break
    fi
    echo "Waiting... ($i/10)"
    sleep 2
done

# Start Nginx (serves frontend on port 7860)
echo "Starting Nginx on port 7860..."
nginx -g 'daemon off;' &
NGINX_PID=$!

echo "=========================================="
echo "Services started!"
echo "Frontend: http://localhost:7860"
echo "API: http://localhost:7860/api/"
echo "Docs: http://localhost:7860/docs"
echo "=========================================="

# Handle shutdown
trap "kill $BACKEND_PID $NGINX_PID 2>/dev/null; exit 0" SIGTERM SIGINT

wait $BACKEND_PID $NGINX_PID
