#!/bin/bash

echo "=========================================="
echo "  FocalAI - AI Photography Assistant"
echo "=========================================="

# Ensure persistent storage directory exists
mkdir -p /mnt/workspace/user_styles/images

# Initialize user_styles if empty
if [ ! -f /mnt/workspace/user_styles/profile.json ]; then
    echo '{"style_description": ""}' > /mnt/workspace/user_styles/profile.json
    echo "Initialized empty profile.json"
fi

# Start FastAPI backend
echo "Starting FastAPI backend on port 8000..."
uvicorn main:app --host 0.0.0.0 --port 8000 &
BACKEND_PID=$!

# Wait for backend to start
sleep 3

# Start Nginx
echo "Starting Nginx on port 7860..."
nginx -g 'daemon off;' &
NGINX_PID=$!

echo "=========================================="
echo "Services started!"
echo "Frontend: http://localhost:7860"
echo "Backend API: http://localhost:7860/api/"
echo "API Docs: http://localhost:7860/docs"
echo "=========================================="

# Keep container running and handle signals
trap "kill $BACKEND_PID $NGINX_PID; exit 0" SIGTERM SIGINT

# Wait for processes
wait $BACKEND_PID $NGINX_PID
