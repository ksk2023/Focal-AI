#!/bin/bash

echo "=========================================="
echo "  FocalAI - AI Photography Assistant"
echo "=========================================="

# Ensure persistent storage directory exists
mkdir -p /mnt/workspace/user_styles/images 2>/dev/null || mkdir -p /home/user/app/user_styles/images

# Initialize user_styles if empty
PROFILE_PATH="/mnt/workspace/user_styles/profile.json"
if [ ! -f "$PROFILE_PATH" ]; then
    echo '{"style_description": ""}' > "$PROFILE_PATH" 2>/dev/null || true
    echo "Initialized empty profile.json"
fi

# Start FastAPI backend with python main.py
echo "Starting FastAPI backend on port 8000..."
cd /home/user/app
python main.py &
BACKEND_PID=$!

# Wait for backend to be ready
echo "Waiting for backend to start..."
sleep 5

# Test backend health
for i in {1..10}; do
    if curl -s http://127.0.0.1:8000/ > /dev/null 2>&1; then
        echo "Backend is ready!"
        break
    fi
    echo "Waiting for backend... ($i/10)"
    sleep 2
done

# Start Nginx on port 7860
echo "Starting Nginx on port 7860..."
nginx -g 'daemon off;' &
NGINX_PID=$!

echo "=========================================="
echo "Services started!"
echo "App URL: http://localhost:7860"
echo "API Docs: http://localhost:7860/docs"
echo "=========================================="

# Handle shutdown signals
trap "echo 'Shutting down...'; kill $BACKEND_PID $NGINX_PID 2>/dev/null; exit 0" SIGTERM SIGINT

# Keep container running
wait $BACKEND_PID $NGINX_PID
