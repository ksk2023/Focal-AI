#!/bin/bash

# Kill ports if needed (optional, uncomment if you want auto-cleanup)
# fuser -k 8000/tcp
# fuser -k 3001/tcp

echo "🚀 Starting AI Photography Assistant..."

# Start Backend
echo "Starting Backend (Port 8000)..."
nohup uvicorn main:app --reload --port 8000 > backend.log 2>&1 &
BACKEND_PID=$!

# Start Frontend
echo "Starting Frontend (Port 3001)..."
cd mobile
nohup python -m http.server 0.0.0.0:3001 > frontend.log 2>&1 &
FRONTEND_PID=$!
cd ..

echo "✅ Services started!"
echo "-----------------------------------"
echo "📱 Frontend: http://$(hostname -I | awk '{print $1}'):3001"
echo "🔧 Backend:  http://$(hostname -I | awk '{print $1}'):8000"
echo "📱 手机访问: http://$(hostname -I | awk '{print $1}'):3001?api=http://$(hostname -I | awk '{print $1}'):8000"
echo "-----------------------------------"
echo "Logs are being written to backend.log and mobile/frontend.log"
echo "To stop services, run: kill $BACKEND_PID $FRONTEND_PID"
