FROM modelscope-registry.cn-beijing.cr.aliyuncs.com/modelscope-repo/python:3.10

# Install Nginx, curl and OpenCV/MediaPipe dependencies
RUN apt-get update && apt-get install -y \
    nginx \
    curl \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /home/user/app

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

# Setup Nginx config
COPY nginx.conf /etc/nginx/sites-available/default
RUN rm -f /etc/nginx/sites-enabled/default && \
    ln -s /etc/nginx/sites-available/default /etc/nginx/sites-enabled/default

# Create user_styles directory
RUN mkdir -p /home/user/app/user_styles/images

# Make start script executable
RUN chmod +x start.sh

# Expose port 7860
EXPOSE 7860

# Start services
ENTRYPOINT ["./start.sh"]
