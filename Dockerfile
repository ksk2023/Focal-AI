FROM modelscope-registry.cn-beijing.cr.aliyuncs.com/modelscope-repo/python:3.10

# Install Nginx and system dependencies for OpenCV/MediaPipe
RUN apt-get update && apt-get install -y \
    nginx \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /home/user/app

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

# Copy Nginx config
COPY nginx.conf /etc/nginx/sites-available/default

# Create persistent storage symlink for user_styles
RUN mkdir -p /mnt/workspace/user_styles && \
    rm -rf /home/user/app/user_styles && \
    ln -s /mnt/workspace/user_styles /home/user/app/user_styles

# Make start script executable
RUN chmod +x start.sh

# Expose port 7860
EXPOSE 7860

# Start services
ENTRYPOINT ["./start.sh"]
