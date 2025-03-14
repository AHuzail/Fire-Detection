FROM python:3.9-slim

WORKDIR /app

# Install OpenGL dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libgl1-mesa-dev \
    libglib2.0-0 \
    libgtk2.0-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libv4l-dev \
    libxvidcore-dev \
    libx264-dev \
    libjpeg-dev \
    libpng-dev \
    libatlas-base-dev \
    gfortran \
    libsm6 \
    libxext6 \
    libxrender-dev

# Create necessary directories
RUN mkdir -p /app/api /app/app /app/models

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy model and code
COPY models/fire_best.pt /app/models/
COPY api/ /app/api/
COPY app/ /app/app/
COPY run.py /app/

# Expose ports for both FastAPI and Streamlit
EXPOSE 8000 8501

# Run both servers
CMD ["python", "run.py"]
