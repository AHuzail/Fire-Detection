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

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
