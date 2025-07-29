# Fire Detection System

A comprehensive system for detecting fire in images using YOLO (You Only Look Once) object detection model, with both REST API and web interface.


## Features

- REST API for fire detection in images
- Interactive web interface for uploading and analyzing images
- Real-time detection results with bounding boxes and confidence scores
- Docker support for easy deployment
- Detailed metrics and visualization

## Requirements

- Python 3.9+
- FastAPI
- Streamlit
- Ultralytics YOLO
- OpenCV
- Other dependencies in requirements.txt

## Installation

### Local Installation

1. Clone this repository:
```bash
git clone https://github.com/AHuzail/Fire-Detection.git
cd Fire-Detection
```

2. Install the required packages:
```bash
pip install -r requirements.txt
```


### Docker Installation

1. Build the Docker image:
```bash
docker build -t fire-detection .
```

2. Run the container:
```bash
docker run -p 8000:8000 -p 8501:8501 fire-detection
```

## Usage

### Running Both Services

Use the convenience script to run both the API and web UI:

```bash
python run.py
```

### Running Only the API

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

The API server will start on http://localhost:8000

### Running Only the Web UI

```bash
streamlit run app/app.py
```
