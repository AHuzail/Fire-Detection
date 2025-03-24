# Fire Detection System

A comprehensive system for detecting fire in images using YOLO (You Only Look Once) object detection model, with both REST API and web interface.

## Project Structure

```
Fire-Detection/
├── api/                  # FastAPI application
│   └── main.py           # API entry point
├── app/                  # Streamlit web application
│   └── app.py            # Web UI entry point
├── models/               # Model files
│   └── fire_best.pt      # YOLO fire detection model
├── utils/                # Shared utilities
├── Dockerfile            # Main Dockerfile
├── docker-compose.yml    # Docker compose configuration
├── requirements.txt      # Project dependencies
└── run.py                # Script to run both API and web UI
```

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

The Streamlit interface will be available at http://localhost:8501

## API Documentation

### Endpoints

#### POST /predict/

Upload an image file to get fire detection results.

**Request:**
- Form data with a file field named "file" containing the image (JPEG or PNG format)

**Response:**
```json
{
  "predictions": [
    {
      "x": 123.4,
      "y": 234.5,
      "width": 100.0,
      "height": 100.0,
      "confidence": 0.95
    }
  ],
  "time_taken": 0.45
}
```

### Example API Usage

#### With Python

```python
import requests

url = "http://localhost:8000/predict/"
files = {"file": open("path/to/your/image.jpg", "rb")}
response = requests.post(url, files=files)
print(response.json())
```

## Web Interface

The Streamlit web interface provides:

- User-friendly image upload interface
- Visualization of detection results with bounding boxes
- Confidence scores and detailed metrics for each detection
- Performance metrics like processing time

## Model Information

The system uses a YOLOv11 model trained specifically for fire detection. The model file (`models/fire_best.pt`) contains pre-trained weights for identifying fire instances in various environments.
