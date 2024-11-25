# Fire Detection API

This API is designed for detecting fire in images using a pre-trained YOLO (You Only Look Once) model. The API accepts image uploads (JPEG or PNG format) and returns predictions with bounding box coordinates and confidence scores. 

## Deployment

The API is hosted on Railway and can be accessed via the following link:
- **API URL**: [https://fire-detection-production.up.railway.app/](https://fire-detection-production.up.railway.app/)

## Features

- **Image Upload**: Accepts image files (JPEG or PNG).
- **Fire Detection**: Runs a YOLO model to detect fire in the image.
- **Bounding Box Output**: Provides bounding box coordinates and confidence scores for each detected instance of fire.
- **Logging**: Logs each prediction request with details for monitoring and debugging.

## Endpoints

### `POST /predict/`

Upload an image for fire detection. The endpoint processes the image through the YOLO model and returns bounding boxes and confidence scores for each detected fire area.

#### Request

- **Method**: `POST`
- **URL**: `/predict/`
- **Parameters**: 
  - `file`: Image file (JPEG or PNG)

#### Example Request


```shell
curl -X POST "https://fire-detection-production.up.railway.app/predict/" -F "file=@path_to_your_image.jpg"
