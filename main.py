# Install necessary dependencies:
# pip install fastapi uvicorn[standard] pillow ultralytics

from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import JSONResponse
from PIL import Image
import io
import time
import logging
from functools import wraps
from inference_sdk import InferenceHTTPClient

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

# Initialize the Inference client
CLIENT = InferenceHTTPClient(api_url="https://detect.roboflow.com", api_key='7tusGpEEX3t2pAwcL3re')

# Initialize the FastAPI app
app = FastAPI()


# ====== Decorators ======

def log_request(func):
    """Logs the details of the incoming request."""

    @wraps(func)
    async def wrapper(*args, **kwargs):
        logging.info(f"Processing request: {func.__name__}")
        return await func(*args, **kwargs)

    return wrapper


def measure_time(func):
    """Measures the time taken to execute a request."""

    @wraps(func)
    async def wrapper(*args, **kwargs):
        start_time = time.time()
        result = await func(*args, **kwargs)
        elapsed_time = time.time() - start_time
        logging.info(f"{func.__name__} executed in {elapsed_time:.2f} seconds")
        return result

    return wrapper


def validate_image_format(func):
    """Validates if the uploaded file is a JPEG or PNG image."""

    @wraps(func)
    async def wrapper(image: UploadFile = File(...), *args, **kwargs):
        if image.content_type not in ["image/jpeg", "image/png"]:
            raise HTTPException(status_code=400, detail="Invalid image format. Only JPEG and PNG are supported.")
        return await func(image, *args, **kwargs)

    return wrapper


# ====== Route with Decorators ======

@app.post("/predict/")
@log_request
@measure_time
@validate_image_format
async def predict_fire(image: UploadFile = File(...)):
    # Read the image file
    contents = await image.read()
    img = Image.open(io.BytesIO(contents))

    # Call the inference client
    results = CLIENT.infer(img, model_id="fire-detection-new-rc3st/3")

    # Extract predictions
    response = results.get('predictions', [])

    if not response:
        return JSONResponse(content={"message": "No fire detected in the image."})

    return JSONResponse(content={"detections": response})

# To run the app: uvicorn main:app --reload --host 0.0.0.0 --port 8000
