import logging
import time
from fastapi import FastAPI, UploadFile, File
from ultralytics import YOLO
from PIL import Image
import io
import json
# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI()

# Load the YOLO model once when the server starts
model = YOLO("fire_best.pt")


@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    start_time = time.time()  # Start time measurement
    logger.info("Received file: %s", file.filename)

    try:
        # Read the image file
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data))
        result = model.predict(image)
        xywh = result[0].boxes.xywh
        conf = result[0].boxes.conf

        # Extracting values into the specified format
        output = {
            "predictions": []
        }

        for i in range(xywh.shape[0]):
            x, y, width, height = xywh[i]
            confidence = conf[i].item()
            output["predictions"].append({
                "x": x.item(),
                "y": y.item(),
                "width": width.item(),
                "height": height.item(),
                "confidence": confidence
            })

        # Convert to JSON string for API response
        json_output = json.dumps(output, indent=2)

    except Exception as e:
        logger.error("Error processing file %s: %s", file.filename, e)
        return {"error": str(e)}

    elapsed_time = time.time() - start_time  # Calculate elapsed time
    logger.info("Time taken for prediction: %.2f seconds", elapsed_time)

    return {"predictions": output, "time_taken": elapsed_time}

