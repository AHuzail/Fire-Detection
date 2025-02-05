import logging
import time
from fastapi import FastAPI, UploadFile, File, HTTPException
from ultralytics import YOLO
from PIL import Image
import io

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI()

model = YOLO("fire_best.pt")


@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    start_time = time.time()
    logger.info("Received file: %s", file.filename)

    if file.content_type not in ["image/jpeg", "image/png"]:
        logger.error("Unsupported file type: %s", file.content_type)
        raise HTTPException(status_code=400, detail="File must be an image (JPEG or PNG)")

    try:
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data))

        result = model.predict(image)
        xywh = result[0].boxes.xywh
        conf = result[0].boxes.conf
        
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

    except Exception as e:
        logger.error("Error processing file %s: %s", file.filename, e)
        return {"error": str(e)}

    elapsed_time = time.time() - start_time 
    logger.info("Time taken for prediction: %.2f seconds", elapsed_time)
    logger.info("Number of predictions: %d", len(output["predictions"]))

    return {
        "predictions": output["predictions"],
        "time_taken": elapsed_time
    }
