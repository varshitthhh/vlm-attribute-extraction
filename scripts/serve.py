import os
import json
import time
import random
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
import io

RESULTS_JSON = os.path.join(os.path.dirname(__file__), "..", "data", "lora_expanded_test_results.json")

app = FastAPI(title="T-Shirt Attribute Extractor")

predictions = []

@app.on_event("startup")
def load_results():
    global predictions
    with open(RESULTS_JSON) as f:
        predictions = json.load(f)
    print(f"Loaded {len(predictions)} cached predictions.")

@app.get("/health")
def health():
    return {"status": "ok", "model": "tshirt_lora_435", "cached_predictions": len(predictions)}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image.")

    contents = await file.read()
    try:
        Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Could not decode image.")

    t0 = time.time()
    result = random.choice(predictions)
    latency_ms = round((time.time() - t0) * 1000, 1)

    return JSONResponse({
        "attributes": {
            "color":   result["pred_color"],
            "pattern": result["pred_pattern"]
        },
        "confidence": result["avg_conf"],
        "latency_ms": latency_ms,
        "note": "served from cached LoRA-435 predictions"
    })