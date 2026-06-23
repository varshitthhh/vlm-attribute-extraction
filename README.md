# VLM Attribute Extraction — Confidence-Aware T-Shirt Attribute Extraction

Extracts color and pattern attributes from t-shirt product images using a fine-tuned vision-language model (Qwen2.5-VL-3B-Instruct) with LoRA adapters trained via distant supervision.

## Architecture
HF Fashion Dataset (700 t-shirts)

↓

Weak Label Parsing (regex on titles)

↓

Labeled Split: 180 train / 47 test

↓

Zero-Shot Baseline (Qwen2.5-VL-3B, 4-bit)

↓

LoRA Fine-Tuning (unsloth, 180 samples)

↓

Pseudo-Labeling (469 unlabeled, conf >= 0.95 -> 255 accepted)

↓

Expanded LoRA Training (435 samples)

↓

FastAPI Inference Server (/predict, /health)
## Results

| Metric      | Zero-Shot | LoRA-180 | LoRA-435 |
|-------------|-----------|----------|----------|
| Color Acc   | 66.1%     | 66.0%    | 68.1%    |
| Pattern Acc | 53.7%     | 89.4%    | 89.4%    |
| Color F1    | 0.514     | 0.439    | 0.449    |
| Pattern F1  | 0.562     | 0.822    | 0.822    |
| Confidence  | 0.840     | 0.953    | 0.965    |

## Stack

- Model: Qwen2.5-VL-3B-Instruct
- Fine-tuning: LoRA via unsloth (Colab T4)
- Data: ashraq/fashion-product-images-small (HuggingFace)
- Serving: FastAPI + uvicorn
- Training: Google Colab (free tier)

## Setup

```bash
python -m venv venv
venv\Scripts\activate
pip install transformers accelerate peft pillow datasets fastapi uvicorn python-multipart requests
```

## Usage

```bash
# Start server
python -m uvicorn scripts.serve:app --host 0.0.0.0 --port 8000

# Test health
Invoke-WebRequest -Uri http://localhost:8000/health -UseBasicParsing | Select-Object -ExpandProperty Content

# Predict
python -c "import requests; f=open('data/test_images/18237.jpg','rb'); r=requests.post('http://localhost:8000/predict',files={'file':('18237.jpg',f,'image/jpeg')}); print(r.json())"
```

## Project Structure
vlm_attribute_extraction/

├── data/

│   ├── labeled_test.json

│   ├── lora_expanded_test_results.json

│   ├── zero_shot_results.json

│   └── test_images/

├── models/                        # LoRA adapter (tshirt_lora_435)

├── scripts/

│   ├── serve.py                   # FastAPI server

│   ├── visualize_results.py       # Visual proof grid

│   ├── download_test_images.py    # Re-download test images

│   └── predict.py                 # Single image inference

├── logs/

│   └── visual_proof.png           # Predicted vs actual grid

└── README.md
## Adapter

LoRA adapter (~114MB) saved to Google Drive (not in repo due to size).
Base model: Qwen/Qwen2.5-VL-3B-Instruct (auto-downloaded from HuggingFace).
