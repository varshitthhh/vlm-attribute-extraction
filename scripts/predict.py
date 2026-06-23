# scripts/predict.py
import os
import json
import torch
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from peft import PeftModel

# ── Paths ─────────────────────────────────────────────────────────────
BASE_MODEL  = "Qwen/Qwen2.5-VL-3B-Instruct"
ADAPTER_DIR = os.path.join(os.path.dirname(__file__), "..", "models")

# ── Labels ────────────────────────────────────────────────────────────
COLOR_LIST   = ['Beige','Black','Blue','Brown','Charcoal','Cream','Green','Grey',
                'Lavender','Maroon','Mustard','Navy Blue','Olive','Orange','Pink',
                'Purple','Red','Rust','Teal','White','Yellow']
PATTERN_LIST = ['Printed','Striped','Solid']

def make_prompt():
    return (
        f"Look at this t-shirt image. Answer with exactly two lines:\n"
        f"Color: <one of: {', '.join(COLOR_LIST)}>\n"
        f"Pattern: <one of: {', '.join(PATTERN_LIST)}>\n"
        f"No explanation. No extra text."
    )

PROMPT = make_prompt()

# ── Load model ────────────────────────────────────────────────────────
def load_model():
    print("Loading base model...")
    base = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float32,
        device_map="cpu"
    )
    print("Loading LoRA adapter...")
    model = PeftModel.from_pretrained(base, ADAPTER_DIR)
    model.eval()
    processor = AutoProcessor.from_pretrained(ADAPTER_DIR)
    print("Model ready.")
    return model, processor

# ── Inference ─────────────────────────────────────────────────────────
def predict(image_path, model, processor):
    image = Image.open(image_path).convert("RGB")
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text",  "text": PROMPT}
        ]
    }]
    text   = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], return_tensors="pt")

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=30,
            return_dict_in_generate=True,
            output_scores=True,
            do_sample=False
        )

    generated_ids = out.sequences[0][inputs["input_ids"].shape[1]:]
    decoded = processor.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    token_confs = [torch.softmax(s[0], dim=-1).max().item() for s in out.scores]
    avg_conf    = round(sum(token_confs) / len(token_confs), 4)

    color, pattern = None, None
    for line in decoded.splitlines():
        line = line.strip()
        if line.lower().startswith("color:"):
            color = line.split(":", 1)[1].strip()
        elif line.lower().startswith("pattern:"):
            pattern = line.split(":", 1)[1].strip()

    return {
        "color":    color,
        "pattern":  pattern,
        "avg_conf": avg_conf,
        "decoded":  decoded
    }

# ── Smoke test ────────────────────────────────────────────────────────
if __name__ == "__main__":
    import time

    model, processor = load_model()

    with open(os.path.join(os.path.dirname(__file__), "..", "data", "labeled_test.json")) as f:
        test = json.load(f)

    sample = test[0]
    # labeled_test.json has Colab paths — just use the image from HF dataset directly
    # We'll test with a placeholder check first
    print(f"\nSample id     : {sample['id']}")
    print(f"GT color      : {sample['attributes']['color']}")
    print(f"GT pattern    : {sample['attributes']['pattern']}")
    print(f"Image path    : {sample['image_path']}")
    print("\nNote: image_path points to Colab /content/. We will fix paths in next step.")