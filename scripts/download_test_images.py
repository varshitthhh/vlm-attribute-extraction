import os
import json
from datasets import load_dataset
from PIL import Image

OUT_DIR   = os.path.join(os.path.dirname(__file__), "..", "data", "test_images")
TEST_JSON = os.path.join(os.path.dirname(__file__), "..", "data", "labeled_test.json")

os.makedirs(OUT_DIR, exist_ok=True)

with open(TEST_JSON) as f:
    test = json.load(f)

target_ids = {str(s["id"]) for s in test}
print(f"Need {len(target_ids)} images. Loading dataset (streaming)...")

ds = load_dataset("ashraq/fashion-product-images-small", split="train", streaming=True)

saved  = {}
for item in ds:
    item_id = str(item["id"])
    if item_id in target_ids and item_id not in saved:
        img_path = os.path.join(OUT_DIR, f"{item_id}.jpg")
        item["image"].convert("RGB").save(img_path, quality=90)
        saved[item_id] = img_path
        print(f"  Saved {item_id}  ({len(saved)}/{len(target_ids)})")
        if len(saved) == len(target_ids):
            break

print(f"\nDone. {len(saved)} images saved to data/test_images/")

missing = target_ids - set(saved.keys())
if missing:
    print(f"Missing: {missing}")