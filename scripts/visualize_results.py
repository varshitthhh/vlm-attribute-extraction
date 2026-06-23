import os
import json
from PIL import Image, ImageDraw, ImageFont

RESULTS_JSON = os.path.join(os.path.dirname(__file__), "..", "data", "lora_expanded_test_results.json")
TEST_JSON    = os.path.join(os.path.dirname(__file__), "..", "data", "labeled_test.json")
IMG_DIR      = os.path.join(os.path.dirname(__file__), "..", "data", "test_images")
OUT_DIR      = os.path.join(os.path.dirname(__file__), "..", "logs")

os.makedirs(OUT_DIR, exist_ok=True)

with open(RESULTS_JSON) as f:
    results = json.load(f)
with open(TEST_JSON) as f:
    test = json.load(f)

id_to_gt = {str(s["id"]): s["attributes"] for s in test}

THUMB   = 224
PAD     = 10
LABEL_H = 60
COLS    = 6
ROWS    = (len(results) + COLS - 1) // COLS

W = COLS * (THUMB + PAD) + PAD
H = ROWS * (THUMB + LABEL_H + PAD) + PAD

canvas = Image.new("RGB", (W, H), (245, 245, 245))
draw   = ImageDraw.Draw(canvas)

try:
    font = ImageFont.truetype("arial.ttf", 13)
    font_small = ImageFont.truetype("arial.ttf", 11)
except:
    font = ImageFont.load_default()
    font_small = font

correct_color   = 0
correct_pattern = 0

for idx, r in enumerate(results):
    row = idx // COLS
    col = idx % COLS
    x   = PAD + col * (THUMB + PAD)
    y   = PAD + row * (THUMB + LABEL_H + PAD)

    img_path = os.path.join(IMG_DIR, f"{r['id']}.jpg")
    img = Image.open(img_path).convert("RGB").resize((THUMB, THUMB))
    canvas.paste(img, (x, y))

    gt      = id_to_gt[str(r["id"])]
    p_color = r["pred_color"]   or "?"
    p_pat   = r["pred_pattern"] or "?"
    g_color = gt["color"]
    g_pat   = gt["pattern"]

    color_ok   = p_color == g_color
    pattern_ok = p_pat   == g_pat
    if color_ok:   correct_color   += 1
    if pattern_ok: correct_pattern += 1

    # Border: green = both correct, orange = partial, red = both wrong
    if color_ok and pattern_ok:
        border_col = (34, 197, 94)
    elif color_ok or pattern_ok:
        border_col = (251, 146, 60)
    else:
        border_col = (239, 68, 68)

    draw.rectangle([x-2, y-2, x+THUMB+1, y+THUMB+1], outline=border_col, width=3)

    # Labels below image
    c_mark = "✓" if color_ok   else "✗"
    p_mark = "✓" if pattern_ok else "✗"
    draw.text((x, y+THUMB+2),  f"C:{c_mark} GT={g_color}",  font=font_small, fill=(30,30,30))
    draw.text((x, y+THUMB+16), f"   Pr={p_color}",           font=font_small,
              fill=(34,139,34) if color_ok else (200,0,0))
    draw.text((x, y+THUMB+32), f"P:{p_mark} GT={g_pat}",    font=font_small, fill=(30,30,30))
    draw.text((x, y+THUMB+46), f"   Pr={p_pat}",             font=font_small,
              fill=(34,139,34) if pattern_ok else (200,0,0))

# Summary header
summary = (f"LoRA-435 | Color Acc: {correct_color}/{len(results)} "
           f"({correct_color/len(results)*100:.1f}%)  |  "
           f"Pattern Acc: {correct_pattern}/{len(results)} "
           f"({correct_pattern/len(results)*100:.1f}%)")
print(summary)

out_path = os.path.join(OUT_DIR, "visual_proof.png")
canvas.save(out_path)
print(f"Saved: {out_path}")