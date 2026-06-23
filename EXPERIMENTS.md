\# Experiments



\## Phase 1 — Data Collection



\- Source: `ashraq/fashion-product-images-small` (HuggingFace, streamed)

\- Filter: t-shirts only (subCategory == "Tshirts")

\- Total collected: 700 images

\- Weak labels via title parsing (regex on productDisplayName)

\- Target attributes: color, pattern (sleeve/neck dropped — insufficient coverage)

\- Classes kept: Printed, Striped, Solid (Melange/Abstract dropped — too few samples)



| Split          | Count |

|----------------|-------|

| labeled\_train  | 180   |

| labeled\_test   | 47    |

| unlabeled\_pool | 469   |



Color coverage: 100% | Pattern coverage: 100%



\## Phase 2 — Zero-Shot Baseline



\- Model: Qwen2.5-VL-3B-Instruct (4-bit, BitsAndBytes)

\- Prompt: constrained two-line format (Color / Pattern)

\- Parse failures: 0/227

\- OOV predictions: 0/227



| Metric      | Value |

|-------------|-------|

| Color Acc   | 66.1% |

| Pattern Acc | 53.7% |

| Color F1    | 0.514 |

| Pattern F1  | 0.562 |

| Avg Conf    | 0.840 |



Failure analysis: color errors are semantically close (Black→Charcoal, Yellow→Mustard).

Pattern failures heavily biased toward Solid, missing Printed — classic zero-shot majority-class confusion.



\## Phase 3 — LoRA Fine-Tuning (180 samples)



\- Framework: unsloth + TRL SFTTrainer

\- Base: Qwen2.5-VL-3B-Instruct (4-bit)

\- LoRA: r=16, alpha=16, vision encoder frozen

\- Trainable params: 29.9M / 3.78B (0.79%)

\- Epochs: 3 | LR: 2e-4 | Batch: 4 (grad accum)

\- Hardware: Colab T4 (15GB)

\- Training time: \~7 minutes



Loss curve: 1.97 → 0.01 (plateaus at step 25 — overfitting on 180 samples, expected)



| Metric      | Zero-Shot | LoRA-180 | Delta    |

|-------------|-----------|----------|----------|

| Color Acc   | 66.1%     | 66.0%    | \~0       |

| Pattern Acc | 53.7%     | 89.4%    | +35.7%   |

| Color F1    | 0.514     | 0.439    | -0.075   |

| Pattern F1  | 0.562     | 0.822    | +26.0%   |



Pattern improved massively. Color flat — 21 classes need more data.



\## Phase 4 — Pseudo-Label Self-Training



\- Model: LoRA-180 run on 469 unlabeled images

\- Confidence threshold: 0.95 (selected from distribution analysis)

\- Accepted: 255/469 (54.4%)

\- Rejected: 214/469 (45.6%)

\- Expanded dataset: 180 + 255 = 435 samples

\- Retrain: 2 epochs, LR=1e-4 (lower to avoid overwriting LoRA-180 weights)

\- Training loss: stable 0.003–0.008 throughout



| Metric      | LoRA-180 | LoRA-435 | Delta  |

|-------------|----------|----------|--------|

| Color Acc   | 66.0%    | 68.1%    | +2.1%  |

| Pattern Acc | 89.4%    | 89.4%    | 0      |

| Color F1    | 0.439    | 0.449    | +0.010 |

| Pattern F1  | 0.822    | 0.822    | 0      |



Pseudo-labeling gave marginal color gains. Pattern saturated at ceiling for 47-sample test set.

