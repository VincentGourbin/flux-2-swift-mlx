# Lion vs AdamW Optimizer Comparison

Comparison of **Lion** (EvoLved Sign Momentum) vs **AdamW** optimizers on the Cat Toy LoRA training dataset using Klein 4B.

## Test Setup

| Parameter | AdamW | Lion |
|-----------|-------|------|
| **Model** | Klein 4B | Klein 4B |
| **Dataset** | Cat Toy (6 images) | Cat Toy (6 images) |
| **Steps** | 500 | 500 |
| **Rank** | 32 | 32 |
| **Learning Rate** | 1e-4 | 3e-5 (auto) |
| **Weight Decay** | 0.0001 | 0.1 (auto) |
| **Beta2** | 0.999 | 0.99 (auto) |
| **Timestep** | balanced | balanced |
| **DOP** | yes (cat) | yes (cat) |
| **Checkpoints** | every 100 steps | every 100 steps |
| **Validation** | 3 prompts at each checkpoint | 3 prompts at each checkpoint |
| **Seed** | 42 | 42 |

Lion uses smart defaults (auto-applied when not explicitly set in YAML):
- Learning rate 3-10x smaller than AdamW
- Weight decay 3-10x larger
- Beta2 closer to 1 (0.99 vs 0.999)

## How to Reproduce

```bash
# Build CLI
xcodebuild -scheme Flux2CLI -configuration Release -destination 'platform=macOS' build

# Run the full comparison (trains both, evaluates with VLM)
bash docs/examples/lion-vs-adamw/run_comparison.sh
```

## Results

> **TODO**: Run comparison and fill in results

### Loss Curves

| Step | AdamW Loss | Lion Loss |
|-----:|:----------:|:---------:|
| 100 | | |
| 200 | | |
| 300 | | |
| 400 | | |
| 500 | | |

### VLM Evaluation (Scene/Style scores at checkpoints)

Evaluated by comparing each checkpoint's validation image against the reference (`examples/cat-toy/train/6.jpeg`) using the Qwen3.5 VLM.

| Step | AdamW Scene | AdamW Style | Lion Scene | Lion Style |
|-----:|:-----------:|:-----------:|:----------:|:----------:|
| 100 | | | | |
| 200 | | | | |
| 300 | | | | |
| 400 | | | | |
| 500 | | | | |

### Validation Images

> **TODO**: Add side-by-side validation images at key checkpoints

### Training Time

| Metric | AdamW | Lion |
|--------|-------|------|
| Total time | | |
| Time per step | | |
| Peak memory | | |

## Analysis

> **TODO**: Fill after running comparison

Key questions to answer:
1. Does Lion converge faster or slower than AdamW?
2. Does Lion produce better or worse image quality at the same step count?
3. How do the loss curves differ?
4. Is the ~50% optimizer memory reduction noticeable in practice?
5. Do Lion's smart defaults (3e-5 LR, 0.1 WD) work well out of the box?

## Configs

- [AdamW config](cat_toy_adamw.yaml)
- [Lion config](cat_toy_lion.yaml)
- [Comparison script](run_comparison.sh)
