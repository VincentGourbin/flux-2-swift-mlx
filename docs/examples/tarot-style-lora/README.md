# Tarot Style LoRA Training Example

This example demonstrates training a **style LoRA** using the classic Rider-Waite tarot card illustrations. The goal is to teach the model the distinctive vintage woodcut illustration style without memorizing specific card content.

## Dataset

**Source**: [multimodalart/tarot-dataset](https://huggingface.co/datasets/multimodalart/tarot-dataset)

| Split | Images | Purpose |
|-------|--------|---------|
| Train | 32 | Learning the style |
| Validation | 8 | Monitoring overfitting |

### Style Characteristics
- Vintage woodcut/linocut illustration style
- Limited color palette (yellow, red, green, blue, black)
- Strong black outlines
- Medieval/Renaissance imagery
- Distinctive hatching and cross-hatching

### Caption Strategy
Captions describe **what is visible** without mentioning:
- "tarot" or "card"
- The specific style (woodcut, vintage, etc.)
- Card names (The Fool, King of Cups, etc.)

**Example captions:**
```
rwaite, a king seated on a throne holding a golden goblet and scepter
rwaite, a blindfolded woman in white seated by the sea holding two crossed swords
rwaite, three women in flowing robes raising golden goblets in celebration
```

The trigger word `rwaite` (for Rider-Waite) activates the style.

## Training Configuration

```yaml
model:
  name: klein-4b
  quantization: int8

lora:
  rank: 32
  alpha: 32.0
  target_layers: attention

training:
  max_steps: 1000
  learning_rate: 1e-4
  batch_size: 1

loss:
  timestep_sampling: balanced
  weighting: bell_shaped

checkpoints:
  save_every: 250
```

## Running the Training

```bash
# From the repository root
flux2 train-lora --config examples/tarot-style/tarot_training.yaml
```

## Validation Strategy

Two validation prompts are generated at each checkpoint:
1. **With trigger**: `rwaite, a wizard holding a glowing staff...` - Should show tarot style
2. **Without trigger**: `a wizard holding a glowing staff...` - Should be normal style

This allows visual comparison of style transfer effectiveness.

## Expected Results

### Checkpoints

| Step | Loss | Notes |
|------|------|-------|
| 0 | ~1.17 | Initial loss (before training) |
| 250 | TBD | Style starting to emerge |
| 500 | TBD | Style should be recognizable |
| 750 | TBD | Style refined |
| 1000 | TBD | Final checkpoint |

### Visual Progress

#### Step 250
<!-- TODO: Add validation images after training -->
*With trigger* | *Without trigger*
:---:|:---:
![step_250_with](./results/step_250_with_trigger.png) | ![step_250_without](./results/step_250_without_trigger.png)

#### Step 500
*With trigger* | *Without trigger*
:---:|:---:
![step_500_with](./results/step_500_with_trigger.png) | ![step_500_without](./results/step_500_without_trigger.png)

#### Step 750
*With trigger* | *Without trigger*
:---:|:---:
![step_750_with](./results/step_750_with_trigger.png) | ![step_750_without](./results/step_750_without_trigger.png)

#### Step 1000 (Final)
*With trigger* | *Without trigger*
:---:|:---:
![step_1000_with](./results/step_1000_with_trigger.png) | ![step_1000_without](./results/step_1000_without_trigger.png)

### Learning Curve

![Learning Curve](./results/learning_curve.svg)

## Using the Trained LoRA

```bash
# Generate with the tarot style
flux2 generate \
  --prompt "rwaite, a cat sitting on a throne wearing a crown" \
  --lora ./output/tarot-lora/tarot_lora_step_1000.safetensors \
  --lora-scale 1.0 \
  --output tarot_cat.png

# Generate without style (baseline comparison)
flux2 generate \
  --prompt "a cat sitting on a throne wearing a crown" \
  --output normal_cat.png
```

## Tips for Style LoRAs

1. **Trigger word**: Use a unique, memorable trigger (e.g., `rwaite`, `artdeco`, `pixelart`)
2. **Captions**: Describe content, not style - let the LoRA learn the style implicitly
3. **Dataset size**: 30-50 images is often enough for style transfer
4. **Rank**: 32 is a good starting point for styles
5. **Loss weighting**: `bell_shaped` helps focus on the style-relevant timesteps
6. **Validation**: Always compare with/without trigger to verify style isolation

## Files

```
examples/tarot-style/
├── train/                    # 32 training images + captions
│   ├── tarot_fixed_0000_Layer-78.png
│   ├── tarot_fixed_0000_Layer-78.txt
│   └── ...
├── validation/               # 8 validation images + captions
│   ├── tarot_fixed_0032_Layer-46.png
│   ├── tarot_fixed_0032_Layer-46.txt
│   └── ...
└── tarot_training.yaml       # Training configuration

docs/examples/tarot-style-lora/
├── README.md                 # This file
└── results/                  # Training results (after running)
    ├── learning_curve.svg
    └── step_*_*.png
```

## Credits

- Dataset: [multimodalart](https://huggingface.co/multimodalart) on HuggingFace
- Original artwork: Rider-Waite Tarot (public domain, 1909)
