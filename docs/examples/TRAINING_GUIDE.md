# Flux.2 LoRA Training Guide

This guide covers the key parameters for LoRA training with Flux.2 models.

## Model Comparison

| Model | Parameters | VRAM (8-bit) | Training Speed | Quality |
|-------|-----------|--------------|----------------|---------|
| Klein 4B | 4B | ~8 GB | Fast | Good for quick iterations |
| Klein 9B | 9B | ~18 GB | Medium | Better quality |
| Dev | 32B | ~50-70 GB | Slow | Best quality |

## When to Use Each Model

- **Klein 4B**: Prototyping, quick tests, limited VRAM
- **Klein 9B**: Balance of quality and speed, recommended for most use cases
- **Dev**: Production quality, when you have the hardware

---

## DOP (Differential Output Preservation)

DOP is a regularization technique that prevents the LoRA from affecting outputs when the trigger word is NOT present.

### When to Use DOP

| LoRA Type | DOP Recommended | Reason |
|-----------|-----------------|--------|
| **Subject/Character** (e.g., cat-toy) | Yes | You want the subject to appear ONLY with trigger |
| **Style** (e.g., tarot) | No | You WANT the style to affect everything |
| **Concept** (e.g., pose, action) | Maybe | Depends on use case |

### DOP Configuration

```yaml
loss:
  diff_output_preservation: true  # Enable DOP
  diff_output_preservation_class: "cat"  # Replace trigger with this word
  diff_output_preservation_multiplier: 1.0  # Strength (1.0 = equal to main loss)
  diff_output_preservation_every_n_steps: 1  # Performance optimization
```

### DOP Performance Optimization

DOP requires **3 forward passes per step** instead of 1:
1. Main forward+backward (with LoRA)
2. Base forward (LoRA disabled)
3. DOP forward+backward (with LoRA)

For larger models, this significantly increases training time.

**Recommended `diff_output_preservation_every_n_steps`:**

| Model | Recommended | Effect |
|-------|-------------|--------|
| Klein 4B | 1-2 | Full DOP, minimal overhead |
| Klein 9B | 4 | ~4x speedup, good regularization |
| Dev | 4-8 | ~4-8x speedup, still effective |

---

## Timestep Sampling

Controls which noise levels are sampled during training.

| Mode | Description | Best For |
|------|-------------|----------|
| `uniform` | Equal probability for all timesteps | General training |
| `content` | Bias toward low noise (content focus) | Subject LoRAs |
| `style` | Bias toward high noise (style focus) | Style LoRAs |
| `balanced` | 50/50 mix of content and style | Recommended default |

```yaml
loss:
  timestep_sampling: balanced
```

---

## Loss Weighting

Controls how much each timestep contributes to the loss.

| Mode | Description | Best For |
|------|-------------|----------|
| `uniform` | Equal weight for all timesteps | General training |
| `bell_shaped` | Focus on medium noise levels | Recommended |

```yaml
loss:
  weighting: bell_shaped
```

---

## LoRA Configuration

### Rank

| Rank | Memory | Capacity | Recommended For |
|------|--------|----------|-----------------|
| 8 | Low | Limited | Simple styles |
| 16 | Medium | Good | Most use cases |
| 32 | Higher | High | Complex subjects/styles |
| 64 | High | Very high | Very detailed training |

### Target Layers

| Target | Layers Trained | Memory | Effect |
|--------|----------------|--------|--------|
| `attention` | Q, K, V, O projections | Lower | Core style/content |
| `all` | Attention + FFN | Higher | More expressive |

```yaml
lora:
  rank: 32
  alpha: 32.0  # Usually same as rank
  target_layers: attention  # or 'all'
```

---

## Learning Rate

Recommended starting points:

| Model | Learning Rate |
|-------|--------------|
| Klein 4B | 1e-4 |
| Klein 9B | 1e-4 |
| Dev | 1e-4 |

All Flux models use the same learning rate (Ostris standard).
Higher learning rates train faster but risk overfitting.

---

## Example Configurations

### Style LoRA (Tarot) - No DOP

```yaml
model:
  name: klein-4b

loss:
  timestep_sampling: balanced
  weighting: bell_shaped
  diff_output_preservation: false  # Style = affects everything

training:
  max_steps: 500
  learning_rate: 1e-4
```

### Subject LoRA (Cat Toy) - With DOP

```yaml
model:
  name: klein-4b

loss:
  timestep_sampling: balanced
  weighting: bell_shaped
  diff_output_preservation: true
  diff_output_preservation_class: "cat"
  diff_output_preservation_multiplier: 1.0

training:
  max_steps: 250  # Sufficient for subject LoRAs
  learning_rate: 1e-4
```

**Real Results (cat-toy on Klein 4B):**
- Training time: ~75 minutes (250 steps)
- Best loss: 0.24 at step 244
- DOP verified: trigger shows statue, no-trigger shows real cat

See `docs/examples/cat-toy-results/` for learning curve and progression images.

### Subject LoRA on Klein 9B - Optimized DOP

```yaml
model:
  name: klein-9b

loss:
  timestep_sampling: balanced
  weighting: bell_shaped
  diff_output_preservation: true
  diff_output_preservation_class: "cat"
  diff_output_preservation_multiplier: 1.0
  diff_output_preservation_every_n_steps: 4  # Optimization for larger model

training:
  max_steps: 250
  learning_rate: 1e-4
```

### Subject LoRA on Dev - Optimized DOP

```yaml
model:
  name: dev
  quantization: int8  # Mistral 24B text encoder

lora:
  rank: 32
  alpha: 32.0
  target_layers: attention  # Memory-efficient for Dev

loss:
  timestep_sampling: balanced
  weighting: bell_shaped
  diff_output_preservation: true
  diff_output_preservation_class: "cat"
  diff_output_preservation_multiplier: 1.0
  diff_output_preservation_every_n_steps: 8  # Dev is 32B, use 8 for best perf

training:
  max_steps: 250
  learning_rate: 1e-4  # Standard Ostris recommendation
```

**Note:** Dev requires ~60GB VRAM. Use `target_layers: attention` (not `all`) to save memory.

---

## Troubleshooting

### Training is very slow with DOP on Klein 9B/Dev

Use `diff_output_preservation_every_n_steps` to reduce DOP overhead:
- Klein 9B: `4` (~4x speedup)
- Dev: `8` (~8x speedup)

### Style LoRA affects images even without trigger word

This is expected for style LoRAs. If you want trigger-controlled style, you're actually training a "concept" LoRA - consider using DOP.

### Subject LoRA appears even without trigger word

Increase `diff_output_preservation_multiplier` (try 1.5 or 2.0) or ensure your captions properly include the trigger word.

### Loss is not decreasing

- Check learning rate (try 5e-5 if using 1e-4)
- Verify dataset captions match your training goals
- Ensure trigger word is in captions

### Out of memory

- Use smaller batch size (1)
- Use smaller model (Klein 4B)
- Reduce rank (16 instead of 32)
- For Dev: use `target_layers: attention` instead of `all`
- For Dev: limit resolutions to 512 only
- Disable DOP for larger models (Klein 9B, Dev)

> **Note:** `gradient_checkpointing` config option exists but is **not yet implemented**. It requires layer-wise checkpointing in the transformer model.

---

## Training Output Structure

A complete training run produces:

```
output/cat-toy-lora/
├── baseline/                    # Images BEFORE training (no LoRA)
├── checkpoint_000125/           # Checkpoint at step 125
│   ├── lora.safetensors         # LoRA weights
│   ├── optimizer_state.safetensors
│   ├── training_state.json
│   └── prompt_*_512x512.png     # Validation images
├── checkpoint_000250/           # Final checkpoint
├── .latent_cache/               # Cached VAE latents (reused on resume)
├── learning_curve.svg           # Loss visualization
└── lora_final.safetensors       # Final LoRA weights
```

## Example Results (Klein 4B, 250 steps)

```
docs/examples/cat-toy-results/
├── learning_curve.svg           # Loss progression
├── training_state.json          # Training metrics
└── progression/
    ├── step_000_trigger.png     # Baseline WITH trigger
    ├── step_000_notrigger.png   # Baseline WITHOUT trigger
    ├── step_125_trigger.png     # Step 125 WITH trigger
    ├── step_125_notrigger.png   # Step 125 WITHOUT trigger
    ├── step_250_trigger.png     # Final WITH trigger (cat-toy statue)
    └── step_250_notrigger.png   # Final WITHOUT trigger (real cat = DOP works)
```

## Training Control (File-based)

Control training via files in the output directory:

```bash
# Pause training (saves checkpoint, waits)
touch output/cat-toy-lora/.pause
rm output/cat-toy-lora/.pause     # Resume

# Request immediate checkpoint + validation images
touch output/cat-toy-lora/.checkpoint

# Stop training gracefully (saves final checkpoint)
touch output/cat-toy-lora/.stop
```
