# Generation Examples

This folder contains example images generated with Flux.2 Swift MLX.

## Cat on Beach - Standard Generation

**Prompt:** `"a cat wearing sunglasses, sitting on a sunny beach"`

**Parameters:**
- Size: 1024x1024
- Steps: 28
- Guidance: 4.0
- Seed: random
- Prompt upsampling: disabled

### Progression

| Step 7 | Step 14 | Step 21 | Final (Step 28) |
|--------|---------|---------|-----------------|
| ![Step 7](cat_beach_standard/step_007.png) | ![Step 14](cat_beach_standard/step_014.png) | ![Step 21](cat_beach_standard/step_021.png) | ![Final](cat_beach_standard/final.png) |

### Performance Report

```
╔══════════════════════════════════════════════════════════════╗
║                  FLUX.2 PERFORMANCE REPORT                   ║
╠══════════════════════════════════════════════════════════════╣
📊 PHASE TIMINGS:
────────────────────────────────────────────────────────────────
  1. Load Text Encoder                4.08s    0.2%
  2. Text Encoding                    2.41s    0.1%
  3. Unload Text Encoder            113.7ms    0.0%
  4. Load Transformer                23.63s    1.1%
  5. Load VAE                        80.9ms    0.0%
  6. Denoising Loop               34m 19.2s   98.5% ███████████████████
  7. VAE Decode                       1.94s    0.1%
  8. Post-processing                  1.5ms    0.0%
────────────────────────────────────────────────────────────────
  TOTAL                           34m 51.4s  100.0%

📈 DENOISING STEP STATISTICS:
────────────────────────────────────────────────────────────────
  Steps:              28
  Total denoising:    34m 19.2s
  Average per step:   1m 13.1s
  Fastest step:       52.02s
  Slowest step:       2m 49.3s
╚══════════════════════════════════════════════════════════════╝
```

---

## Cat on Beach - With Prompt Upsampling

**Original prompt:** `"a cat wearing sunglasses, sitting on a sunny beach"`

**Enhanced prompt (by Mistral):** *(generated automatically with `--upsample-prompt`)*

The prompt upsampling feature uses Mistral to enhance the original prompt with more visual details before encoding, potentially improving image quality and coherence.

**Parameters:**
- Size: 1024x1024
- Steps: 28
- Guidance: 4.0
- Seed: random
- Prompt upsampling: **enabled**

### Progression

| Step 7 | Step 14 | Step 21 | Final (Step 28) |
|--------|---------|---------|-----------------|
| ![Step 7](cat_beach_upsampled/step_007.png) | ![Step 14](cat_beach_upsampled/step_014.png) | ![Step 21](cat_beach_upsampled/step_021.png) | ![Final](cat_beach_upsampled/final.png) |

### Performance Report (with prompt upsampling)

```
╔══════════════════════════════════════════════════════════════╗
║                  FLUX.2 PERFORMANCE REPORT                   ║
╠══════════════════════════════════════════════════════════════╣
📊 PHASE TIMINGS:
────────────────────────────────────────────────────────────────
  1. Load Text Encoder                4.47s    0.2%
  2. Text Encoding                 2m 28.7s    7.7% █
  3. Unload Text Encoder            289.7ms    0.0%
  4. Load Transformer                30.30s    1.6%
  5. Load VAE                        83.1ms    0.0%
  6. Denoising Loop               29m 16.0s   90.4% ██████████████████
  7. VAE Decode                       1.86s    0.1%
  8. Post-processing                  1.5ms    0.0%
────────────────────────────────────────────────────────────────
  TOTAL                           32m 21.7s  100.0%

📈 DENOISING STEP STATISTICS:
────────────────────────────────────────────────────────────────
  Steps:              28
  Total denoising:    29m 7.4s
  Average per step:   1m 2.4s
  Fastest step:       54.11s
  Slowest step:       1m 48.4s

  📐 Estimated times for different step counts:
     10 steps: 10m 24.1s
     20 steps: 20m 48.2s
     28 steps: 29m 7.4s
     50 steps: 52m 0.4s

💡 INSIGHTS:
────────────────────────────────────────────────────────────────
  Bottleneck: 6. Denoising Loop (90.4% of total)
  Overhead (non-denoising): 3m 5.7s

╚══════════════════════════════════════════════════════════════╝
```

**Note:** Text encoding takes longer with prompt upsampling (~2.5 min vs ~2.4s) because Mistral generates an enhanced prompt before encoding.

---

## CLI Commands Used

```bash
# Standard generation
.build/release/Flux2CLI t2i "a cat wearing sunglasses, sitting on a sunny beach" \
  --width 1024 --height 1024 \
  --steps 28 --guidance 4.0 \
  --checkpoint 7 \
  --profile \
  --output cat_beach.png

# With prompt upsampling
.build/release/Flux2CLI t2i "a cat wearing sunglasses, sitting on a sunny beach" \
  --width 1024 --height 1024 \
  --steps 28 --guidance 4.0 \
  --upsample-prompt \
  --checkpoint 7 \
  --profile \
  --output cat_beach_upsampled.png
```

---

## Image-to-Image Examples

### I2I - Artistic Variation (Watercolor)

**Prompt:** `"transform into a beautiful watercolor painting with soft brushstrokes and vibrant colors"`

**Reference:** Cat on beach (single image)

**Parameters:**
- Size: 1024x1024
- Steps: 28 effective
- Strength: 0.7 (30% original preserved)
- Prompt upsampling: disabled

| Step 7 | Step 14 | Step 21 | Final (Step 28) |
|--------|---------|---------|-----------------|
| ![Step 7](i2i_artistic_variation/step_007.png) | ![Step 14](i2i_artistic_variation/step_014.png) | ![Step 21](i2i_artistic_variation/step_021.png) | ![Final](i2i_artistic_variation/final.png) |

**Command:**
```bash
flux2 i2i "transform into a beautiful watercolor painting with soft brushstrokes and vibrant colors" \
  --images cat_beach_upsampled.png \
  --strength 0.7 --steps 28 \
  --checkpoint 7 --profile \
  --output artistic_variation.png
```

> **Note:** The I2I mode transforms the input image by encoding it with VAE, adding noise based on strength, and denoising with the text prompt as guidance. Lower strength preserves more of the original image.

---

### I2I - Multi-Image: Cat + Jacket (2 images)

**Prompt:** `"Put the jacket from image 2 on the cat from image 1"`

**Reference Images:**
1. `cat_beach_upsampled/final.png` (1024x1024) - Cat with sunglasses on beach
2. `jacket.jpg` (1080x1620) - Yellow jacket

**Parameters:**
- Size: 1024x1024
- Steps: 28
- Strength: 1.0 (full conditioning mode)
- Guidance: 4.0

| Step 7 | Step 14 | Step 21 | Final (Step 28) |
|--------|---------|---------|-----------------|
| ![Step 7](i2i_cat_jacket/final_checkpoints/step_007.png) | ![Step 14](i2i_cat_jacket/final_checkpoints/step_014.png) | ![Step 21](i2i_cat_jacket/final_checkpoints/step_021.png) | ![Final](i2i_cat_jacket/final.png) |

**Performance Report:**
```
╔══════════════════════════════════════════════════════════════╗
║                  FLUX.2 PERFORMANCE REPORT                   ║
╠══════════════════════════════════════════════════════════════╣
📊 PHASE TIMINGS:
────────────────────────────────────────────────────────────────
  1. Load Text Encoder                4.10s    0.1%
  2. Text Encoding                    2.39s    0.0%
  3. Unload Text Encoder            141.3ms    0.0%
  4. Load Transformer                21.95s    0.4%
  5. Load VAE                        68.5ms    0.0%
  6. Denoising Loop                96m 3.2s   99.5% ███████████████████
  7. VAE Decode                       2.19s    0.0%
────────────────────────────────────────────────────────────────
  TOTAL                           96m 34.0s  100.0%

📈 DENOISING STEP STATISTICS:
────────────────────────────────────────────────────────────────
  Steps:              28
  Average per step:   3m 25.5s
  Fastest step:       2m 44.5s
  Slowest step:       4m 33.4s
╚══════════════════════════════════════════════════════════════╝
```

**Command:**
```bash
flux2 i2i "Put the jacket from image 2 on the cat from image 1" \
  --images cat_beach.png --images jacket.jpg \
  --width 1024 --height 1024 \
  --strength 1.0 --steps 28 \
  --checkpoint 7 --profile \
  --output cat_jacket.png
```

> **Note:** With `strength=1.0`, Flux.2 uses full conditioning mode where reference images provide visual context but the output is generated from random noise. The model extracts visual elements (colors, textures, objects) from each reference image and combines them according to the prompt.

---

### I2I - Multi-Image: Cat + Jacket + Hat (3 images)

**Prompt:** `"Put the jacket from image 2 on the cat from image 1 and put the hat from image 3 on the head of the cat"`

**Reference Images:**
1. Cat on beach (1024x1024)
2. Yellow jacket (1080x1620)
3. Rainbow cap (1193x1000)

**Parameters:**
- Size: 1024x1024
- Steps: 28
- Strength: 1.0 (full conditioning mode)
- Guidance: 4.0

| Step 7 | Step 14 | Step 21 | Final (Step 28) |
|--------|---------|---------|-----------------|
| ![Step 7](i2i_cat_jacket_hat/final_checkpoints/step_007.png) | ![Step 14](i2i_cat_jacket_hat/final_checkpoints/step_014.png) | ![Step 21](i2i_cat_jacket_hat/final_checkpoints/step_021.png) | ![Final](i2i_cat_jacket_hat/final.png) |

**Performance Report:**
```
╔══════════════════════════════════════════════════════════════╗
║                  FLUX.2 PERFORMANCE REPORT                   ║
╠══════════════════════════════════════════════════════════════╣
📊 PHASE TIMINGS:
────────────────────────────────────────────────────────────────
  1. Load Text Encoder                4.08s    0.0%
  2. Text Encoding                    2.38s    0.0%
  3. Unload Text Encoder            143.5ms    0.0%
  4. Load Transformer                26.43s    0.2%
  5. Load VAE                        72.4ms    0.0%
  6. Denoising Loop              176m 28.2s   99.7% ███████████████████
  7. VAE Decode                       1.93s    0.0%
  8. Post-processing                  1.4ms    0.0%
────────────────────────────────────────────────────────────────
  TOTAL                           177m 3.3s  100.0%

📈 DENOISING STEP STATISTICS:
────────────────────────────────────────────────────────────────
  Steps:              28
  Total denoising:    176m 16.2s
  Average per step:   6m 17.7s
  Fastest step:       4m 52.1s
  Slowest step:       7m 54.5s
╚══════════════════════════════════════════════════════════════╝
```

> **Note:** With 3 reference images, each denoising step takes significantly longer (~6m 17.7s) compared to 2 images (~3m 25.5s) due to increased context length in the transformer attention.

**Command:**
```bash
flux2 i2i "Put the jacket from image 2 on the cat from image 1 and put the hat from image 3 on the head of the cat" \
  --images cat_beach.png --images jacket.jpg --images hat.jpg \
  --width 1024 --height 1024 \
  --strength 1.0 --steps 28 \
  --checkpoint 7 --profile \
  --output cat_jacket_hat.png
```

> **Key insight:** When using multiple reference images, explicitly reference them in the prompt (e.g., "from image 1", "from image 2"). The model correctly identifies and transfers visual elements from each reference image without needing explicit color or style descriptions.

---

## Hardware

- **Machine:** MacBook Pro 14" (Nov 2023)
- **Chip:** Apple M3 Max
- **RAM:** 96 GB Unified Memory
- **macOS:** Tahoe 26.2
- **Quantization:** 8-bit text encoder + qint8 transformer (~60GB peak)
