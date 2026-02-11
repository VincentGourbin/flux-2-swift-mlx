# Flux.2 CLI Documentation

The `flux2` command-line tool provides access to Flux.2 image generation on Mac with MLX.

## Commands

| Command | Description |
|---------|-------------|
| `t2i` | Text-to-Image generation (default) |
| `i2i` | Image-to-Image generation (up to 4 images for Klein, 6 for Dev) |
| `download` | Download required models |
| `info` | Show system and model information |

## Text-to-Image (t2i)

Generate images from text prompts.

### Usage

```bash
flux2 t2i <prompt> [options]
```

### Arguments

| Argument | Description |
|----------|-------------|
| `<prompt>` | Text prompt describing the image to generate |

### Options

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--output` | `-o` | `output.png` | Output file path |
| `--width` | `-w` | `1024` | Image width in pixels |
| `--height` | `-h` | `1024` | Image height in pixels |
| `--model` | | `dev` | Model variant: `dev` (32B), `klein-4b` (4B), `klein-9b` (9B) |
| `--steps` | `-s` | varies* | Number of inference steps |
| `--guidance` | `-g` | varies* | Guidance scale (CFG) |
| `--seed` | | random | Random seed for reproducibility |
| `--text-quant` | | `8bit` | Text encoder quantization: `bf16`, `8bit`, `6bit`, `4bit` |
| `--transformer-quant` | | `qint8` | Transformer quantization: `bf16`, `qint8` |
| `--upsample-prompt` | | | Enhance prompt with visual details before encoding |
| `--interpret` | | | Image to analyze with VLM and inject into prompt (all models) |
| `--checkpoint` | | | Save intermediate images every N steps |
| `--debug` | | | Enable verbose debug output |
| `--profile` | | | Enable performance profiling |

\* **Model-specific defaults:**
| Model | Steps | Guidance |
|-------|-------|----------|
| `dev` | 50 | 4.0 |
| `klein-4b` | **4** | **1.0** |
| `klein-9b` | **4** | **1.0** |

### Examples

**Basic generation:**
```bash
flux2 t2i "a beautiful sunset over mountains"
```

**Custom size and output:**
```bash
flux2 t2i "a red apple on a white table" \
  --width 512 \
  --height 512 \
  --output apple.png
```

**Reproducible generation with seed:**
```bash
flux2 t2i "cosmic nebula in deep space" \
  --seed 42 \
  --steps 30 \
  --output nebula.png
```

**Save checkpoints during generation:**
```bash
flux2 t2i "portrait of a robot" \
  --steps 20 \
  --checkpoint 5 \
  --output robot.png
# Saves: robot_checkpoints/step_005.png, step_010.png, step_015.png, step_020.png
```

**Memory-efficient generation:**
```bash
flux2 t2i "landscape painting" \
  --text-quant 4bit \
  --transformer-quant qint8 \
  --output landscape.png
```

### Klein 4B (Fast Generation)

Klein 4B is optimized for speed with only 4 steps and guidance=1.0. These defaults are applied automatically when using `--model klein-4b`.

**Basic Klein generation:**
```bash
flux2 t2i "a beaver building a dam" --model klein-4b
# Uses: steps=4, guidance=1.0 automatically
```

**Klein at higher resolution:**
```bash
flux2 t2i "a majestic eagle flying over mountains at sunset" \
  --model klein-4b \
  --width 1536 --height 1024 \
  -o eagle.png
```

**Klein at maximum resolution (2048×2048):**
```bash
flux2 t2i "a futuristic city with flying cars and neon lights" \
  --model klein-4b \
  --width 2048 --height 2048 \
  -o city.png
```

> **Note:** Klein 4B uses ~8GB VRAM in bf16 (vs ~62GB for Dev) and generates images in ~26s at 1024×1024 (vs ~30min for Dev).

### Klein 4B Quantization Options

Klein 4B supports both full precision (bf16) and quantized (qint8) modes:

| Quantization | Transformer Memory | Speed (4 steps) | Quality |
|--------------|-------------------|-----------------|---------|
| `bf16` | ~7.4GB | ~26s | Best |
| `qint8` | ~3.9GB | ~28s | Excellent |
| `int4` | ~2.1GB | ~30s | Very Good |

**Full precision (bf16):**
```bash
flux2 t2i "a beaver building a dam" \
  --model klein-4b \
  --transformer-quant bf16 \
  -o beaver_bf16.png
```

**Quantized (qint8) - default:**
```bash
flux2 t2i "a beaver building a dam" \
  --model klein-4b \
  --transformer-quant qint8 \
  -o beaver_qint8.png
```

> **Recommendation:** For most use cases, the default qint8 quantization provides excellent quality with lower memory usage. Use bf16 when maximum quality is required and memory is not constrained.

### Klein 9B (Quality/Speed Balance)

Klein 9B offers better quality than Klein 4B while remaining much faster than Dev. It uses the same 4-step distillation with guidance=1.0.

**Basic Klein 9B generation:**
```bash
flux2 t2i "a beaver building a dam" --model klein-9b
# Uses: steps=4, guidance=1.0 automatically
# Time: ~56s at 1024x1024
```

**Klein 9B is ideal for:**
- Better quality than Klein 4B without Dev's long generation time
- Non-commercial projects where quality matters
- When you have ~32GB+ RAM available

| Model | Time (1024x1024) | Transformer Memory (qint8) | Quality |
|-------|------------------|---------------------------|---------|
| Klein 4B | ~28s | ~3.9GB | Good |
| **Klein 9B** | **~60s** | **~9.2GB** | **Better** |
| Dev | ~30min | ~32.7GB | Best |

> **Note:** Klein 9B supports on-the-fly quantization to qint8 and int4 (no pre-quantized variant needed). The text encoder uses Qwen3-8B (8bit).

---

## Image-to-Image (i2i)

Transform or combine images using a text prompt. Flux.2 supports two modes:

### Two Modes of Operation

**1. Traditional I2I (Single Image + Strength < 1.0)**
- Encodes the input image, mixes with noise based on strength, then denoises
- Lower strength = more of the original image preserved
- Use for: style transfer, subtle modifications, preserving structure

**2. Multi-Image Conditioning (2-3 Images OR Strength = 1.0)**
- Reference images provide visual context for generation
- Output starts from random noise, references guide the transformer
- Full denoising (all steps, no timestep skip)
- Use for: combining elements from multiple images, inspired generation

### Usage

```bash
# Traditional I2I (single image with strength)
flux2 i2i <prompt> --images <reference_image> --strength 0.7 [options]

# Multi-image conditioning
flux2 i2i <prompt> --images <img1> --images <img2> [--images <img3>] [options]
```

### Arguments

| Argument | Description |
|----------|-------------|
| `<prompt>` | Text prompt describing the desired output |

### Options

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--images` | `-i` | required | Reference image to transform |
| `--output` | `-o` | `output.png` | Output file path |
| `--width` | `-w` | from image | Output width (default: first reference image) |
| `--height` | `-h` | from image | Output height (default: first reference image) |
| `--steps` | `-s` | `28` | Number of **effective** denoising steps |
| `--total-steps` | | false | Interpret `--steps` as total steps before strength reduction |
| `--strength` | | `0.8` | Denoising strength (0.0-1.0). Lower = preserve more original |
| `--guidance` | `-g` | `4.0` | Guidance scale |
| `--seed` | | random | Random seed |
| `--upsample-prompt` | | false | Enhance prompt with Mistral before encoding |
| `--checkpoint` | | | Save intermediate images every N steps |
| `--profile` | | false | Show detailed performance profiling |
| `--text-quant` | | `8bit` | Text encoder quantization |
| `--transformer-quant` | | `qint8` | Transformer quantization |

### Understanding Strength

The `--strength` parameter controls how much of the original image is preserved:

| Strength | Effect | Use Case |
|----------|--------|----------|
| `1.0` | Full denoising (ignores reference) | Maximum creativity |
| `0.8` | 80% new, 20% original | Default - good balance |
| `0.5` | 50/50 mix | Moderate changes |
| `0.3` | 30% new, 70% original | Subtle modifications |

### Understanding Steps

By default, `--steps` specifies **effective steps** (what you actually get):

```bash
flux2 i2i "prompt" --steps 28 --strength 0.7
# Output: "Steps: 28 effective (total: 40)"
```

Use `--total-steps` for the legacy behavior where strength reduces the step count:

```bash
flux2 i2i "prompt" --steps 28 --strength 0.7 --total-steps
# Output: "Steps: 19 effective (from 28 total)"
```

### Examples

**Style transfer:**
```bash
flux2 i2i "transform into a watercolor painting" \
  --images photo.jpg \
  --strength 0.7 \
  --steps 28 \
  --output watercolor.png
```

**With prompt upsampling:**
```bash
flux2 i2i "make it look like a cyberpunk scene" \
  --images original.jpg \
  --strength 0.6 \
  --upsample-prompt \
  --output cyberpunk.png
```

**Save progress checkpoints:**
```bash
flux2 i2i "artistic interpretation" \
  --images original.jpg \
  --strength 0.8 \
  --steps 20 \
  --checkpoint 5 \
  --profile \
  --output artistic.png
# Saves: artistic_checkpoints/step_005.png, step_010.png, etc.
```

**Preserve more of original (lower strength):**
```bash
flux2 i2i "add subtle vintage film grain effect" \
  --images photo.jpg \
  --strength 0.3 \
  --steps 28 \
  --output vintage.png
```

**Multi-image conditioning (combine elements):**
```bash
flux2 i2i "a cat wearing the jacket" \
  --images cat.jpg \
  --images jacket.jpg \
  --steps 28 \
  --output cat_with_jacket.png
```

**Multi-image conditioning (inspired by references):**
```bash
flux2 i2i "create a scene combining these elements" \
  --images landscape.jpg \
  --images character.jpg \
  --images style_reference.jpg \
  --steps 28 \
  --output combined.png
```

> **Note:** Multi-image mode ignores the `--strength` parameter and always performs full denoising. Reference images provide visual context that guides the transformer's attention during generation.

### I2I with Klein Models

Klein 4B and 9B support Image-to-Image generation with the same VAE-based encoding as Dev.

**Klein I2I example:**
```bash
flux2 i2i "transform into watercolor style" \
  --model klein-4b \
  --images photo.jpg \
  --strength 0.7 \
  -o watercolor.png
```

**Multi-image with Klein:**
```bash
flux2 i2i "a cat wearing this hat" \
  --model klein-9b \
  --images cat.jpg \
  --images hat.jpg \
  --steps 4 \
  -o cat_hat.png
```

#### Token Limits per Model

Reference images consume tokens in the transformer's attention. Here are practical limits:

| Model | VRAM | Recommended Max Images | Max Tokens |
|-------|------|------------------------|------------|
| Klein 4B | ~8GB | 2-3 images | ~16k |
| Klein 9B | ~20GB | 3-5 images | ~25k |
| Dev | ~60GB | 5-10 images | ~45k |

**Token calculation:** Each 1024×1024 reference image = ~4,096 tokens

#### Upsampling Behavior

| Model | T2I Upsampling | I2I Upsampling |
|-------|----------------|----------------|
| **Dev** | Mistral VLM (text) | Mistral VLM (sees images) ✅ |
| **Klein** | Qwen3 (text only) | **Mistral VLM (sees images)** ✅ |

> **Note:** For Klein I2I with `--upsample-prompt`, the pipeline automatically loads Mistral VLM temporarily to analyze reference images, then unloads it and uses Qwen3 for the final text encoding. This matches the official Flux.2 implementation and provides context-aware upsampling.

---

## Download Models

Download required models from HuggingFace.

### Usage

```bash
flux2 download [options]
```

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `--hf-token` | `$HF_TOKEN` | HuggingFace token for gated models |
| `--transformer-quant` | `qint8` | Which transformer variant to download |
| `--all` | false | Download all model variants |
| `--vae-only` | false | Only download VAE |

### Examples

**Download default models (qint8 transformer + VAE):**
```bash
flux2 download
```

**Download with HuggingFace token:**
```bash
flux2 download --hf-token hf_xxxxxxxxxxxxx
# Or set environment variable:
export HF_TOKEN=hf_xxxxxxxxxxxxx
flux2 download
```

**Download specific variant:**
```bash
flux2 download --transformer-quant bf16
```

---

## LoRA Adapters

Apply LoRA (Low-Rank Adaptation) weights to customize model behavior for specific tasks.

| Option | Default | Description |
|--------|---------|-------------|
| `--lora` | none | Path to LoRA safetensors file |
| `--lora-scale` | `1.0` | LoRA scale factor (typically 0.5-1.5) |
| `--lora-config` | none | JSON config file (for advanced LoRAs with scheduler overrides) |

**Quick example:**
```bash
flux2 i2i "2x2 sprite sheet" \
  --images object.png \
  --lora flux-spritesheet-lora.safetensors \
  --lora-scale 1.1 \
  --model klein-4b \
  -o spritesheet.png
```

**With JSON config (for Turbo LoRAs):**
```bash
flux2 t2i "a mountain landscape" \
  --lora-config turbo-lora.json \
  --model dev \
  -o output.png
```

For complete LoRA documentation, examples, and troubleshooting, see **[LoRA.md](LoRA.md)**.

---

## System Information

Show system information and model status.

### Usage

```bash
flux2 info
```

### Output

```
Flux.2 Swift MLX Framework
Version: 0.1.0

System Information:
  RAM: 64GB
  Recommended config: Balanced (~60GB)

Available Quantization Presets:
  High Quality (~90GB): bf16 text + bf16 transformer
  Balanced (~57GB): 8bit text + qint8 transformer
  Memory Efficient (~47GB): 4bit text + qint8 transformer
  Minimal (~47GB): 4bit text + qint8 transformer
  Ultra-Minimal (~30GB): 4bit text + int4 transformer

Model Status:
  [✓] Flux.2 Transformer (qint8)
  [✗] Flux.2 Transformer (bf16)
  [✓] Mistral Small 3.2 (8bit)
  [✓] Flux.2 VAE
```

---

## Quantization Guide

### Text Encoder (Mistral Small 3.2)

| Option | Memory | Quality |
|--------|--------|---------|
| `bf16` | ~48GB | Best |
| `8bit` | ~25GB | Excellent |
| `6bit` | ~19GB | Very Good |
| `4bit` | ~14GB | Good |

### Transformer (Dev 32B)

| Option | Transformer Memory | Quality |
|--------|-------------------|---------|
| `bf16` | ~61.5GB | Best (requires 96GB+ RAM) |
| `qint8` | ~32.7GB | Excellent (recommended) |
| `int4` | ~17.3GB | Very Good (32GB+ Macs) |

### Recommended Configurations

| Config | Text | Transformer | Estimated Memory | Use Case |
|--------|------|-------------|-----------------|----------|
| High Quality | bf16 | bf16 | ~90GB | Maximum quality (96GB+ RAM) |
| **Balanced** | 8bit | qint8 | ~57GB | **Recommended** |
| Memory Efficient | 4bit | qint8 | ~47GB | 64GB Macs |
| Minimal | 4bit | qint8 | ~47GB | 48GB Macs |
| Ultra-Minimal | 4bit | int4 | ~30GB | 32GB Macs |

---

## Tips

### Performance

- **Smaller images are faster**: Start with 256×256 or 512×512 for testing
- **Fewer steps**: 20-30 steps often produces good results
- **Use checkpoints**: Add `--checkpoint 5` to monitor progress

### Reproducibility

- Use `--seed` with the same value to reproduce results
- Note: Different quantization levels may produce slightly different outputs

### Troubleshooting

**"Missing models" error:**
```bash
flux2 download  # Download required models first
```

**Out of memory:**
```bash
# Use more aggressive quantization
flux2 t2i "prompt" --text-quant 4bit --transformer-quant qint8
```

**Slow generation:**
- This is expected. Current performance: ~20min for 256×256
- Performance optimization is planned for future releases
