# Flux.2 CLI Documentation

> **⚠️ WORK IN PROGRESS** - Some features may not be fully implemented yet.

The `flux2` command-line tool provides access to Flux.2 image generation on Mac with MLX.

## Commands

| Command | Description |
|---------|-------------|
| `t2i` | Text-to-Image generation (default) |
| `i2i` | Image-to-Image generation *(not yet implemented)* |
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
| `--steps` | `-s` | `50` | Number of inference steps |
| `--guidance` | `-g` | `4.0` | Guidance scale (CFG) |
| `--seed` | | random | Random seed for reproducibility |
| `--text-quant` | | `8bit` | Text encoder quantization: `bf16`, `8bit`, `6bit`, `4bit` |
| `--transformer-quant` | | `qint8` | Transformer quantization: `bf16`, `qint8`, `qint4` |
| `--checkpoint` | | | Save intermediate images every N steps |
| `--debug` | | | Enable verbose debug output |

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

---

## Image-to-Image (i2i)

> ⚠️ **Not yet implemented** - This feature is planned for a future release.

Generate images using reference images as guidance.

### Usage

```bash
flux2 i2i <prompt> --images <image1> [image2] [image3] [options]
```

### Arguments

| Argument | Description |
|----------|-------------|
| `<prompt>` | Text prompt describing the desired output |

### Options

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--images` | `-i` | required | Reference image(s), 1-3 images |
| `--output` | `-o` | `output.png` | Output file path |
| `--steps` | `-s` | `50` | Number of inference steps |
| `--guidance` | `-g` | `4.0` | Guidance scale |
| `--seed` | | random | Random seed |
| `--text-quant` | | `8bit` | Text encoder quantization |
| `--transformer-quant` | | `qint8` | Transformer quantization |

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
  Balanced (~60GB): 8bit text + qint8 transformer
  Memory Efficient (~50GB): 4bit text + qint8 transformer
  Minimal (~35GB): 4bit text + qint4 transformer

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

### Transformer

| Option | Memory | Quality |
|--------|--------|---------|
| `bf16` | ~64GB | Best |
| `qint8` | ~32GB | Excellent (recommended) |
| `qint4` | ~16GB | Good (experimental) |

### Recommended Configurations

| Config | Text | Transformer | Total Memory | Use Case |
|--------|------|-------------|--------------|----------|
| High Quality | bf16 | bf16 | ~90GB | Maximum quality |
| **Balanced** | 8bit | qint8 | ~60GB | **Recommended** |
| Memory Efficient | 4bit | qint8 | ~50GB | 64GB Macs |
| Minimal | 4bit | qint4 | ~35GB | Testing only |

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
