# LoRA Adapters for Flux.2

LoRA (Low-Rank Adaptation) allows fine-tuning models for specific tasks without modifying the base weights. This enables specialized capabilities like object removal, spritesheet generation, style transfer, and more.

## Quick Start

```bash
flux2 i2i "your prompt" \
  --images input.jpg \
  --lora path/to/lora.safetensors \
  --lora-scale 1.0 \
  --model klein-4b \
  -o output.png
```

## CLI Options

| Option | Default | Description |
|--------|---------|-------------|
| `--lora` | none | Path to LoRA safetensors file |
| `--lora-scale` | `1.0` | LoRA scale factor (typically 0.5-1.5) |

## How It Works

LoRA weights are merged into the transformer at load time:

```
new_weight = original_weight + scale × (loraB @ loraA)
```

This approach:
- Preserves the original model weights
- Allows adjusting influence via `--lora-scale`
- Adds minimal overhead (~76MB for rank-16 LoRAs)

## Compatibility

**Important:** LoRA files must match the model architecture:

| Model | Compatible LoRAs |
|-------|-----------------|
| `klein-4b` | Klein 4B LoRAs (3072 inner dim) |
| `klein-9b` | Klein 9B LoRAs (4096 inner dim) |
| `dev` | Dev LoRAs (6144 inner dim) |

Using a LoRA with the wrong model will result in a shape mismatch error.

## Recommended Settings

For best results with LoRA:

```bash
--transformer-quant bf16  # Full precision recommended
--lora-scale 1.0-1.1      # Adjust based on LoRA
```

---

## Examples

### Object Removal

**LoRA:** [fal/flux-2-klein-4B-object-remove-lora](https://huggingface.co/fal/flux-2-klein-4B-object-remove-lora)

Removes highlighted/masked objects from images.

| Input | Output |
|-------|--------|
| ![Input](examples/lora_object_removal/input.jpg) | ![Output](examples/lora_object_removal/output.png) |

**Command:**
```bash
flux2 i2i "Remove the highlighted object from the scene" \
  --images input.jpg \
  --lora flux-object-remove-lora.safetensors \
  --lora-scale 1.1 \
  --model klein-4b \
  --transformer-quant bf16 \
  --strength 0.8 \
  -s 4 \
  -o output.png
```

**Notes:**
- Requires exact prompt: "Remove the highlighted object from the scene"
- Objects should be highlighted/masked in the input image
- Recommended scale: 1.1

---

### Spritesheet Generation

**LoRA:** [fal/flux-2-klein-4b-spritesheet-lora](https://huggingface.co/fal/flux-2-klein-4b-spritesheet-lora)

Generates 2×2 sprite sheets with multiple views of an object.

| Input | Output |
|-------|--------|
| ![Input](examples/lora_spritesheet/input.png) | ![Output](examples/lora_spritesheet/output.png) |

**Command:**
```bash
flux2 i2i "2x2 sprite sheet" \
  --images object.png \
  --lora flux-spritesheet-lora.safetensors \
  --lora-scale 1.1 \
  --model klein-4b \
  --transformer-quant bf16 \
  --strength 0.8 \
  -s 4 \
  -o spritesheet.png
```

**Output views:**
- **Top-Left**: Isometric view (↘)
- **Top-Right**: Isometric view (↙)
- **Bottom-Left**: Side profile (←)
- **Bottom-Right**: Top-down view (↑)

**Notes:**
- Requires prompt: "2x2 sprite sheet"
- Recommended scale: 1.1
- Works best with isolated objects on clean backgrounds

---

## Finding LoRAs

### Recommended Sources

- **[HuggingFace](https://huggingface.co/models?search=flux-2-klein)** - Official and community LoRAs
- **[fal.ai LoRAs](https://huggingface.co/fal)** - High-quality Klein 4B LoRAs
- **[Civitai](https://civitai.com/)** - Community LoRAs (check compatibility)

### Known Klein 4B LoRAs

| LoRA | Purpose | Prompt | Scale |
|------|---------|--------|-------|
| [object-remove](https://huggingface.co/fal/flux-2-klein-4B-object-remove-lora) | Remove objects | "Remove the highlighted object from the scene" | 1.1 |
| [spritesheet](https://huggingface.co/fal/flux-2-klein-4b-spritesheet-lora) | 2×2 sprite grid | "2x2 sprite sheet" | 1.1 |

---

## Troubleshooting

### Shape Mismatch Error

```
MLX error: Shapes (6144,15360) and (3072,7680) cannot be broadcast
```

**Cause:** LoRA is for a different model (e.g., Dev LoRA with Klein 4B model).

**Solution:** Use `--model` matching the LoRA's target architecture.

### Desaturated/Washed Out Colors

**Cause:** dtype mismatch between LoRA weights and model weights.

**Solution:** This was fixed in commit e9de5b5. Update to latest version.

### LoRA Has No Effect

**Possible causes:**
1. Wrong prompt (some LoRAs require specific activation keywords)
2. Scale too low (try increasing `--lora-scale`)
3. Incompatible LoRA format

---

## Technical Details

### Supported Format

- **File format:** SafeTensors (`.safetensors`)
- **Weight naming:** `*.lora_A.weight`, `*.lora_B.weight`
- **Typical rank:** 16 (configurable by LoRA creator)

### Memory Usage

| Rank | Parameters | Memory |
|------|------------|--------|
| 8 | ~10M | ~38 MB |
| 16 | ~20M | ~76 MB |
| 32 | ~40M | ~152 MB |

### Layer Coverage

Klein 4B LoRAs typically cover:
- 88 layers (44 pairs of loraA/loraB)
- Single transformer blocks: `attn.to_out`, `attn.to_qkv_mlp_proj`
- Double transformer blocks: attention projections
