# Flux.2 Model Comparison: Dev vs Klein 4B

A side-by-side comparison of Flux.2 Dev (32B) and Flux.2 Klein 4B models.

## Model Specifications

| Feature | Flux.2 Dev | Flux.2 Klein 4B |
|---------|------------|-----------------|
| **Parameters** | 32B | 4B |
| **Text Encoder** | Mistral Small 3.2 | Qwen3-4B |
| **Default Steps** | 50 | 4 (distilled) |
| **Default Guidance** | 4.0 | 1.0 |
| **VRAM Usage** | ~60GB | ~5-8GB |
| **License** | Non-commercial | Apache 2.0 |
| **1024×1024 Time** | ~35 min | ~26s |

---

## When to Use Each Model

### Choose Flux.2 Dev (32B) when:
- **Maximum quality** is required
- Generating images for **professional** or **artistic** purposes
- You need **fine-grained control** over the generation process
- Memory is not a constraint (~64GB+ RAM recommended)
- You can wait for longer generation times

### Choose Flux.2 Klein 4B when:
- **Speed** is a priority (~80x faster than Dev)
- You need **commercial use** (Apache 2.0 license)
- Working with **limited memory** (~8GB sufficient)
- **Iterating quickly** on prompts
- Generating **many images** in a batch

---

## Performance Comparison

| Metric | Flux.2 Dev | Klein 4B (bf16) | Klein 4B (qint8) |
|--------|------------|-----------------|------------------|
| **Total Time (1024²)** | ~35 min | ~26s | ~27s |
| **Steps** | 28-50 | 4 | 4 |
| **Per-Step Time** | ~1 min | ~5.5s | ~5.8s |
| **Memory Usage** | ~60GB | ~5.6GB | ~3.8GB |
| **Speedup vs Dev** | 1x | **~80x** | **~78x** |

---

## Quality Comparison

While Klein 4B is significantly faster, Flux.2 Dev generally produces:
- More **detailed textures**
- Better **coherent compositions**
- More accurate **prompt following** for complex prompts
- Better handling of **multiple subjects**
- Superior **fine details** (faces, hands, text)

Klein 4B excels at:
- **Quick iterations** and concept exploration
- **Simple to medium complexity** prompts
- **Commercial projects** requiring Apache 2.0 license
- **Memory-constrained** environments

---

## Quantization Options

### Flux.2 Dev

| Quantization | Memory | Quality |
|--------------|--------|---------|
| bf16 | ~64GB | Best |
| qint8 | ~32GB | Excellent (recommended) |
| qint4 | ~16GB | Good (experimental) |

### Flux.2 Klein 4B

| Quantization | Memory | Speed | Quality |
|--------------|--------|-------|---------|
| bf16 | ~5.6GB | ~26s | Best |
| qint8 | ~3.8GB | ~27s | Excellent |

---

## CLI Examples

### Same Prompt, Different Models

**Prompt:** `"a cat wearing sunglasses, sitting on a sunny beach"`

#### Flux.2 Dev
```bash
flux2 t2i "a cat wearing sunglasses, sitting on a sunny beach" \
  --model dev \
  --steps 28 \
  -o cat_dev.png
# Time: ~20-35 min, Memory: ~60GB
```

#### Klein 4B
```bash
flux2 t2i "a cat wearing sunglasses, sitting on a sunny beach" \
  --model klein-4b \
  -o cat_klein.png
# Time: ~26s, Memory: ~5GB
```

---

## Recommended Workflows

### Iterative Development
1. **Explore** with Klein 4B (fast iterations)
2. **Refine** prompt based on results
3. **Final render** with Dev for maximum quality

### Production (Commercial)
- Use **Klein 4B** (Apache 2.0 license allows commercial use)
- Dev is non-commercial only

### Production (Non-Commercial)
- Use **Dev** for hero images
- Use **Klein 4B** for variations and exploration

---

## Hardware Requirements

| Model | Minimum RAM | Recommended RAM |
|-------|-------------|-----------------|
| Flux.2 Dev (qint8) | 64GB | 96GB+ |
| Klein 4B (qint8) | 16GB | 32GB+ |
| Klein 4B (bf16) | 16GB | 32GB+ |

---

## See Also

- [Flux.2 Dev Examples](flux2-dev/README.md)
- [Flux.2 Klein 4B Examples](flux2-klein-4b/README.md)
- [CLI Documentation](../CLI.md)
