# Flux.2 Swift MLX

> **⚠️ WORK IN PROGRESS**
>
> This project is under active development. APIs may change, and some features are not yet implemented.

A native Swift implementation of [Flux.2 Dev](https://blackforestlabs.ai/) image generation model, running locally on Apple Silicon Macs using [MLX](https://github.com/ml-explore/mlx-swift).

## Features

- **Native Swift**: Pure Swift implementation, no Python dependencies at runtime
- **MLX Acceleration**: Optimized for Apple Silicon (M1/M2/M3/M4) using MLX
- **Multiple Models**: Dev (32B), Klein 4B, and Klein 9B variants
- **Quantized Models**: Support for 8-bit quantized transformer (~32GB VRAM for Dev)
- **Text-to-Image**: Generate images from text prompts
- **Image-to-Image**: Transform images with text prompts and configurable strength
- **Multi-Image Conditioning**: Combine elements from up to 3 reference images
- **Prompt Upsampling**: Enhance prompts with Mistral/Qwen3 before generation
- **CLI Tool**: Full-featured command-line interface for image generation

## Requirements

- macOS 14.0 (Sonoma) or later
- Apple Silicon Mac (M1/M2/M3/M4)
- Xcode 15.0 or later

**Memory requirements by model:**
| Model | Minimum RAM | Recommended |
|-------|-------------|-------------|
| Klein 4B | 16GB | 32GB+ |
| Klein 9B | 32GB | 48GB+ |
| Dev (32B) | 64GB | 96GB+ |

## Installation

### Build from Source

```bash
git clone https://github.com/VincentGourbin/flux-2-swift-mlx.git
cd flux-2-swift-mlx
```

**Important**: Build with Xcode, not `swift build`:

1. Open the project in Xcode
2. Select the `Flux2CLI` scheme
3. Build with `Cmd+B`
4. Find the binary in Products folder

### Download Models

The models are downloaded automatically from HuggingFace on first run.

**For Dev (32B):**
- Text Encoder: Mistral Small 3.2 (~25GB 8-bit)
- Transformer: Flux.2 Dev qint8 (~32GB)
- VAE: Flux.2 VAE (~3GB)

**For Klein 4B/9B:**
- Text Encoder: Qwen3-4B or Qwen3-8B (~4-8GB 8-bit)
- Transformer: Klein 4B (~8GB) or Klein 9B (~18GB)
- VAE: Flux.2 VAE (~3GB)

Models are cached in `~/Library/Caches/models/`.

## Usage

### CLI

```bash
# Fast generation with Klein 4B (~26s, commercial OK)
flux2 t2i "a beaver building a dam" --model klein-4b

# Better quality with Klein 9B (~62s)
flux2 t2i "a beaver building a dam" --model klein-9b

# Maximum quality with Dev (~35min, requires 64GB+ RAM)
flux2 t2i "a beautiful sunset over mountains" --model dev

# With custom parameters
flux2 t2i "a red apple on a white table" \
  --width 512 \
  --height 512 \
  --steps 20 \
  --guidance 4.0 \
  --seed 42 \
  --output apple.png

# Image-to-Image with reference image
flux2 i2i "transform into a watercolor painting" \
  --images photo.jpg \
  --strength 0.7 \
  --steps 28 \
  --output watercolor.png

# Multi-image conditioning (combine elements)
flux2 i2i "a cat wearing this jacket" \
  --images cat.jpg \
  --images jacket.jpg \
  --steps 28 \
  --output cat_jacket.png
```

See [CLI Documentation](docs/CLI.md) for all options.

### As a Library

```swift
import Flux2Core

// Initialize pipeline
let pipeline = try await Flux2Pipeline()

// Generate image
let image = try await pipeline.generateTextToImage(
    prompt: "a beautiful sunset over mountains",
    height: 512,
    width: 512,
    steps: 20,
    guidance: 4.0
) { current, total in
    print("Step \(current)/\(total)")
}
```

## Architecture

Flux.2 Dev is a ~32B parameter rectified flow transformer:

- **8 Double-stream blocks**: Joint attention between text and image
- **48 Single-stream blocks**: Combined text+image processing
- **4D RoPE**: Rotary position embeddings for T, H, W, L axes
- **SwiGLU FFN**: Gated activation in feed-forward layers
- **AdaLN**: Adaptive layer normalization with timestep conditioning

Text encoding uses [Mistral Small 3.2](https://github.com/VincentGourbin/mistral-small-3.2-swift-mlx) to generate 15360-dim embeddings.

## Current Limitations

- **Dev Performance**: Generation takes ~35 min for 1024x1024 images (use Klein for faster results)
- **Dev Memory**: Requires 64GB+ unified memory (Klein 4B works with 16GB)
- **No LoRA support**: Adapter loading not yet available
- **Klein 9B**: Only bf16 available (no quantized variants yet)

## Roadmap

See [GitHub Issues](https://github.com/VincentGourbin/flux-2-swift-mlx/issues) for planned features:

- [x] Text-to-Image generation
- [x] Image-to-Image support (single image with strength)
- [x] Multi-image conditioning (up to 3 reference images)
- [x] Prompt upsampling
- [x] Flux.2 Klein 4B (4B, ~26s, Apache 2.0)
- [x] Flux.2 Klein 9B (9B, ~62s, non-commercial)
- [ ] Performance optimizations
- [ ] Demo SwiftUI application
- [ ] LoRA adapter support

## Acknowledgments

- [Black Forest Labs](https://blackforestlabs.ai/) for Flux.2
- [Hugging Face Diffusers](https://github.com/huggingface/diffusers) for reference implementation
- [MLX](https://github.com/ml-explore/mlx) team at Apple for the ML framework

## License

MIT License - see [LICENSE](LICENSE) file.

---

**Disclaimer**: This is an independent implementation and is not affiliated with Black Forest Labs. Flux.2 model weights are subject to their own license terms.
