# Flux.2 Swift MLX

> **⚠️ WORK IN PROGRESS**
>
> This project is under active development. APIs may change, and some features are not yet implemented.

A native Swift implementation of [Flux.2 Dev](https://blackforestlabs.ai/) image generation model, running locally on Apple Silicon Macs using [MLX](https://github.com/ml-explore/mlx-swift).

## Features

- **Native Swift**: Pure Swift implementation, no Python dependencies at runtime
- **MLX Acceleration**: Optimized for Apple Silicon (M1/M2/M3/M4) using MLX
- **Quantized Models**: Support for 8-bit quantized transformer (~32GB VRAM)
- **Text-to-Image**: Generate images from text prompts
- **CLI Tool**: Simple command-line interface for image generation

## Requirements

- macOS 14.0 (Sonoma) or later
- Apple Silicon Mac (M1/M2/M3/M4)
- **Minimum 64GB unified memory** (for transformer + text encoder)
- Xcode 15.0 or later

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

The models are downloaded automatically from HuggingFace on first run. You'll need:

- **Text Encoder**: Mistral Small 3.2 (~25GB 8-bit)
- **Transformer**: Flux.2 Dev qint8 (~32GB)
- **VAE**: Flux.2 VAE (~3GB)

Models are cached in `~/.cache/huggingface/`.

## Usage

### CLI

```bash
# Basic text-to-image generation
flux2 t2i "a beautiful sunset over mountains" --output sunset.png

# With custom parameters
flux2 t2i "a red apple on a white table" \
  --width 512 \
  --height 512 \
  --steps 20 \
  --guidance 4.0 \
  --seed 42 \
  --output apple.png

# Save intermediate checkpoints
flux2 t2i "cosmic nebula in deep space" \
  --steps 30 \
  --checkpoint 5 \
  --output nebula.png
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

- **Performance**: Generation is slow (~20 min for 256×256, ~1.7h for 512×512)
- **Memory**: Requires 64GB+ unified memory
- **Text-to-Image only**: Image-to-Image not yet implemented
- **No LoRA support**: Adapter loading not yet available

## Roadmap

See [GitHub Issues](https://github.com/VincentGourbin/flux-2-swift-mlx/issues) for planned features:

- [ ] Performance optimizations
- [ ] Demo SwiftUI application
- [ ] Image-to-Image support
- [ ] LoRA adapter support
- [ ] Flux.2 Klein (smaller model)

## Acknowledgments

- [Black Forest Labs](https://blackforestlabs.ai/) for Flux.2
- [Hugging Face Diffusers](https://github.com/huggingface/diffusers) for reference implementation
- [MLX](https://github.com/ml-explore/mlx) team at Apple for the ML framework

## License

MIT License - see [LICENSE](LICENSE) file.

---

**Disclaimer**: This is an independent implementation and is not affiliated with Black Forest Labs. Flux.2 model weights are subject to their own license terms.
