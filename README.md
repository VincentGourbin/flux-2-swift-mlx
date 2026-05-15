# Flux.2 Swift MLX

A native Swift implementation of [Flux.2](https://blackforestlabs.ai/) image generation models, running locally on Apple Silicon Macs using [MLX](https://github.com/ml-explore/mlx-swift).

> **Fork notice**: This is the `intrusive-memory` fork. **As of v3.0.0 it is library-only** — the `Flux2CLI` and `FluxEncodersCLI` executables were removed and the project is now consumed exclusively via Swift Package Manager. The original upstream [VincentGourbin/flux-2-swift-mlx](https://github.com/VincentGourbin/flux-2-swift-mlx) still publishes pre-built CLI/app binaries if you need them.

## Features

### Image Generation (Flux2Core)
- **Native Swift**: Pure Swift implementation, no Python dependencies at runtime
- **MLX Acceleration**: Optimized for Apple Silicon (M1/M2/M3/M4) using MLX
- **Multiple Models**: Dev (32B), Klein 4B, and Klein 9B variants
- **Quantized Models**: On-the-fly quantization (qint8/int4) for all models — Dev fits in ~17GB at int4
- **Text-to-Image**: Generate images from text prompts
- **Image-to-Image**: Transform images with text prompts and configurable strength
- **Multi-Image Conditioning**: Combine elements from up to 3 reference images
- **Prompt Upsampling**: Enhance prompts with Mistral/Qwen3 before generation
- **LoRA Support**: Load and apply LoRA adapters for style transfer
- **LoRA Training**: Train your own LoRAs on Apple Silicon ([guide](docs/examples/TRAINING_GUIDE.md))
- **Image-to-Image Training**: Train paired I2I LoRAs (e.g. style transfer, image restoration)
- **macOS App**: Demo SwiftUI application (`Flux2App`) with T2I, I2I, and chat

### Text Encoders (FluxTextEncoders)
- **Mistral Small 3.2 (24B)**: Text encoder for FLUX.2 dev/pro
- **Qwen3 (4B/8B)**: Text encoder for FLUX.2 Klein
- **Text Generation**: Streaming text generation with configurable parameters
- **Interactive Chat**: Multi-turn conversation with chat template support
- **Vision Analysis**: Image understanding via Pixtral vision encoder (VLM)
- **FLUX.2 Embeddings**: Extract embeddings compatible with FLUX.2 image generation

## Requirements

- macOS 26.0 (Tahoe) or later
- Apple Silicon Mac (M1/M2/M3/M4) — MLX has no x86_64 path
- Xcode 16 or later, Swift 6.2+

**Memory requirements by model (with on-the-fly quantization):**

| Model | int4 | qint8 | bf16 |
|-------|------|-------|------|
| Klein 4B | 16 GB | 16 GB | 24 GB |
| Klein 9B | 16 GB | 24 GB | 32 GB |
| Dev (32B) | 32 GB | 96 GB | 96 GB |

## Installation

### As a Swift Package (recommended for library use)

Add to your `Package.swift`:

```swift
dependencies: [
    .package(url: "https://github.com/intrusive-memory/flux-2-swift-mlx", .upToNextMajor(from: "3.0.0")),
],
targets: [
    .target(
        name: "YourTarget",
        dependencies: [
            .product(name: "Flux2Core", package: "flux-2-swift-mlx"),
            // or .product(name: "FluxTextEncoders", package: "flux-2-swift-mlx"),
        ]
    ),
]
```

Available products: `Flux2Core` (image generation), `FluxTextEncoders` (text encoders + VLM).

### Build from Source

```bash
git clone https://github.com/intrusive-memory/flux-2-swift-mlx.git
cd flux-2-swift-mlx
```

Build with Xcode (not `swift build` — see [CLAUDE.md](CLAUDE.md) / [GEMINI.md](GEMINI.md) for the build conventions used in this repository):

1. Open the project in Xcode
2. Select the `Flux2App` scheme (or build the `Flux2Core` / `FluxTextEncoders` library targets directly)
3. Build with `Cmd+B` (or `Cmd+R` to run the demo app)

### Pre-built Binaries

This fork does **not** publish CLI or app binaries. If you need a signed CLI/app bundle, the upstream [VincentGourbin/flux-2-swift-mlx releases](https://github.com/VincentGourbin/flux-2-swift-mlx/releases/latest) page has them.

### Download Models

As of v3.0.0, models are fetched through the [SwiftAcervo](https://github.com/intrusive-memory/SwiftAcervo) CDN (Cloudflare R2) rather than directly from HuggingFace. This is a breaking change from v2.x, which used a hand-rolled HuggingFace client. Note that not every model is provisioned on the CDN yet — unprovisioned models throw `notProvisionedOnCDN` rather than falling back to HuggingFace.

**For Dev (32B):**
- Text Encoder: Mistral Small 3.2 (~25GB 8-bit)
- Transformer: Flux.2 Dev (~33GB qint8, ~17GB int4)
- VAE: Flux.2 VAE (~3GB)

**For Klein 4B/9B:**
- Text Encoder: Qwen3-4B or Qwen3-8B (~4-8GB 8-bit)
- Transformer: Klein 4B (~4-7GB) or Klein 9B (~5-17GB depending on quantization)
- VAE: Flux.2 VAE (~3GB)

Models are cached in `~/Library/Caches/models/` by default (configurable via `--models-dir` or `ModelRegistry.customModelsDirectory` for sandboxed apps).

## App Group configuration (required)

This package depends on [SwiftAcervo](https://github.com/intrusive-memory/SwiftAcervo) for shared model storage. SwiftAcervo v0.10.0 resolves its App Group ID in this order: `ACERVO_APP_GROUP_ID` env var → `com.apple.security.application-groups` entitlement (macOS only) → `fatalError`. There is **no silent fallback**.

- **Signed UI apps (macOS / iOS)**: declare `com.apple.security.application-groups` with `group.intrusive-memory.models` in your `.entitlements` file. iOS apps additionally need `ACERVO_APP_GROUP_ID=group.intrusive-memory.models` in the launch environment.
- **CLI tools, scripts, CI jobs, test runners**: export `ACERVO_APP_GROUP_ID=group.intrusive-memory.models` in the shell or job environment. The standard place is `~/.zprofile`:

    ```sh
    export ACERVO_APP_GROUP_ID=group.intrusive-memory.models
    ```

Without this, `Acervo.sharedModelsDirectory` traps with `fatalError`. See [SwiftAcervo's USAGE.md](https://github.com/intrusive-memory/SwiftAcervo/blob/main/USAGE.md) for full details.

## Usage

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

## Telemetry

`Flux2Core` emits typed `Flux2TelemetryEvent` events covering all major pipeline phases: `pipelineInit`, `pipelineDispose`, `weightLoadComplete` (one per component), `textEncodeComplete`, `schedulerConfigured`, `denoiseLoopStart`, `denoiseLoopEnd`, `vaeDecodeComplete`, `numericalAnomaly`, `generationCancelled`, and `errorThrown` — 11 cases in total. Subscribe once at process startup using the process-wide seam:

```swift
import Flux2Core

// 1. Implement the reporter protocol (Sendable, single async method):
struct PrintingReporter: Flux2TelemetryReporter {
    func capture(_ event: Flux2TelemetryEvent) async {
        print("[flux2] \(event)")
    }
}

// 2. Install at startup (thread-safe, lock-guarded):
Flux2Telemetry.setReporter(PrintingReporter())

// 3. Remove at shutdown:
Flux2Telemetry.setReporter(nil)
```

For test isolation, install a reporter on a specific pipeline instance via `Flux2Pipeline.setTelemetry(_:)` — the instance reporter takes precedence over the process-wide reporter when both are set. See `AGENTS.md §Telemetry` for contributor guidance on adding new event cases.

## Architecture

Flux.2 Dev is a ~32B parameter rectified flow transformer:

- **8 Double-stream blocks**: Joint attention between text and image
- **48 Single-stream blocks**: Combined text+image processing
- **4D RoPE**: Rotary position embeddings for T, H, W, L axes
- **SwiGLU FFN**: Gated activation in feed-forward layers
- **AdaLN**: Adaptive layer normalization with timestep conditioning

Text encoding uses [Mistral Small 3.2](https://github.com/VincentGourbin/mistral-small-3.2-swift-mlx) to generate 15360-dim embeddings.

## On-the-fly Quantization

All models support on-the-fly quantization to reduce transformer memory. No need to download separate variants — one bf16 model file serves all levels.

| Model | bf16 | qint8 (-47%) | int4 (-72%) |
|-------|------|-------------|-------------|
| Klein 4B | 7.4 GB | 3.9 GB | 2.1 GB |
| Klein 9B | 17.3 GB | 9.2 GB | 4.9 GB |
| Dev (32B) | 61.5 GB | 32.7 GB | 17.3 GB |

Configure quantization on `Flux2Config` (or `Flux2Pipeline.init` overrides) before initializing the pipeline. See [Quantization Benchmark](docs/examples/quantization-benchmark/) for measurements and visual comparison.

## Documentation

### Guides

| Guide | Description |
|-------|-------------|
| [LoRA Guide](docs/LoRA.md) | Loading and using LoRA adapters |
| [LoRA Training Guide](docs/examples/TRAINING_GUIDE.md) | Training parameters, DOP, gradient checkpointing, YAML config |
| [Text Encoders](docs/TextEncoders.md) | FluxTextEncoders library API |
| [Custom Model Integration](docs/CustomModelIntegration.md) | Integrating custom MLX-compatible models into the framework |
| [Flux2App Guide](docs/Flux2App.md) | Demo macOS application |

### For AI Agents Working on This Repo

| File | Audience |
|---|---|
| [AGENTS.md](AGENTS.md) | Universal — read first |
| [CLAUDE.md](CLAUDE.md) | Claude Code (XcodeBuildMCP, build conventions) |
| [GEMINI.md](GEMINI.md) | Gemini (standard `xcodebuild`) |
| [TESTING_REQUIREMENTS.md](TESTING_REQUIREMENTS.md) | Authoritative testing standard |

### Examples and Benchmarks

| Example | Description |
|---------|-------------|
| [Examples Gallery](docs/examples/) | Overview of all examples with sample outputs |
| [Model Comparison](docs/examples/comparison.md) | Dev vs Klein 4B vs Klein 9B — performance, quality, when to use each |
| [Quantization Benchmark](docs/examples/quantization-benchmark/) | Measured memory, speed, and visual quality for bf16/qint8/int4 |
| [Flux.2 Dev Examples](docs/examples/flux2-dev/) | T2I, I2I, multi-image conditioning, VLM image interpretation |
| [Flux.2 Klein 4B Examples](docs/examples/flux2-klein-4b/) | Fast T2I, multiple resolutions, quantization comparison |
| [Flux.2 Klein 9B Examples](docs/examples/flux2-klein-9b/) | T2I, multiple resolutions, prompt upsampling |

### LoRA Training Examples

| Example | Model | Description |
|---------|-------|-------------|
| [Cat Toy (Subject LoRA)](examples/cat-toy/) | Klein 4B | Subject injection with DOP, trigger word `sks` |
| [Tarot Style (Style LoRA)](docs/examples/tarot-style-lora/) | Klein 4B | Style transfer, trigger word `rwaite`, 32 training images |

## Current Limitations

- **Dev Performance**: Generation takes ~30 min for 1024x1024 images (use Klein for faster results)
- **Dev Memory**: Requires 32GB+ with int4, 64GB+ with qint8 (Klein 4B works with 16GB)
- **LoRA Training**: Supported on Klein 4B, Klein 9B, and Dev. Enable `gradient_checkpointing: true` for larger models to reduce memory by ~50%. Image-to-Image training doubles sequence length — gradient checkpointing is recommended.

## Acknowledgments

### Open-source dependencies

| Dependency | License |
|---|---|
| [mlx-swift](https://github.com/ml-explore/mlx-swift) | MIT |
| [swift-argument-parser](https://github.com/apple/swift-argument-parser) | Apache-2.0 |
| [swift-tokenizers](https://github.com/DePasqualeOrg/swift-tokenizers) | Apache-2.0 |
| [SwiftAcervo](https://github.com/intrusive-memory/SwiftAcervo) | MIT |
| [universal](https://github.com/marcprux/universal) | Apache-2.0 |
| [Black Forest Labs](https://blackforestlabs.ai/) | — (Flux.2 model architecture) |
| [Vincent Gourbin](https://github.com/VincentGourbin/flux-2-swift-mlx) | MIT (original upstream Swift implementation) |
| [Hugging Face Diffusers](https://github.com/huggingface/diffusers) | Apache-2.0 (reference implementation) |

### Model weights

Model weights for the variants distributed via this project's Acervo CDN were originally published on HuggingFace by Mistral AI, the Black Forest Labs team, the LM Studio community, and individual contributors (`VincentGOURBIN`, `aydin99`). Intrusive Memory mirrors these weights via Cloudflare R2 for distribution to Flux2Swift users.

| Acervo model ID | HuggingFace origin | License |
|---|---|---|
| `lmstudio-community/Qwen3-4B-MLX-8bit` | [huggingface.co/lmstudio-community/Qwen3-4B-MLX-8bit](https://huggingface.co/lmstudio-community/Qwen3-4B-MLX-8bit) | Apache-2.0 |
| `lmstudio-community/Qwen3-4B-MLX-4bit` | [huggingface.co/lmstudio-community/Qwen3-4B-MLX-4bit](https://huggingface.co/lmstudio-community/Qwen3-4B-MLX-4bit) | Apache-2.0 |
| `lmstudio-community/Qwen3-8B-MLX-8bit` | [huggingface.co/lmstudio-community/Qwen3-8B-MLX-8bit](https://huggingface.co/lmstudio-community/Qwen3-8B-MLX-8bit) | Apache-2.0 |
| `lmstudio-community/Qwen3-8B-MLX-4bit` | [huggingface.co/lmstudio-community/Qwen3-8B-MLX-4bit](https://huggingface.co/lmstudio-community/Qwen3-8B-MLX-4bit) | Apache-2.0 |
| `lmstudio-community/Mistral-Small-3.2-24B-Instruct-2506-MLX-8bit` | [huggingface.co/lmstudio-community/Mistral-Small-3.2-24B-Instruct-2506-MLX-8bit](https://huggingface.co/lmstudio-community/Mistral-Small-3.2-24B-Instruct-2506-MLX-8bit) | Apache-2.0 |
| `lmstudio-community/Mistral-Small-3.2-24B-Instruct-2506-MLX-6bit` | [huggingface.co/lmstudio-community/Mistral-Small-3.2-24B-Instruct-2506-MLX-6bit](https://huggingface.co/lmstudio-community/Mistral-Small-3.2-24B-Instruct-2506-MLX-6bit) | Apache-2.0 |
| `lmstudio-community/Mistral-Small-3.2-24B-Instruct-2506-MLX-4bit` | [huggingface.co/lmstudio-community/Mistral-Small-3.2-24B-Instruct-2506-MLX-4bit](https://huggingface.co/lmstudio-community/Mistral-Small-3.2-24B-Instruct-2506-MLX-4bit) | Apache-2.0 |
| `aydin99/FLUX.2-klein-4B-int8` | [huggingface.co/aydin99/FLUX.2-klein-4B-int8](https://huggingface.co/aydin99/FLUX.2-klein-4B-int8) | Apache-2.0 (same as BFL Klein) |
| `black-forest-labs/FLUX.2-klein-4B` | [huggingface.co/black-forest-labs/FLUX.2-klein-4B](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B) | Apache-2.0 |
| `black-forest-labs/FLUX.2-klein-9B` | [huggingface.co/black-forest-labs/FLUX.2-klein-9B](https://huggingface.co/black-forest-labs/FLUX.2-klein-9B) | Gated — FLUX.2 Non-Commercial License |
| `VincentGOURBIN/flux_qint_8bit` | [huggingface.co/VincentGOURBIN/flux_qint_8bit](https://huggingface.co/VincentGOURBIN/flux_qint_8bit) | Gated — FLUX.2 Non-Commercial License (derived from Dev) |

## License

MIT License - see [LICENSE](LICENSE) file.

---

**Disclaimer**: This is an independent implementation and is not affiliated with Black Forest Labs. Flux.2 model weights are subject to their own license terms.
