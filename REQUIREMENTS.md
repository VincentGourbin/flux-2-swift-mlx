# flux-2-swift-mlx — Requirements

**Status**: DRAFT — debate and refine before implementation.
**Parent project**: [`PROJECT_PIPELINE.md`](../PROJECT_PIPELINE.md) — Unified MLX Inference Architecture (§4. flux-2-swift-mlx, Wave 4)
**Scope**: FLUX.2 model plugin for SwiftTubería. Provides the FLUX.2 DiT backbone, model-specific text encoders (Qwen3, Mistral), FLUX VAE, and LoRA target declarations. Shared infrastructure (weight loading, memory management, image rendering, pipeline orchestration) migrates to SwiftTubería.

---

## Motivation

flux-2-swift-mlx is a working, production-tested library. This requirements document describes how it evolves from a standalone pipeline into a SwiftTubería model plugin — preserving all current functionality while shedding ~80% of its infrastructure code to the shared pipeline.

The migration is not a rewrite. The backbone (FLUX DiT), text encoders (Qwen3, Mistral), and FLUX VAE remain here. What leaves is the generic plumbing: safetensors loading, quantization, memory management, progress reporting, and the pipeline orchestration loop.

### What This Package Provides vs What SwiftTubería Provides

| Concern | This Package | SwiftTubería |
|---|---|---|
| FLUX DiT backbone (double + single stream) | **Yes** — the unique architecture | — |
| Qwen3 text encoder (Klein models) | **Yes** — model-specific encoder | — |
| Mistral text encoder (Dev model) | **Yes** — model-specific encoder | — |
| FLUX VAE decoder | **Yes** — model-specific decoder | — |
| LoRA target layer declarations | **Yes** | — |
| Weight key mappings | **Yes** | — |
| Model configuration (Klein 4B, Klein 9B, Dev) | **Yes** | — |
| Pipeline recipe | **Yes** | — |
| Flow-match Euler scheduler | — | **Yes** (catalog: `FlowMatchEulerScheduler`) |
| Image rendering (MLXArray → CGImage) | — | **Yes** (catalog: `ImageRenderer`) |
| Weight loading + quantization | — | **Yes** (infrastructure: `WeightLoader`) |
| Model downloading + caching | — | **Yes** (via SwiftAcervo Component Registry) |
| Memory management + two-phase loading | — | **Yes** (infrastructure: `MemoryManager`) |
| Progress reporting | — | **Yes** (infrastructure: `PipelineProgress`) |
| LoRA loading/applying/scaling mechanics | — | **Yes** (infrastructure: `LoRALoader`) |

---

## F1. Package Structure

### F1.1 Products

```swift
.library(name: "Flux2Backbone", targets: ["Flux2Backbone"]),
.library(name: "Flux2Encoders", targets: ["Flux2Encoders"]),
.executable(name: "Flux2CLI", targets: ["Flux2CLI"]),
```

- **`Flux2Backbone`** — FLUX DiT transformer, FLUX VAE, configuration, weight key mapping, pipeline recipes. Depends on `Flux2Encoders`.
- **`Flux2Encoders`** — Qwen3 and Mistral text encoder implementations conforming to SwiftTubería's `TextEncoder` protocol. Separate library to support two-phase loading (encode → unload encoder → load backbone).
- **`Flux2CLI`** — Standalone generation and LoRA training tool.

**Naming alignment**: Current code uses `Flux2Core`/`FluxTextEncoders`; these will be renamed to `Flux2Backbone`/`Flux2Encoders` to match this spec during Wave 4 (FLUX migration). Avoids breaking existing consumers during the transition.

### F1.2 Dependencies

```swift
.package(url: "<SwiftTubería>", from: "0.1.0"),
.package(url: "https://github.com/apple/swift-argument-parser", from: "1.3.0"),
```

### F1.3 Platforms

```swift
platforms: [.macOS(.v26)]
```

FLUX.2 requires 16+ GB minimum memory — **macOS only**. No iPad target. Even Klein 4B at int4 requires ~16 GB, which exceeds any current iPad's memory. Adding iOS to Package.swift would create false expectations and untestable code paths. PixArt is the iPad engine; FLUX is the macOS engine. Revisit if Apple ships a 16+ GB iPad.

---

## F2. FLUX DiT Backbone

The backbone conforms to SwiftTubería's `Backbone` protocol.

### F2.1 Architecture Summary

FLUX.2 uses a hybrid transformer with two phases:

1. **Double-stream blocks** — Image tokens and text tokens each have their own self-attention and FFN, with cross-information flow via concatenated attention.
2. **Single-stream blocks** — Image and text tokens are concatenated and processed together through standard transformer blocks.

### F2.2 Model Variants

| Variant | Parameters | Encoder | Quantization | Min Memory |
|---|---|---|---|---|
| Klein 4B | 4B | Qwen3-4B | int4 (ultraMinimal) | 16 GB |
| Klein 9B | 9B | Qwen3-8B | qint8 (balanced) | 24 GB |
| Dev | 32B | Mistral | varies | 64 GB |

Each variant produces a distinct pipeline recipe (different encoder, different quantization config, same backbone architecture at different scales).

### F2.3 Backbone Protocol Conformance

```
inlet:  BackboneInput {
            latents:          MLXArray [B, H, W, 16]     ← FLUX VAE latent space (16 channels)
            conditioning:     MLXArray [B, seq, dim]      ← from Qwen3/Mistral encoder outlet
            conditioningMask: MLXArray [B, seq]
            timestep:         MLXArray [B]
        }
outlet: MLXArray [B, H, W, 16]                           ← velocity prediction (flow matching)
```

**Shape contract properties**:
- `expectedConditioningDim`: model-variant-specific (matches connected encoder's `outputEmbeddingDim`)
- `outputLatentChannels: 16` — matches `FluxVAEDecoder.expectedInputChannels`

**Patchification is internal**: The backbone's patchify layer converts `[B, H, W, 16]` → `[B, H/2, W/2, 64]` on entry for the transformer blocks, and unpatchifies back to `[B, H, W, 16]` on exit. The pipeline and the pipe contract only see 16-channel latents. The 64-channel internal representation is a backbone implementation detail.

---

## F3. Text Encoders

FLUX uses LLM-based text encoders rather than CLIP or T5. These are model-specific and remain in this package, conforming to SwiftTubería's `TextEncoder` protocol.

### F3.1 Qwen3 Encoder (Klein models)

- Qwen3-4B for Klein 4B variant
- Qwen3-8B for Klein 9B variant
- Causal LLM used as encoder (final hidden states as embeddings)

### F3.2 Mistral Encoder (Dev model)

- Mistral 7B for the Dev variant
- Same pattern: causal LLM as encoder

Both encoders conform to `TextEncoder` protocol:
```
inlet:  TextEncoderInput  { text: String, maxLength: Int }
outlet: TextEncoderOutput { embeddings: MLXArray, mask: MLXArray }
```

---

## F4. FLUX VAE Decoder

FLUX uses its own VAE, distinct from the SDXL VAE. This decoder remains in this package, conforming to SwiftTubería's `Decoder` and `BidirectionalDecoder` protocols.

```
Decoder:
  inlet:  MLXArray [B, H, W, 16]                         ← 16 latent channels
  outlet: DecodedOutput { data: MLXArray [B, H, W, 3], metadata: ... }

BidirectionalDecoder (for image-to-image):
  encode inlet:  MLXArray [B, H, W, 3]                   ← normalized float pixels
  encode outlet: MLXArray [B, H, W, 16]                  ← latent representation
```

**Shape contract**: `expectedInputChannels: 16` — matches FLUX backbone's `outputLatentChannels`.

**Scaling factor**: The FLUX VAE Decoder applies `latents * (1.0 / scalingFactor)` internally in its `decode()` method. The pipeline passes raw latents from the last denoising step — it does NOT touch the scaling factor. This co-locates scaling logic with the VAE implementation.

**Image-to-image**: `FluxVAEDecoder` conforms to `BidirectionalDecoder`, providing `encode()` for the pipeline to convert reference images into the FLUX latent space. The pipeline handles the full img2img flow (encode → add noise → denoise → decode). The backbone never knows it's doing img2img — it receives latents and produces velocity predictions either way. See SwiftTubería REQUIREMENTS.md §R2.4.1.

---

## F5. Pipeline Recipes

### F5.1 Klein 4B Recipe

```
┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│Qwen3Encoder  │───▶│  FluxDiT     │───▶│ FluxVAE      │───▶│ImageRenderer │
│ (THIS REPO)  │    │ (THIS REPO)  │    │ (THIS REPO)  │    │  (catalog)   │
└──────────────┘    └──────────────┘    └──────────────┘    └──────────────┘
                          ▲
                    ┌─────┴──────┐
                    │FlowMatch   │
                    │Euler       │
                    │ (catalog)  │
                    └────────────┘
```

### F5.2 Dev Recipe

Same structure but with `MistralEncoder` instead of `Qwen3Encoder`.

### F5.3 Generation Modes

FLUX supports both text-to-image and image-to-image (1–3 reference images):
- **Text-to-image** — standard noise → denoise → decode
- **Image-to-image** — pipeline encodes references via `FluxVAEDecoder.encode()` (BidirectionalDecoder) → adds noise at strength → denoises from noisy reference latents → decodes

Image-to-image is handled at the pipeline level (see SwiftTubería §R2.4.1, §R3.2). The FLUX recipe declares `supportsImageToImage = true`, which requires `FluxVAEDecoder` to conform to `BidirectionalDecoder`. The backbone never sees reference images or knows the generation mode.

---

## F6. LoRA Support

FLUX LoRA adapters target:
- Double-stream block attention layers (Q, K, V, out — both image and text streams)
- Single-stream block attention layers (Q, K, V, out)

This package declares the target layer paths. SwiftTubería's LoRA infrastructure handles:
- Loading safetensors LoRA weights
- Applying adapters with configurable scale (0.0–1.0)
- Unloading (restoring base weights)
- Activation keyword detection

### F6.1 LoRA Constraint

Single active LoRA per generation (verified limitation from current implementation). Multiple LoRAs require sequential load/unload.

---

## F7. Acervo Component Descriptors

Components registered into SwiftAcervo's Component Registry at import time:

| Component | Acervo ID | Type | Size | Variants |
|---|---|---|---|---|
| FLUX Klein 4B DiT | `flux2-klein-4b-dit-int4` | backbone | ~6 GB (int4) | bf16, 8bit |
| FLUX Klein 9B DiT | `flux2-klein-9b-dit-qint8` | backbone | ~12 GB (qint8) | bf16 |
| FLUX Dev DiT | `flux2-dev-dit` | backbone | ~32 GB+ | varies |
| Qwen3-4B | `qwen3-4b-encoder` | encoder | ~4 GB | — |
| Qwen3-8B | `qwen3-8b-encoder` | encoder | ~8 GB | — |
| Mistral 7B | `mistral-7b-encoder` | encoder | ~7 GB | — |
| FLUX VAE | `flux2-vae-decoder-fp16` | decoder | ~200 MB | — |

Pipeline code accesses these components exclusively through `AcervoManager.shared.withComponentAccess(id)` — never through file paths.

**Quantization variants**: Each quantization level is a separate Acervo component ID. Different quantizations produce different files with different sizes and checksums. Pattern: `{model}-{quantization}`, e.g., `flux2-klein-4b-dit-bf16`, `flux2-klein-4b-dit-int4`. Pre-quantized safetensors are the norm; on-the-fly quantization is a fallback.

**Registration timing**: Components are registered at import time via Swift static `let` initialization, following the same pattern as pixart-swift-mlx. Swift guarantees thread-safety and exactly-once execution.

---

## F8. Migration Path

The migration from standalone library to pipeline plugin is incremental:

1. **Phase 1 — Dual mode**: Flux2Backbone exports both the current `Flux2Pipeline` class (unchanged) and new `Backbone`-conforming types. SwiftVinetas can use either path.

2. **Phase 2 — Pipeline integration**: SwiftVinetas's `Flux2Engine` switches from wrapping `Flux2Pipeline` directly to assembling a `DiffusionPipeline` from the FLUX recipe. The standalone `Flux2Pipeline` class is marked deprecated.

3. **Phase 3 — Cleanup**: Remove the standalone pipeline orchestration code, weight loading code, memory management code. What remains is backbone + encoders + VAE + recipes + key mappings.

This phased approach ensures SwiftVinetas and Flux2CLI continue working throughout the migration.

---

## F9. Testing Strategy

### F9.1 Backbone Tests
- Double-stream block: synthetic input → expected output shape
- Single-stream block: synthetic input → expected output shape
- Full backbone forward pass: expected shapes per model variant

### F9.2 Encoder Tests
- Qwen3 encoder: known prompt → expected embedding shape
- Mistral encoder: known prompt → expected embedding shape
- Both conform to TextEncoder protocol contract

### F9.3 Integration Tests
- Klein 4B recipe: prompt → CGImage
- Image-to-image: reference image + prompt → CGImage
- LoRA: load adapter → verify generation differs from base
- Seed reproducibility

### F9.4 What Is NOT Tested Here
- FlowMatchEulerScheduler (tested in SwiftTubería catalog)
- ImageRenderer (tested in SwiftTubería catalog)
- Weight loading mechanics (tested in SwiftTubería infrastructure)
- LoRA loading mechanics (tested in SwiftTubería infrastructure)

### F9.5 Coverage and CI Stability Requirements

- All new code must achieve **≥90% line coverage** in unit tests. Coverage is measured per-target and enforced in CI.
- **No timed tests**: Tests must not use `sleep()`, `Task.sleep()`, `Thread.sleep()`, fixed-duration `XCTestExpectation` timeouts, or any wall-clock assertions. All asynchronous behavior must be validated via deterministic synchronization (`async`/`await`, `AsyncStream`, fulfilled expectations with immediate triggers).
- **No environment-dependent tests**: Backbone and encoder unit tests (F9.1, F9.2) must use synthetic inputs and run without real model weights or GPU. Integration tests (F9.3) that require downloaded models and GPU compute must be clearly separated (separate test target or `#if INTEGRATION_TESTS` gate).
- **Flaky tests are test failures**: A test that passes intermittently is treated as a failing test until fixed. CI must not use retry-on-failure to mask flakiness.

---

## F10. SwiftVinetas Integration

SwiftVinetas's `Flux2Engine` currently wraps `Flux2Pipeline` directly. Post-migration:

1. `Flux2Engine` imports `Flux2Backbone` and `Tubería`
2. Constructs a `DiffusionPipeline` using the appropriate FLUX recipe (Klein 4B, Klein 9B, or Dev)
3. Delegates all operations to the pipeline
4. `Flux2ModelDescriptor` continues to describe model variants with memory requirements, download sizes, and default generation parameters

---

## F11. CLI Tool

The Flux2CLI continues to work as a standalone generation and training tool:

```bash
flux2-cli generate --prompt "..." --model klein-4b --output image.png
flux2-cli generate --prompt "..." --images ref1.png ref2.png --mode image-to-image
flux2-cli lora train --dataset ./training-data --output adapter.safetensors
```

Internally, the CLI assembles the appropriate FLUX recipe and uses the pipeline.

---

## F12. Current Public API Preservation

The following types remain public (potentially re-exported through the pipeline):

- `Flux2Model` enum (klein4B, klein9B, dev)
- `Flux2QuantizationConfig` (full, balanced, ultraMinimal)
- `Flux2GenerationMode` (textToImage, imageToImage)
- `Flux2GenerationResult` (image, seed, steps, guidance, duration)
- `LoRAConfig` (filePath, scale, activationKeyword)

These may be adapted to conform to SwiftTubería's corresponding protocols/types, or wrapped with type aliases for backward compatibility during migration.
