# Integrating Custom Models into Flux.2 Swift MLX

This guide describes how to integrate a custom, pre-trained MLX-compatible image generation model into the Flux.2 Swift MLX framework. The framework provides clear extension points through its registry, configuration, and pipeline architecture.

## Prerequisites

Before starting, ensure your custom model:

- Follows the Flux.2 rectified flow transformer architecture (double-stream + single-stream blocks)
- Has weights in safetensors format (BFL native or Diffusers naming)
- Has a compatible text encoder that produces fixed-dimension embeddings
- Is hosted on HuggingFace (or available as local safetensors files)

## Architecture Overview

The framework has four main layers for model integration:

```
┌─────────────────────────────────────┐
│  Flux2Pipeline                      │  Orchestrates generation phases
│  (Pipeline/Flux2Pipeline.swift)     │
├─────────────────────────────────────┤
│  ModelRegistry                      │  Defines variants, HF repos, paths
│  (Configuration/ModelRegistry.swift)│
├─────────────────────────────────────┤
│  Flux2Model + TransformerConfig     │  Architecture parameters
│  (Configuration/Flux2Config.swift)  │
├─────────────────────────────────────┤
│  Flux2WeightLoader                  │  Weight format mapping + application
│  (Loading/WeightLoader.swift)       │
└─────────────────────────────────────┘
```

## Step-by-Step Integration

### Step 1: Define the Model Family

Add your model to the `Flux2Model` enum in `Sources/Flux2Core/Configuration/Flux2Config.swift`:

```swift
public enum Flux2Model: String, CaseIterable, Sendable {
    case dev = "dev"
    case klein4B = "klein-4b"
    case klein9B = "klein-9b"
    // ...existing cases...
    case myCustomModel = "my-custom"  // Add your model
}
```

Then implement all required properties. The most critical ones are:

```swift
// Joint attention dimension — MUST match your text encoder output
public var jointAttentionDim: Int {
    switch self {
    case .dev: return 15360          // Mistral: 3 × 5120
    case .klein4B: return 7680       // Qwen3-4B: 3 × 2560
    case .klein9B: return 12288      // Qwen3-8B: 3 × 4096
    case .myCustomModel: return 7680 // Your encoder: 3 × hidden_dim
    }
}

// Transformer architecture configuration
public var transformerConfig: Flux2TransformerConfig {
    switch self {
    case .myCustomModel: return .myCustom
    // ...
    }
}

// Whether guidance embeddings are used (Dev uses them, Klein does not)
public var usesGuidanceEmbeds: Bool {
    switch self {
    case .myCustomModel: return false
    // ...
    }
}
```

Other properties to implement: `displayName`, `isForInference`, `isForTraining`, `defaultSteps`, `defaultGuidance`, `estimatedVRAM`, `license`, etc.

### Step 2: Define the Transformer Architecture

Add a static configuration in `Flux2TransformerConfig` that matches your model's `config.json`:

```swift
extension Flux2TransformerConfig {
    /// Custom model: 6 double-stream blocks, 32 single-stream blocks
    public static let myCustom = Flux2TransformerConfig(
        patchSize: 1,
        inChannels: 128,
        outChannels: 128,
        numLayers: 6,              // Double-stream (joint attention) blocks
        numSingleLayers: 32,       // Single-stream blocks
        attentionHeadDim: 128,     // Per-head dimension
        numAttentionHeads: 24,     // Total heads (24 × 128 = 3072 inner dim)
        jointAttentionDim: 7680,   // Text encoder output dim (must match Step 1)
        pooledProjectionDim: 768,  // Timestep/guidance embedding dim
        guidanceEmbeds: false,     // Whether model uses guidance conditioning
        axesDimsRope: [32, 32, 32, 32],  // RoPE dimensions [T, H, W, L]
        ropeTheta: 2000.0,         // RoPE base frequency
        mlpRatio: 3.0,             // FFN expansion ratio
        activationFunction: "silu"
    )
}
```

**How to find these values**: Read the `config.json` from your HuggingFace model and map the fields:

| config.json field | Swift property | Example |
|---|---|---|
| `num_layers` | `numLayers` | 8 |
| `num_single_layers` | `numSingleLayers` | 48 |
| `attention_head_dim` | `attentionHeadDim` | 128 |
| `num_attention_heads` | `numAttentionHeads` | 48 |
| `joint_attention_dim` | `jointAttentionDim` | 15360 |
| `pooled_projection_dim` | `pooledProjectionDim` | 768 |
| `guidance_embeds` | `guidanceEmbeds` | true/false |
| `axes_dims_rope` | `axesDimsRope` | [32,32,32,32] |

Alternatively, you can load it at runtime via `Flux2TransformerConfig.load(from: configURL)` which reads `config.json` directly.

### Step 3: Register Download Variants

Add your model's HuggingFace repository to `TransformerVariant` in `Sources/Flux2Core/Configuration/ModelRegistry.swift`:

```swift
public enum TransformerVariant: String, CaseIterable, Sendable {
    // ...existing variants...
    case myCustom_bf16 = "mycustom-bf16"
}
```

Then implement the required properties:

```swift
public var huggingFaceRepo: String {
    switch self {
    case .myCustom_bf16: return "your-org/your-custom-model"
    // ...
    }
}

public var huggingFaceSubfolder: String? {
    switch self {
    // Some models store transformer in a subfolder
    case .myCustom_bf16: return "transformer"  // or nil if at repo root
    // ...
    }
}

public var modelType: Flux2Model {
    switch self {
    case .myCustom_bf16: return .myCustomModel
    // ...
    }
}

public var estimatedSizeGB: Int {
    switch self {
    case .myCustom_bf16: return 12
    // ...
    }
}

public var isGated: Bool {
    switch self {
    case .myCustom_bf16: return false  // true if license acceptance required
    // ...
    }
}
```

Also update `variant(for:quantization:)` to handle your model:

```swift
public static func variant(for model: Flux2Model, quantization: TransformerQuantization) -> TransformerVariant {
    switch (model, quantization) {
    case (.myCustomModel, _): return .myCustom_bf16
    // On-the-fly quantization: loads bf16, then quantizes to qint8/int4
    // ...existing cases...
    }
}
```

### Step 4: Handle Weight Format Mapping

The framework supports two weight naming formats out of the box:

- **BFL format**: Uses `double_blocks.N.`, `single_blocks.N.` prefixes
- **Diffusers format**: Uses `transformer_blocks.N.`, `single_transformer_blocks.N.` prefixes

Both are auto-detected and mapped to the internal Swift naming (`transformerBlocks.N.`, `singleTransformerBlocks.N.`).

**If your model uses a standard format**, no changes needed — `Flux2WeightLoader.applyTransformerWeights()` handles it automatically.

**If your model uses a custom format**, add a mapping function in `Sources/Flux2Core/Loading/WeightLoader.swift`:

```swift
private static func isMyCustomFormat(_ weights: [String: MLXArray]) -> Bool {
    return weights.keys.contains { $0.hasPrefix("my_custom_prefix.") }
}

private static func mapMyCustomWeights(_ weights: inout [String: MLXArray]) -> [String: MLXArray] {
    var mapped: [String: MLXArray] = [:]
    let allKeys = Array(weights.keys)

    for key in allKeys {
        guard let value = weights.removeValue(forKey: key) else { continue }

        var newKey = key
        // Map your format to internal names
        newKey = newKey.replacingOccurrences(of: "my_prefix.blocks.", with: "transformerBlocks.")
        newKey = newKey.replacingOccurrences(of: ".self_attn.", with: ".attn.")
        // Handle fused QKV if needed (split qkv.weight → toQ/toK/toV)

        mapped[newKey] = value
    }
    return mapped
}
```

Then update `mapTransformerWeights()` to call your function:

```swift
static func mapTransformerWeights(_ weights: inout [String: MLXArray]) -> [String: MLXArray] {
    if isMyCustomFormat(weights) {
        return mapMyCustomWeights(&weights)
    } else if isBFLFormat(weights) {
        return mapBFLTransformerWeights(&weights)
    } else {
        return mapDiffusersTransformerWeights(&weights)
    }
}
```

**Debugging tip**: Use `Flux2WeightLoader.summarizeWeights()` to print weight key names and shapes, then compare with the model's expected parameter names.

### Step 5: Pair with a Text Encoder

Flux.2 uses a two-phase pipeline: text encoding happens first, then the text encoder is unloaded before loading the transformer. This means the text encoder and transformer never coexist in memory.

The text encoder must produce embeddings with dimension matching `jointAttentionDim`. The framework includes:

| Text Encoder | Output dim | Joint Attention Dim | Used by |
|---|---|---|---|
| Mistral Small 3.2 (24B) | 5120 | 15360 (5120 × 3) | Dev |
| Qwen3 4B | 2560 | 7680 (2560 × 3) | Klein 4B |
| Qwen3 8B | 4096 | 12288 (4096 × 3) | Klein 9B |

**If your model uses an existing encoder** (e.g., same architecture as Klein 4B), reuse the existing encoder class and update `loadTextEncoder()` in `Flux2Pipeline.swift`:

```swift
private func loadTextEncoder() async throws {
    switch model {
    case .myCustomModel:
        // Reuse Klein's Qwen3-4B encoder
        kleinEncoder = KleinTextEncoder(variant: .klein4B, quantization: qwen3Quant)
        try await kleinEncoder!.load()
    // ...
    }
}
```

**If your model uses a different encoder**, you need to:

1. Define a variant enum in `Sources/FluxTextEncoders/Configuration/TextEncoderModelRegistry.swift`
2. Create an encoder class that produces embeddings of the correct dimension
3. Register it in the pipeline's `loadTextEncoder()` switch

### Step 6: Update the Pipeline

In `Sources/Flux2Core/Pipeline/Flux2Pipeline.swift`, update the generation flow to handle your model. The key methods to update:

```swift
// In loadTextEncoder() — add encoder selection for your model
// In loadTransformer() — usually works automatically via ModelRegistry
// In generate() — usually works automatically via transformerConfig
```

Most of the pipeline logic is model-agnostic — it reads parameters from `Flux2Model` and `Flux2TransformerConfig`. If your model follows the same architecture pattern, the generation methods work without changes.

### Step 7: On-the-fly Quantization

On-the-fly quantization works automatically for any model that follows the standard architecture. The framework loads bf16 weights and converts `Linear` layers to `QuantizedLinear`:

```swift
// In Flux2Pipeline.loadTransformer():
if quantization.transformer == .qint8 {
    MLX.quantize(model: transformer!, groupSize: 64, bits: 8)
} else if quantization.transformer == .int4 {
    MLX.quantize(model: transformer!, groupSize: 64, bits: 4)
}
```

No changes needed for custom models — `quantize()` is architecture-agnostic. Just ensure your `variant(for:quantization:)` returns the bf16 variant when no pre-quantized variant exists, and the pipeline handles the rest.

### Step 8: LoRA Training Support

If you want to support LoRA training on your custom model:

1. Add a base (non-distilled) variant if applicable
2. Set `isForTraining: true` on the appropriate variant
3. Update `trainingVariant(for:)` to return the correct variant

The LoRA injection system (`LoRAInjectedLinear`) works on any `Linear` layer regardless of model architecture. Training configuration via YAML references the model by name:

```yaml
model:
  name: my-custom
  quantization: bf16
```

## Complete Example: Adding a Hypothetical "Klein 16B" Model

Here is a condensed example showing all the changes needed to add a hypothetical Klein 16B model that uses Qwen3-8B as its text encoder:

```swift
// 1. Flux2Config.swift — Add model enum case
case klein16B = "klein-16b"

// 2. Flux2Config.swift — Add transformer config
public static let klein16B = Flux2TransformerConfig(
    patchSize: 1, inChannels: 128, outChannels: 128,
    numLayers: 12, numSingleLayers: 36,
    attentionHeadDim: 128, numAttentionHeads: 48,
    jointAttentionDim: 12288,   // Qwen3-8B: 3 × 4096
    pooledProjectionDim: 768, guidanceEmbeds: false,
    axesDimsRope: [32, 32, 32, 32], ropeTheta: 2000.0,
    mlpRatio: 3.0, activationFunction: "silu"
)

// 3. ModelRegistry.swift — Add variant
case klein16B_bf16 = "klein16b-bf16"
// huggingFaceRepo: "black-forest-labs/FLUX.2-klein-16B"
// modelType: .klein16B

// 4. Flux2Pipeline.swift — Reuse Qwen3-8B encoder
case .klein16B:
    kleinEncoder = KleinTextEncoder(variant: .klein9B, quantization: qwen3Quant)
```

That's it — the rest (weight loading, quantization, LoRA support) works automatically because the model follows the Flux.2 architecture pattern.

## Key Constraints

| Constraint | Detail |
|---|---|
| **Architecture** | Must follow Flux.2's double-stream + single-stream transformer pattern |
| **Joint Attention Dim** | Must exactly match `text_encoder.hidden_size × 3` |
| **Weight Format** | safetensors with BFL or Diffusers naming (or add custom mapping) |
| **Memory Phases** | Text encoder and transformer are never loaded simultaneously |
| **Training** | Requires non-distilled (base) model weights for LoRA training |
| **VAE** | All Flux.2 models share the same VAE — no changes needed |

## Troubleshooting

**Shape mismatch errors during generation**: Check that `jointAttentionDim` matches your text encoder's output dimension exactly.

**Missing weights during loading**: Use `Flux2WeightLoader.summarizeWeights()` to print the key names from your safetensors file, then compare with the expected internal parameter names.

**Model downloads fail**: Check `isGated` — if true, the user needs to accept the license on HuggingFace and provide an `HF_TOKEN`.

**Poor generation quality**: Ensure `defaultSteps` and `defaultGuidance` are appropriate for your model (distilled models typically use 4 steps with guidance 1.0, non-distilled need 20-50 steps with guidance 3-5).
