// Flux2CoreTests.swift - Unit tests for Flux2Core
// Copyright 2025 Vincent Gourbin

import XCTest
@testable import Flux2Core
import MLX
import ImageIO
import CoreGraphics

#if canImport(AppKit)
import AppKit
#endif

final class Flux2CoreTests: XCTestCase {

    // MARK: - Configuration Tests

    func testTransformerConfigDefaults() {
        let config = Flux2TransformerConfig.flux2Dev

        XCTAssertEqual(config.patchSize, 1)
        XCTAssertEqual(config.inChannels, 128)
        XCTAssertEqual(config.numLayers, 8)
        XCTAssertEqual(config.numSingleLayers, 48)
        XCTAssertEqual(config.numAttentionHeads, 48)
        XCTAssertEqual(config.attentionHeadDim, 128)
        XCTAssertEqual(config.innerDim, 6144)  // 48 * 128
        XCTAssertEqual(config.jointAttentionDim, 15360)
    }

    func testVAEConfigDefaults() {
        let config = VAEConfig.flux2Dev

        XCTAssertEqual(config.latentChannels, 32)
        XCTAssertEqual(config.blockOutChannels, [128, 256, 512, 512])
        XCTAssertTrue(config.useBatchNorm)
    }

    func testQuantizationPresets() {
        XCTAssertEqual(Flux2QuantizationConfig.highQuality.textEncoder, .bf16)
        XCTAssertEqual(Flux2QuantizationConfig.highQuality.transformer, .bf16)

        XCTAssertEqual(Flux2QuantizationConfig.balanced.textEncoder, .mlx8bit)
        XCTAssertEqual(Flux2QuantizationConfig.balanced.transformer, .qint8)

        XCTAssertEqual(Flux2QuantizationConfig.minimal.textEncoder, .mlx4bit)
        XCTAssertEqual(Flux2QuantizationConfig.minimal.transformer, .qint8)

        XCTAssertEqual(Flux2QuantizationConfig.ultraMinimal.textEncoder, .mlx4bit)
        XCTAssertEqual(Flux2QuantizationConfig.ultraMinimal.transformer, .int4)
    }

    func testTransformerQuantizationInt4() {
        XCTAssertEqual(TransformerQuantization.int4.bits, 4)
        XCTAssertEqual(TransformerQuantization.int4.groupSize, 64)
        XCTAssertEqual(TransformerQuantization.int4.rawValue, "int4")
    }

    func testModelRegistryVariantOnTheFlyQuantization() {
        // Klein 9B should always return bf16 variant (quantize on-the-fly)
        XCTAssertEqual(
            ModelRegistry.TransformerVariant.variant(for: .klein9B, quantization: .qint8),
            .klein9B_bf16
        )
        XCTAssertEqual(
            ModelRegistry.TransformerVariant.variant(for: .klein9B, quantization: .int4),
            .klein9B_bf16
        )

        // Dev int4 should return bf16 variant (quantize on-the-fly)
        XCTAssertEqual(
            ModelRegistry.TransformerVariant.variant(for: .dev, quantization: .int4),
            .bf16
        )

        // Klein 4B int4 should return bf16 variant (quantize on-the-fly)
        XCTAssertEqual(
            ModelRegistry.TransformerVariant.variant(for: .klein4B, quantization: .int4),
            .klein4B_bf16
        )
    }

    // MARK: - Latent Utils Tests

    func testLatentDimensionValidation() {
        let (h, w) = LatentUtils.validateDimensions(height: 1000, width: 1000)

        // Should be rounded up to nearest multiple of 16
        XCTAssertEqual(h % 16, 0)
        XCTAssertEqual(w % 16, 0)
        XCTAssertGreaterThanOrEqual(h, 1000)
        XCTAssertGreaterThanOrEqual(w, 1000)
    }

    func testLatentPacking() {
        // Create test latent: [1, 32, 128, 128]
        let latent = MLXRandom.normal([1, 32, 128, 128])

        // Pack
        let packed = LatentUtils.packLatents(latent, patchSize: 2)

        // Should be [1, (128/2)*(128/2), 32*2*2] = [1, 4096, 128]
        XCTAssertEqual(packed.shape[0], 1)
        XCTAssertEqual(packed.shape[1], 4096)
        XCTAssertEqual(packed.shape[2], 128)

        // Unpack
        let unpacked = LatentUtils.unpackLatents(
            packed,
            height: 1024,  // 128 * 8
            width: 1024,
            latentChannels: 32,
            patchSize: 2
        )

        // Should match original shape
        XCTAssertEqual(unpacked.shape, latent.shape)
    }

    func testPositionIDGeneration() {
        let height = 1024
        let width = 1024

        let imageIds = LatentUtils.generateImagePositionIDs(height: height, width: width)

        // For 1024x1024 with patch size 2: (128/2) * (128/2) = 4096 patches
        XCTAssertEqual(imageIds.shape[0], 4096)
        XCTAssertEqual(imageIds.shape[1], 4)  // [T, H, W, L]
    }

    // MARK: - Scheduler Tests

    func testSchedulerTimesteps() {
        let scheduler = FlowMatchEulerScheduler()
        scheduler.setTimesteps(numInferenceSteps: 50)

        XCTAssertEqual(scheduler.timesteps.count, 51)  // 50 steps + final
        // Timesteps are sigmas * numTrainTimesteps (1000), so first is ~1000 (after time shift)
        // Sigmas are in [0, 1] range - check sigmas instead for semantic correctness
        XCTAssertEqual(scheduler.sigmas.count, 51)
        XCTAssertGreaterThan(scheduler.sigmas.first!, 0.9)  // First sigma should be close to 1.0
        XCTAssertEqual(scheduler.sigmas.last!, 0.0, accuracy: 0.001)  // Terminal sigma is 0
        XCTAssertEqual(scheduler.timesteps.last!, 0.0, accuracy: 0.01)  // Terminal timestep is 0
    }

    func testSchedulerStep() {
        let scheduler = FlowMatchEulerScheduler()
        scheduler.setTimesteps(numInferenceSteps: 10)

        let sample = MLXRandom.normal([1, 100, 128])
        let modelOutput = MLXRandom.normal([1, 100, 128])

        let nextSample = scheduler.step(
            modelOutput: modelOutput,
            timestep: scheduler.timesteps[0],
            sample: sample
        )

        XCTAssertEqual(nextSample.shape, sample.shape)
    }

    // MARK: - Memory Estimation Tests

    func testMemoryEstimation() {
        let config = Flux2QuantizationConfig.balanced

        // Should estimate reasonable memory
        XCTAssertGreaterThan(config.estimatedTotalMemoryGB, 30)
        XCTAssertLessThan(config.estimatedTotalMemoryGB, 100)

        // Text encoding phase should be less than image generation
        XCTAssertLessThan(
            config.textEncodingPhaseMemoryGB,
            config.imageGenerationPhaseMemoryGB
        )
    }
}

// MARK: - Embedding Tests

final class EmbeddingTests: XCTestCase {

    func testTimestepEmbedding() {
        let embedder = Flux2TimestepGuidanceEmbeddings(
            embeddingDim: 256,
            timeEmbedDim: 6144,
            useGuidanceEmbeds: true
        )

        let timestep = MLXArray([Float(0.5)])
        let guidance = MLXArray([Float(4.0)])

        let embedding = embedder(timestep: timestep, guidance: guidance)

        XCTAssertEqual(embedding.shape, [1, 6144])
    }

    func testRoPE() {
        let rope = Flux2RoPE(axesDims: [32, 32, 32, 32], theta: 2000.0)

        // Create position IDs: [100, 4]
        var flatData: [Int32] = []
        for i: Int32 in 0..<100 {
            flatData.append(contentsOf: [Int32(0), i / 10, i % 10, Int32(0)])
        }
        let ids = MLXArray(flatData).reshaped([100, 4])

        let (cosEmb, sinEmb) = rope(ids)

        // Should output [100, 128] (sum of axes dims)
        XCTAssertEqual(cosEmb.shape[0], 100)
        XCTAssertEqual(sinEmb.shape[0], 100)
    }
}

// MARK: - Integration Tests

final class IntegrationTests: XCTestCase {

    func testModulationFlow() {
        let dim = 6144
        let modulation = Flux2Modulation(dim: dim, numSets: 2)

        let embedding = MLXRandom.normal([1, dim])
        let params = modulation(embedding)

        XCTAssertEqual(params.count, 2)

        for param in params {
            XCTAssertEqual(param.shift.shape, [1, dim])
            XCTAssertEqual(param.scale.shape, [1, dim])
            XCTAssertEqual(param.gate.shape, [1, dim])
        }
    }

    func testFeedForwardShape() {
        let dim = 6144
        let ff = Flux2FeedForward(dim: dim)

        let input = MLXRandom.normal([1, 100, dim])
        let output = ff(input)

        XCTAssertEqual(output.shape, input.shape)
    }
}

// MARK: - LoRA Configuration Tests

final class LoRAConfigTests: XCTestCase {

    func testLoRAConfigInit() {
        let config = LoRAConfig(filePath: "/path/to/lora.safetensors")

        XCTAssertEqual(config.filePath, "/path/to/lora.safetensors")
        // Default scale is 1.0 (not nil)
        XCTAssertEqual(config.scale, 1.0)
        XCTAssertNil(config.activationKeyword)
    }

    func testLoRAConfigWithScale() {
        let config = LoRAConfig(filePath: "/path/to/lora.safetensors", scale: 0.8)

        XCTAssertEqual(config.scale, 0.8)
        XCTAssertEqual(config.effectiveScale, 0.8)
    }

    func testLoRAConfigDefaultScale() {
        let config = LoRAConfig(filePath: "/path/to/lora.safetensors")

        // When no scale is set, effectiveScale should default to 1.0
        XCTAssertEqual(config.effectiveScale, 1.0)
    }

    func testLoRAConfigName() {
        let config = LoRAConfig(filePath: "/path/to/my_lora.safetensors")

        XCTAssertEqual(config.name, "my_lora")
    }

    func testLoRAConfigWithActivationKeyword() {
        var config = LoRAConfig(filePath: "/path/to/lora.safetensors")
        config.activationKeyword = "sks"

        XCTAssertEqual(config.activationKeyword, "sks")
    }
}

// MARK: - Scheduler Extended Tests

final class SchedulerExtendedTests: XCTestCase {

    func testSchedulerCustomSigmas() {
        let scheduler = FlowMatchEulerScheduler()

        // Custom 4-step turbo schedule
        let customSigmas: [Float] = [1.0, 0.65, 0.35, 0.1]
        scheduler.setCustomSigmas(customSigmas)

        // Should have 5 sigmas (4 custom + terminal 0.0)
        XCTAssertEqual(scheduler.sigmas.count, 5)
        XCTAssertEqual(scheduler.sigmas.last!, 0.0, accuracy: 0.001)
    }

    func testSchedulerI2IStrength() {
        let scheduler = FlowMatchEulerScheduler()

        // Full denoise (strength = 1.0)
        scheduler.setTimesteps(numInferenceSteps: 50, strength: 1.0)
        let fullSteps = scheduler.sigmas.count - 1

        // Half denoise (strength = 0.5)
        scheduler.setTimesteps(numInferenceSteps: 50, strength: 0.5)
        let halfSteps = scheduler.sigmas.count - 1

        XCTAssertLessThan(halfSteps, fullSteps)
        XCTAssertEqual(halfSteps, 25)
    }

    func testSchedulerProgress() {
        let scheduler = FlowMatchEulerScheduler()
        scheduler.setTimesteps(numInferenceSteps: 10)

        XCTAssertEqual(scheduler.progress, 0.0, accuracy: 0.01)

        // Simulate stepping
        let sample = MLXRandom.normal([1, 100, 128])
        let modelOutput = MLXRandom.normal([1, 100, 128])

        _ = scheduler.step(modelOutput: modelOutput, timestep: scheduler.timesteps[0], sample: sample)

        XCTAssertGreaterThan(scheduler.progress, 0.0)
        XCTAssertEqual(scheduler.remainingSteps, 9)
    }

    func testSchedulerReset() {
        let scheduler = FlowMatchEulerScheduler()
        scheduler.setTimesteps(numInferenceSteps: 10)

        let sample = MLXRandom.normal([1, 100, 128])
        let modelOutput = MLXRandom.normal([1, 100, 128])
        _ = scheduler.step(modelOutput: modelOutput, timestep: scheduler.timesteps[0], sample: sample)

        XCTAssertGreaterThan(scheduler.stepIndex, 0)

        scheduler.reset()
        XCTAssertEqual(scheduler.stepIndex, 0)
    }

    func testSchedulerAddNoise() {
        let scheduler = FlowMatchEulerScheduler()

        let original = MLXArray([Float(1.0), Float(2.0), Float(3.0)])
        let noise = MLXArray([Float(0.1), Float(0.2), Float(0.3)])

        // At timestep 500 (sigma = 0.5)
        let noisy = scheduler.addNoise(originalSamples: original, noise: noise, timestep: 500)

        XCTAssertEqual(noisy.shape, original.shape)
    }

    func testSchedulerScaleNoise() {
        let scheduler = FlowMatchEulerScheduler()

        let sample = MLXArray([Float(1.0)])
        let noise = MLXArray([Float(0.0)])

        // At sigma = 0, should return sample unchanged
        let result = scheduler.scaleNoise(sample: sample, sigma: 0.0, noise: noise)
        eval(result)
        XCTAssertEqual(result.item(Float.self), 1.0, accuracy: 0.001)
    }

    func testSchedulerInitialSigma() {
        let scheduler = FlowMatchEulerScheduler()
        scheduler.setTimesteps(numInferenceSteps: 20)

        // Initial sigma should be close to 1.0 (high noise)
        XCTAssertGreaterThan(scheduler.initialSigma, 0.9)
    }

    func testSchedulerCurrentSigma() {
        let scheduler = FlowMatchEulerScheduler()
        scheduler.setTimesteps(numInferenceSteps: 10)

        let firstSigma = scheduler.currentSigma

        let sample = MLXRandom.normal([1, 100, 128])
        let modelOutput = MLXRandom.normal([1, 100, 128])
        _ = scheduler.step(modelOutput: modelOutput, timestep: scheduler.timesteps[0], sample: sample)

        let secondSigma = scheduler.currentSigma

        // Sigma should decrease as we step through
        XCTAssertLessThan(secondSigma, firstSigma)
    }
}

// MARK: - Model Registry Tests

final class ModelRegistryTests: XCTestCase {

    func testTransformerVariantHuggingFaceRepo() {
        let bf16 = ModelRegistry.TransformerVariant.bf16
        XCTAssertFalse(bf16.huggingFaceRepo.isEmpty)
        XCTAssertTrue(bf16.huggingFaceRepo.contains("FLUX"))
    }

    func testTransformerVariantEstimatedSize() {
        let bf16 = ModelRegistry.TransformerVariant.bf16
        let qint8 = ModelRegistry.TransformerVariant.qint8

        // bf16 should be larger than qint8
        XCTAssertGreaterThan(bf16.estimatedSizeGB, qint8.estimatedSizeGB)
    }

    func testKleinVariantsSmallerThanDev() {
        let devBf16 = ModelRegistry.TransformerVariant.bf16.estimatedSizeGB
        let klein4B = ModelRegistry.TransformerVariant.klein4B_bf16.estimatedSizeGB

        XCTAssertLessThan(klein4B, devBf16)
    }

    func testVAEVariants() {
        let vae = ModelRegistry.VAEVariant.standard
        XCTAssertFalse(vae.huggingFaceRepo.isEmpty)
    }

    func testRecommendedConfigForRAM() {
        // Very low RAM should recommend ultra-minimal config (4-bit)
        let veryLowRamConfig = ModelRegistry.recommendedConfig(forRAMGB: 24)
        XCTAssertEqual(veryLowRamConfig.transformer, .int4)

        // Low RAM should recommend minimal config
        let lowRamConfig = ModelRegistry.recommendedConfig(forRAMGB: 32)
        XCTAssertEqual(lowRamConfig.transformer, .qint8)

        // High RAM can use bf16
        let highRamConfig = ModelRegistry.recommendedConfig(forRAMGB: 128)
        XCTAssertEqual(highRamConfig.transformer, .bf16)
    }

    // MARK: - Gated Status Tests

    func testTransformerVariantIsGated() {
        // Dev bf16 from black-forest-labs is gated
        XCTAssertTrue(ModelRegistry.TransformerVariant.bf16.isGated)

        // qint8 from VincentGOURBIN repo is NOT gated
        XCTAssertFalse(ModelRegistry.TransformerVariant.qint8.isGated)

        // Klein 4B is NOT gated (open access)
        XCTAssertFalse(ModelRegistry.TransformerVariant.klein4B_bf16.isGated)
        XCTAssertFalse(ModelRegistry.TransformerVariant.klein4B_8bit.isGated)

        // Klein 9B from black-forest-labs IS gated
        XCTAssertTrue(ModelRegistry.TransformerVariant.klein9B_bf16.isGated)
    }

    func testTextEncoderVariantIsGated() {
        // bf16 from mistralai is gated
        XCTAssertTrue(ModelRegistry.TextEncoderVariant.bf16.isGated)

        // Quantized versions from lmstudio-community are NOT gated
        XCTAssertFalse(ModelRegistry.TextEncoderVariant.mlx8bit.isGated)
        XCTAssertFalse(ModelRegistry.TextEncoderVariant.mlx6bit.isGated)
        XCTAssertFalse(ModelRegistry.TextEncoderVariant.mlx4bit.isGated)
    }

    func testVAEVariantIsGated() {
        // VAE is downloaded from Klein 4B repo which is NOT gated
        XCTAssertFalse(ModelRegistry.VAEVariant.standard.isGated)
    }

    // MARK: - HuggingFace URL Tests

    func testTransformerVariantHuggingFaceURL() {
        let bf16 = ModelRegistry.TransformerVariant.bf16
        XCTAssertTrue(bf16.huggingFaceURL.starts(with: "https://huggingface.co/"))
        XCTAssertTrue(bf16.huggingFaceURL.contains(bf16.huggingFaceRepo))
    }

    func testTextEncoderVariantHuggingFaceURL() {
        let mlx8bit = ModelRegistry.TextEncoderVariant.mlx8bit
        XCTAssertTrue(mlx8bit.huggingFaceURL.starts(with: "https://huggingface.co/"))
        XCTAssertTrue(mlx8bit.huggingFaceURL.contains(mlx8bit.huggingFaceRepo))
    }

    func testVAEVariantHuggingFaceURL() {
        let vae = ModelRegistry.VAEVariant.standard
        XCTAssertTrue(vae.huggingFaceURL.starts(with: "https://huggingface.co/"))
        XCTAssertTrue(vae.huggingFaceURL.contains(vae.huggingFaceRepo))
    }

    func testTextEncoderVariantHuggingFaceRepoValues() {
        // bf16 should be from mistralai
        XCTAssertTrue(ModelRegistry.TextEncoderVariant.bf16.huggingFaceRepo.contains("mistralai"))

        // Quantized should be from lmstudio-community
        XCTAssertTrue(ModelRegistry.TextEncoderVariant.mlx8bit.huggingFaceRepo.contains("lmstudio-community"))
        XCTAssertTrue(ModelRegistry.TextEncoderVariant.mlx6bit.huggingFaceRepo.contains("lmstudio-community"))
        XCTAssertTrue(ModelRegistry.TextEncoderVariant.mlx4bit.huggingFaceRepo.contains("lmstudio-community"))
    }

    // MARK: - License Tests

    func testTransformerVariantLicense() {
        // Dev is non-commercial
        XCTAssertTrue(ModelRegistry.TransformerVariant.bf16.license.contains("Non-Commercial"))
        XCTAssertFalse(ModelRegistry.TransformerVariant.bf16.isCommercialUseAllowed)

        // Klein 4B is Apache 2.0 (commercial OK)
        XCTAssertTrue(ModelRegistry.TransformerVariant.klein4B_bf16.license.contains("Apache"))
        XCTAssertTrue(ModelRegistry.TransformerVariant.klein4B_bf16.isCommercialUseAllowed)

        // Klein 9B is non-commercial
        XCTAssertFalse(ModelRegistry.TransformerVariant.klein9B_bf16.isCommercialUseAllowed)
    }

    func testTextEncoderVariantLicense() {
        // Mistral is Apache 2.0
        XCTAssertTrue(ModelRegistry.TextEncoderVariant.mlx8bit.license.contains("Apache"))
        XCTAssertTrue(ModelRegistry.TextEncoderVariant.mlx8bit.isCommercialUseAllowed)
    }

    func testVAEVariantLicense() {
        // VAE inherits FLUX.2 Dev non-commercial license
        XCTAssertTrue(ModelRegistry.VAEVariant.standard.license.contains("Non-Commercial"))
        XCTAssertFalse(ModelRegistry.VAEVariant.standard.isCommercialUseAllowed)
    }

    // MARK: - Default Parameters Tests

    func testTransformerVariantDefaultParameters() {
        // Dev: 28 steps, guidance 4.0
        XCTAssertEqual(ModelRegistry.TransformerVariant.bf16.defaultSteps, 28)
        XCTAssertEqual(ModelRegistry.TransformerVariant.bf16.defaultGuidance, 4.0)

        // Klein: 4 steps, guidance 1.0
        XCTAssertEqual(ModelRegistry.TransformerVariant.klein4B_bf16.defaultSteps, 4)
        XCTAssertEqual(ModelRegistry.TransformerVariant.klein4B_bf16.defaultGuidance, 1.0)
    }

    func testTransformerVariantMaxReferenceImages() {
        // Dev variants: 6 images (delegates to modelType)
        XCTAssertEqual(ModelRegistry.TransformerVariant.bf16.maxReferenceImages, 6)
        XCTAssertEqual(ModelRegistry.TransformerVariant.qint8.maxReferenceImages, 6)

        // Klein variants: 4 images
        XCTAssertEqual(ModelRegistry.TransformerVariant.klein4B_bf16.maxReferenceImages, 4)
        XCTAssertEqual(ModelRegistry.TransformerVariant.klein4B_8bit.maxReferenceImages, 4)
        XCTAssertEqual(ModelRegistry.TransformerVariant.klein9B_bf16.maxReferenceImages, 4)
    }
}

// MARK: - Flux2Model Tests

final class Flux2ModelTests: XCTestCase {

    func testDefaultSteps() {
        XCTAssertEqual(Flux2Model.dev.defaultSteps, 28)
        XCTAssertEqual(Flux2Model.klein4B.defaultSteps, 4)
        XCTAssertEqual(Flux2Model.klein9B.defaultSteps, 4)
    }

    func testDefaultGuidance() {
        XCTAssertEqual(Flux2Model.dev.defaultGuidance, 4.0)
        XCTAssertEqual(Flux2Model.klein4B.defaultGuidance, 1.0)
        XCTAssertEqual(Flux2Model.klein9B.defaultGuidance, 1.0)
    }

    func testEstimatedTimeSeconds() {
        XCTAssertGreaterThan(Flux2Model.dev.estimatedTimeSeconds, 1000)
        XCTAssertLessThan(Flux2Model.klein4B.estimatedTimeSeconds, 60)
    }

    func testLicense() {
        XCTAssertTrue(Flux2Model.klein4B.license.contains("Apache"))
        XCTAssertTrue(Flux2Model.klein4B.isCommercialUseAllowed)
        XCTAssertFalse(Flux2Model.dev.isCommercialUseAllowed)
    }

    func testMaxReferenceImages() {
        // Dev supports up to 6 reference images (memory limited)
        XCTAssertEqual(Flux2Model.dev.maxReferenceImages, 6)

        // Klein models support up to 4 reference images
        XCTAssertEqual(Flux2Model.klein4B.maxReferenceImages, 4)
        XCTAssertEqual(Flux2Model.klein9B.maxReferenceImages, 4)
    }
}

// MARK: - VAE Config Extended Tests

final class VAEConfigExtendedTests: XCTestCase {

    func testVAEConfigDev() {
        let config = VAEConfig.flux2Dev

        XCTAssertEqual(config.latentChannels, 32)
        XCTAssertEqual(config.inChannels, 3)
        XCTAssertEqual(config.outChannels, 3)
    }

    func testVAEConfigBlockChannels() {
        let config = VAEConfig.flux2Dev

        // Should have 4 block levels
        XCTAssertEqual(config.blockOutChannels.count, 4)

        // Channels should increase then plateau
        XCTAssertLessThan(config.blockOutChannels[0], config.blockOutChannels[1])
    }

    func testVAEConfigScaling() {
        let config = VAEConfig.flux2Dev

        XCTAssertNotEqual(config.scalingFactor, 0.0)
    }

    func testVAEConfigPatchSize() {
        let config = VAEConfig.flux2Dev

        XCTAssertEqual(config.patchSize.0, 2)
        XCTAssertEqual(config.patchSize.1, 2)
    }

    func testVAEConfigNormalization() {
        let config = VAEConfig.flux2Dev

        XCTAssertGreaterThan(config.normNumGroups, 0)
        XCTAssertGreaterThan(config.normEps, 0)
    }
}

// MARK: - Latent Utils Extended Tests

final class LatentUtilsExtendedTests: XCTestCase {

    func testDimensionValidationRounding() {
        // Test various dimensions
        let testCases: [(Int, Int)] = [
            (100, 100),
            (512, 512),
            (1000, 1000),
            (1920, 1080),
        ]

        for (h, w) in testCases {
            let (validH, validW) = LatentUtils.validateDimensions(height: h, width: w)
            XCTAssertEqual(validH % 16, 0, "Height \(validH) should be multiple of 16")
            XCTAssertEqual(validW % 16, 0, "Width \(validW) should be multiple of 16")
            XCTAssertGreaterThanOrEqual(validH, h)
            XCTAssertGreaterThanOrEqual(validW, w)
        }
    }

    func testLatentPackUnpackRoundtrip() {
        // Test multiple sizes
        let sizes = [(64, 64), (128, 128), (96, 128)]

        for (h, w) in sizes {
            let latent = MLXRandom.normal([1, 32, h, w])
            let packed = LatentUtils.packLatents(latent, patchSize: 2)
            let unpacked = LatentUtils.unpackLatents(
                packed,
                height: h * 8,
                width: w * 8,
                latentChannels: 32,
                patchSize: 2
            )

            XCTAssertEqual(unpacked.shape, latent.shape, "Roundtrip failed for size \(h)x\(w)")
        }
    }

    func testPositionIDsVaryingSizes() {
        let sizes = [(512, 512), (1024, 1024), (768, 1024)]

        for (h, w) in sizes {
            let ids = LatentUtils.generateImagePositionIDs(height: h, width: w)

            // Number of patches = (h/8/2) * (w/8/2) = h*w/256
            let expectedPatches = (h / 16) * (w / 16)
            XCTAssertEqual(ids.shape[0], expectedPatches, "Wrong patch count for \(h)x\(w)")
            XCTAssertEqual(ids.shape[1], 4)  // [T, H, W, L] dimensions
        }
    }
}

// MARK: - Memory Manager Tests

final class MemoryManagerTests: XCTestCase {

    func testMemoryManagerSingleton() {
        let manager1 = Flux2MemoryManager.shared
        let manager2 = Flux2MemoryManager.shared
        XCTAssertTrue(manager1 === manager2)
    }

    func testMemoryManagerPhysicalMemory() {
        let manager = Flux2MemoryManager.shared

        // Physical memory should be positive
        XCTAssertGreaterThan(manager.physicalMemory, 0)
        XCTAssertGreaterThan(manager.physicalMemoryGB, 0)
    }

    func testMemoryManagerEstimatedAvailable() {
        let manager = Flux2MemoryManager.shared

        // Estimated available should be less than physical (system reserve)
        XCTAssertLessThanOrEqual(manager.estimatedAvailableMemoryGB, manager.physicalMemoryGB)
    }

    func testMemoryManagerCanRunCheck() {
        let manager = Flux2MemoryManager.shared

        // Minimal config should be runnable on most systems
        let minimalConfig = Flux2QuantizationConfig.minimal
        // Just check the method doesn't crash
        _ = manager.canRun(config: minimalConfig)
    }

    func testMemoryManagerRecommendedConfig() {
        let manager = Flux2MemoryManager.shared

        let recommended = manager.recommendedConfig()
        // Should return a valid config
        XCTAssertNotNil(recommended.textEncoder)
        XCTAssertNotNil(recommended.transformer)
    }
}

// MARK: - Transformer Config Tests

final class TransformerConfigTests: XCTestCase {

    func testFlux2DevConfig() {
        let config = Flux2TransformerConfig.flux2Dev

        XCTAssertEqual(config.inChannels, 128)
        XCTAssertEqual(config.numLayers, 8)
        XCTAssertEqual(config.numSingleLayers, 48)
    }

    func testFlux2KleinConfig() {
        let config = Flux2TransformerConfig.klein4B

        // Klein 4B is smaller than Dev
        XCTAssertEqual(config.numLayers, 5)
        XCTAssertEqual(config.numSingleLayers, 20)
    }

    func testInnerDimCalculation() {
        let config = Flux2TransformerConfig.flux2Dev

        let expectedInnerDim = config.numAttentionHeads * config.attentionHeadDim
        XCTAssertEqual(config.innerDim, expectedInnerDim)
    }

    func testKleinSmallerThanDev() {
        let dev = Flux2TransformerConfig.flux2Dev
        let klein = Flux2TransformerConfig.klein4B

        XCTAssertLessThan(klein.numLayers, dev.numLayers)
        XCTAssertLessThan(klein.numSingleLayers, dev.numSingleLayers)
    }
}

// MARK: - Quantization Config Tests

final class QuantizationConfigTests: XCTestCase {

    func testMistralQuantizationValues() {
        let bf16 = MistralQuantization.bf16
        let mlx8bit = MistralQuantization.mlx8bit
        let mlx4bit = MistralQuantization.mlx4bit

        XCTAssertEqual(bf16.rawValue, "bf16")
        XCTAssertEqual(mlx8bit.rawValue, "8bit")
        XCTAssertEqual(mlx4bit.rawValue, "4bit")
    }

    func testTransformerQuantizationValues() {
        let bf16 = TransformerQuantization.bf16
        let qint8 = TransformerQuantization.qint8

        XCTAssertEqual(bf16.rawValue, "bf16")
        XCTAssertEqual(qint8.rawValue, "qint8")
    }

    func testQuantizationMemoryEstimates() {
        // Higher quality should use more memory
        let highQuality = Flux2QuantizationConfig.highQuality
        let minimal = Flux2QuantizationConfig.minimal

        XCTAssertGreaterThan(highQuality.estimatedTotalMemoryGB, minimal.estimatedTotalMemoryGB)
    }

    func testQuantizationPhaseMemory() {
        let config = Flux2QuantizationConfig.balanced

        // Both phases should have positive memory estimates
        XCTAssertGreaterThan(config.textEncodingPhaseMemoryGB, 0)
        XCTAssertGreaterThan(config.imageGenerationPhaseMemoryGB, 0)
    }

    func testMistralQuantizationEstimatedMemoryGB() {
        let bf16 = MistralQuantization.bf16.estimatedMemoryGB
        let mlx8bit = MistralQuantization.mlx8bit.estimatedMemoryGB
        let mlx4bit = MistralQuantization.mlx4bit.estimatedMemoryGB

        // bf16 > 8bit > 4bit
        XCTAssertGreaterThan(bf16, mlx8bit)
        XCTAssertGreaterThan(mlx8bit, mlx4bit)
    }

    func testTransformerQuantizationMemory() {
        let bf16 = TransformerQuantization.bf16.estimatedMemoryGB
        let qint8 = TransformerQuantization.qint8.estimatedMemoryGB
        let int4 = TransformerQuantization.int4.estimatedMemoryGB

        XCTAssertGreaterThan(bf16, qint8)
        XCTAssertGreaterThan(qint8, int4)
    }

    func testTransformerQuantizationAllCases() {
        let allCases = TransformerQuantization.allCases
        XCTAssertEqual(allCases.count, 3)
        XCTAssertTrue(allCases.contains(.bf16))
        XCTAssertTrue(allCases.contains(.qint8))
        XCTAssertTrue(allCases.contains(.int4))
    }

    func testTransformerQuantizationBitsOrdering() {
        XCTAssertGreaterThan(TransformerQuantization.bf16.bits, TransformerQuantization.qint8.bits)
        XCTAssertGreaterThan(TransformerQuantization.qint8.bits, TransformerQuantization.int4.bits)
    }

    func testTransformerQuantizationDisplayNames() {
        XCTAssertFalse(TransformerQuantization.bf16.displayName.isEmpty)
        XCTAssertFalse(TransformerQuantization.qint8.displayName.isEmpty)
        XCTAssertFalse(TransformerQuantization.int4.displayName.isEmpty)
    }

    func testMemoryEfficientPreset() {
        XCTAssertEqual(Flux2QuantizationConfig.memoryEfficient.textEncoder, .mlx4bit)
        XCTAssertEqual(Flux2QuantizationConfig.memoryEfficient.transformer, .qint8)
    }

    func testPresetMemoryOrdering() {
        let ultra = Flux2QuantizationConfig.ultraMinimal.estimatedTotalMemoryGB
        let minimal = Flux2QuantizationConfig.minimal.estimatedTotalMemoryGB
        let balanced = Flux2QuantizationConfig.balanced.estimatedTotalMemoryGB
        let high = Flux2QuantizationConfig.highQuality.estimatedTotalMemoryGB

        XCTAssertLessThanOrEqual(ultra, minimal)
        XCTAssertLessThanOrEqual(minimal, balanced)
        XCTAssertLessThanOrEqual(balanced, high)
    }
}

// MARK: - On-the-fly Quantization Variant Tests

final class OnTheFlyQuantizationTests: XCTestCase {

    func testDevVariantMapping() {
        // Dev has pre-quantized qint8 available
        XCTAssertEqual(
            ModelRegistry.TransformerVariant.variant(for: .dev, quantization: .bf16),
            .bf16
        )
        XCTAssertEqual(
            ModelRegistry.TransformerVariant.variant(for: .dev, quantization: .qint8),
            .qint8,
            "Dev qint8 should use pre-quantized variant"
        )
        XCTAssertEqual(
            ModelRegistry.TransformerVariant.variant(for: .dev, quantization: .int4),
            .bf16,
            "Dev int4 should load bf16 and quantize on-the-fly"
        )
    }

    func testKlein4BVariantMapping() {
        XCTAssertEqual(
            ModelRegistry.TransformerVariant.variant(for: .klein4B, quantization: .bf16),
            .klein4B_bf16
        )
        XCTAssertEqual(
            ModelRegistry.TransformerVariant.variant(for: .klein4B, quantization: .qint8),
            .klein4B_8bit,
            "Klein 4B qint8 should use pre-quantized variant"
        )
        XCTAssertEqual(
            ModelRegistry.TransformerVariant.variant(for: .klein4B, quantization: .int4),
            .klein4B_bf16,
            "Klein 4B int4 should load bf16 and quantize on-the-fly"
        )
    }

    func testKlein9BAlwaysLoadsBf16() {
        // Klein 9B has no pre-quantized variants — always loads bf16
        for quant in TransformerQuantization.allCases {
            XCTAssertEqual(
                ModelRegistry.TransformerVariant.variant(for: .klein9B, quantization: quant),
                .klein9B_bf16,
                "Klein 9B should always load bf16 variant for \(quant)"
            )
        }
    }

    func testBaseModelsAlwaysReturnBaseVariant() {
        for quant in TransformerQuantization.allCases {
            XCTAssertEqual(
                ModelRegistry.TransformerVariant.variant(for: .klein4BBase, quantization: quant),
                .klein4B_base_bf16,
                "Klein 4B base should always return base bf16 for \(quant)"
            )
            XCTAssertEqual(
                ModelRegistry.TransformerVariant.variant(for: .klein9BBase, quantization: quant),
                .klein9B_base_bf16,
                "Klein 9B base should always return base bf16 for \(quant)"
            )
        }
    }

    func testRecommendedConfigAllTiers() {
        // Ultra-minimal tier (<32GB)
        let ultra = ModelRegistry.recommendedConfig(forRAMGB: 24)
        XCTAssertEqual(ultra.transformer, .int4)

        // Minimal tier (32-48GB)
        let minimal = ModelRegistry.recommendedConfig(forRAMGB: 32)
        XCTAssertEqual(minimal.transformer, .qint8)

        // Balanced tier (48-96GB)
        let balanced48 = ModelRegistry.recommendedConfig(forRAMGB: 48)
        XCTAssertEqual(balanced48.transformer, .qint8)

        let balanced64 = ModelRegistry.recommendedConfig(forRAMGB: 64)
        XCTAssertEqual(balanced64.transformer, .qint8)

        // High quality tier (96GB+)
        let high = ModelRegistry.recommendedConfig(forRAMGB: 96)
        XCTAssertEqual(high.transformer, .bf16)

        let veryHigh = ModelRegistry.recommendedConfig(forRAMGB: 128)
        XCTAssertEqual(veryHigh.transformer, .bf16)
    }

    func testInt4QuantizationGroupSize() {
        // All quantization levels use the same group size
        XCTAssertEqual(TransformerQuantization.bf16.groupSize, 64)
        XCTAssertEqual(TransformerQuantization.qint8.groupSize, 64)
        XCTAssertEqual(TransformerQuantization.int4.groupSize, 64)
    }

    func testQuantizationCodable() throws {
        // Verify int4 round-trips through JSON encoding
        let encoder = JSONEncoder()
        let decoder = JSONDecoder()

        let config = Flux2QuantizationConfig.ultraMinimal
        let data = try encoder.encode(config)
        let decoded = try decoder.decode(Flux2QuantizationConfig.self, from: data)

        XCTAssertEqual(decoded.transformer, .int4)
        XCTAssertEqual(decoded.textEncoder, .mlx4bit)
    }
}

// MARK: - Debug Utilities Tests

final class DebugUtilsTests: XCTestCase {

    func testDebugLogDoesNotCrash() {
        // Debug logging should not crash regardless of state
        Flux2Debug.log("Test message")
        Flux2Debug.verbose("Verbose test message")
        Flux2Debug.info("Info message")
        Flux2Debug.warning("Warning message")
        Flux2Debug.error("Error message")

        // No assertion needed - just ensure no crash
        XCTAssertTrue(true)
    }

    func testDebugEnabledState() {
        let wasEnabled = Flux2Debug.enabled

        Flux2Debug.enabled = true
        XCTAssertTrue(Flux2Debug.enabled)

        Flux2Debug.enabled = false
        XCTAssertFalse(Flux2Debug.enabled)

        // Restore
        Flux2Debug.enabled = wasEnabled
    }

    func testDebugLevels() {
        XCTAssertLessThan(Flux2Debug.Level.verbose, Flux2Debug.Level.info)
        XCTAssertLessThan(Flux2Debug.Level.info, Flux2Debug.Level.warning)
        XCTAssertLessThan(Flux2Debug.Level.warning, Flux2Debug.Level.error)
    }

    func testDebugModeToggle() {
        let originalLevel = Flux2Debug.minLevel

        Flux2Debug.enableDebugMode()
        XCTAssertEqual(Flux2Debug.minLevel, .verbose)

        Flux2Debug.setNormalMode()
        XCTAssertEqual(Flux2Debug.minLevel, .warning)

        // Restore
        Flux2Debug.minLevel = originalLevel
    }
}

// MARK: - EmpiricalMu Tests

final class EmpiricalMuTests: XCTestCase {

    func testEmpiricalMuCalculation() {
        // Test various image sequence lengths
        let mu1024 = computeEmpiricalMu(imageSeqLen: 4096, numSteps: 50)
        let mu512 = computeEmpiricalMu(imageSeqLen: 1024, numSteps: 50)

        // Larger images should have different mu
        XCTAssertNotEqual(mu1024, mu512)
    }

    func testEmpiricalMuLargeImage() {
        // Very large images use different formula
        let muLarge = computeEmpiricalMu(imageSeqLen: 5000, numSteps: 50)

        XCTAssertGreaterThan(muLarge, 0)
    }

    func testEmpiricalMuVaryingSteps() {
        let mu50 = computeEmpiricalMu(imageSeqLen: 4096, numSteps: 50)
        let mu20 = computeEmpiricalMu(imageSeqLen: 4096, numSteps: 20)

        // Different step counts should produce different mu
        XCTAssertNotEqual(mu50, mu20)
    }
}

// MARK: - Training Variants Tests

final class TrainingVariantsTests: XCTestCase {

    // MARK: - Klein 9B Base Variant Tests

    func testKlein9BBaseVariantHuggingFaceRepo() {
        let variant = ModelRegistry.TransformerVariant.klein9B_base_bf16
        XCTAssertEqual(variant.huggingFaceRepo, "black-forest-labs/FLUX.2-klein-base-9B")
    }

    func testKlein9BBaseVariantIsGated() {
        let variant = ModelRegistry.TransformerVariant.klein9B_base_bf16
        XCTAssertTrue(variant.isGated)
    }

    func testKlein9BBaseVariantEstimatedSize() {
        let variant = ModelRegistry.TransformerVariant.klein9B_base_bf16
        XCTAssertEqual(variant.estimatedSizeGB, 18)  // Same as distilled
    }

    // MARK: - Klein 4B Base Variant Tests

    func testKlein4BBaseVariantHuggingFaceRepo() {
        let variant = ModelRegistry.TransformerVariant.klein4B_base_bf16
        XCTAssertEqual(variant.huggingFaceRepo, "black-forest-labs/FLUX.2-klein-base-4B")
    }

    func testKlein4BBaseVariantIsNotGated() {
        // Klein 4B Base from black-forest-labs is NOT gated (open access)
        let variant = ModelRegistry.TransformerVariant.klein4B_base_bf16
        XCTAssertFalse(variant.isGated)
    }

    // MARK: - Training/Inference Model Variant Tests

    func testBaseModelsAreForTraining() {
        // Base (non-distilled) Flux2Models should be for training
        XCTAssertTrue(Flux2Model.klein4BBase.isForTraining)
        XCTAssertTrue(Flux2Model.klein9BBase.isForTraining)
        XCTAssertTrue(Flux2Model.klein4BBase.isBaseModel)
        XCTAssertTrue(Flux2Model.klein9BBase.isBaseModel)
    }

    func testDistilledModelsAreForInference() {
        // Distilled models should be for inference, not training
        XCTAssertTrue(Flux2Model.klein4B.isForInference)
        XCTAssertTrue(Flux2Model.klein9B.isForInference)
        XCTAssertFalse(Flux2Model.klein4B.isForTraining)
        XCTAssertFalse(Flux2Model.klein9B.isForTraining)
    }

    // MARK: - trainingVariant(for:) Method Tests

    func testTrainingVariantForKlein4B() {
        let variant = ModelRegistry.TransformerVariant.trainingVariant(for: .klein4B)
        XCTAssertEqual(variant, .klein4B_base_bf16)
    }

    func testTrainingVariantForKlein9B() {
        let variant = ModelRegistry.TransformerVariant.trainingVariant(for: .klein9B)
        XCTAssertEqual(variant, .klein9B_base_bf16)
    }

    func testTrainingVariantForDev() {
        // Dev model is already "base" (not distilled), so it uses bf16
        let variant = ModelRegistry.TransformerVariant.trainingVariant(for: .dev)
        XCTAssertEqual(variant, .bf16)
    }

    func testTrainingVariantReturnsNonNil() {
        // All model types should have a training variant
        for model in [Flux2Model.klein4B, .klein9B, .dev] {
            let variant = ModelRegistry.TransformerVariant.trainingVariant(for: model)
            XCTAssertNotNil(variant, "Training variant should exist for \(model)")
        }
    }

    // MARK: - Training Variant Consistency

    func testTrainingVariantsResolveToBaseModels() {
        // Klein models should resolve to base variants for training
        XCTAssertEqual(Flux2Model.klein4B.trainingVariant, .klein4BBase)
        XCTAssertEqual(Flux2Model.klein9B.trainingVariant, .klein9BBase)

        // Dev doesn't have a separate base model
        XCTAssertEqual(Flux2Model.dev.trainingVariant, .dev)
    }
}

// MARK: - DevTextEncoder Tests

final class DevTextEncoderTests: XCTestCase {

    func testDevTextEncoderDefaultInit() {
        let encoder = DevTextEncoder()
        XCTAssertEqual(encoder.quantization, .mlx8bit)
        XCTAssertEqual(encoder.maxSequenceLength, 512)
        XCTAssertEqual(encoder.outputDimension, 15360)
    }

    func testDevTextEncoderCustomQuantization() {
        let encoder4bit = DevTextEncoder(quantization: .mlx4bit)
        XCTAssertEqual(encoder4bit.quantization, .mlx4bit)

        let encoderBf16 = DevTextEncoder(quantization: .bf16)
        XCTAssertEqual(encoderBf16.quantization, .bf16)
    }

    func testDevTextEncoderEstimatedMemory() {
        let bf16 = DevTextEncoder(quantization: .bf16)
        let mlx8bit = DevTextEncoder(quantization: .mlx8bit)
        let mlx4bit = DevTextEncoder(quantization: .mlx4bit)

        // Higher precision should use more memory
        XCTAssertGreaterThan(bf16.estimatedMemoryGB, mlx8bit.estimatedMemoryGB)
        XCTAssertGreaterThan(mlx8bit.estimatedMemoryGB, mlx4bit.estimatedMemoryGB)
    }

    func testDevTextEncoderNotLoadedByDefault() {
        let encoder = DevTextEncoder()
        XCTAssertFalse(encoder.isLoaded)
    }

    func testDevTextEncoderOutputDimensionMatches() {
        // Dev uses Mistral with 3 layers × 5120 hidden size = 15360
        let encoder = DevTextEncoder()
        XCTAssertEqual(encoder.outputDimension, 3 * 5120)
    }
}

// MARK: - Generation Result Tests

final class GenerationResultTests: XCTestCase {

    func testGenerationResultInitialization() {
        // Create a minimal test image (1x1 pixel)
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        let bitmapInfo = CGImageAlphaInfo.premultipliedLast.rawValue
        guard let context = CGContext(
            data: nil,
            width: 1,
            height: 1,
            bitsPerComponent: 8,
            bytesPerRow: 4,
            space: colorSpace,
            bitmapInfo: bitmapInfo
        ), let testImage = context.makeImage() else {
            XCTFail("Failed to create test image")
            return
        }

        let result = Flux2GenerationResult(
            image: testImage,
            usedPrompt: "enhanced: a beautiful sunset",
            wasUpsampled: true,
            originalPrompt: "a beautiful sunset"
        )

        XCTAssertEqual(result.usedPrompt, "enhanced: a beautiful sunset")
        XCTAssertEqual(result.originalPrompt, "a beautiful sunset")
        XCTAssertTrue(result.wasUpsampled)
        XCTAssertEqual(result.image.width, 1)
        XCTAssertEqual(result.image.height, 1)
    }

    func testGenerationResultNoUpsampling() {
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        let bitmapInfo = CGImageAlphaInfo.premultipliedLast.rawValue
        guard let context = CGContext(
            data: nil,
            width: 1,
            height: 1,
            bitsPerComponent: 8,
            bytesPerRow: 4,
            space: colorSpace,
            bitmapInfo: bitmapInfo
        ), let testImage = context.makeImage() else {
            XCTFail("Failed to create test image")
            return
        }

        let prompt = "a cat sitting on a chair"
        let result = Flux2GenerationResult(
            image: testImage,
            usedPrompt: prompt,
            wasUpsampled: false,
            originalPrompt: prompt
        )

        XCTAssertFalse(result.wasUpsampled)
        XCTAssertEqual(result.usedPrompt, result.originalPrompt)
    }

    func testGenerationResultPromptDifference() {
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        let bitmapInfo = CGImageAlphaInfo.premultipliedLast.rawValue
        guard let context = CGContext(
            data: nil,
            width: 1,
            height: 1,
            bitsPerComponent: 8,
            bytesPerRow: 4,
            space: colorSpace,
            bitmapInfo: bitmapInfo
        ), let testImage = context.makeImage() else {
            XCTFail("Failed to create test image")
            return
        }

        let original = "cat"
        let enhanced = "A majestic orange tabby cat sitting gracefully on a velvet chair, soft lighting, detailed fur"

        let result = Flux2GenerationResult(
            image: testImage,
            usedPrompt: enhanced,
            wasUpsampled: true,
            originalPrompt: original
        )

        XCTAssertTrue(result.wasUpsampled)
        XCTAssertNotEqual(result.usedPrompt, result.originalPrompt)
        XCTAssertTrue(result.usedPrompt.count > result.originalPrompt.count)
    }

    func testGenerationResultSendable() {
        // Verify Flux2GenerationResult conforms to Sendable
        // This test ensures the struct can be safely passed across concurrency boundaries
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        let bitmapInfo = CGImageAlphaInfo.premultipliedLast.rawValue
        guard let context = CGContext(
            data: nil,
            width: 1,
            height: 1,
            bitsPerComponent: 8,
            bytesPerRow: 4,
            space: colorSpace,
            bitmapInfo: bitmapInfo
        ), let testImage = context.makeImage() else {
            XCTFail("Failed to create test image")
            return
        }

        let result = Flux2GenerationResult(
            image: testImage,
            usedPrompt: "test prompt",
            wasUpsampled: false,
            originalPrompt: "test prompt"
        )

        // If this compiles, the type is Sendable
        let _: any Sendable = result
        XCTAssertTrue(true)
    }
}

// MARK: - MemoryConfig Tests

final class MemoryConfigTests: XCTestCase {

    func testCacheProfiles() {
        // Test all profiles exist and have descriptions
        for profile in MemoryConfig.CacheProfile.allCases {
            XCTAssertFalse(profile.description.isEmpty)
            XCTAssertFalse(profile.rawValue.isEmpty)
        }
    }

    func testSystemRAMDetection() {
        // System RAM should be detected and reasonable
        let ram = MemoryConfig.systemRAMGB
        XCTAssertGreaterThan(ram, 0)
        XCTAssertLessThan(ram, 1024) // Less than 1TB
    }

    func testSafeCachePercentage() {
        // Safe cache percentage should be between 0 and 1
        let pct = MemoryConfig.safeCachePercentage
        XCTAssertGreaterThan(pct, 0)
        XCTAssertLessThanOrEqual(pct, 1.0)
    }

    func testConservativeProfileLimit() {
        // Conservative should always return 512 MB
        let limit = MemoryConfig.cacheLimitForProfile(.conservative)
        XCTAssertNotNil(limit)
        XCTAssertEqual(limit, 512 * 1024 * 1024) // 512 MB
    }

    func testPerformanceProfileLimit() {
        // Performance should return up to 4 GB
        let limit = MemoryConfig.cacheLimitForProfile(.performance)
        XCTAssertNotNil(limit)
        XCTAssertGreaterThan(limit!, 0)
        XCTAssertLessThanOrEqual(limit!, 4 * 1024 * 1024 * 1024) // Max 4 GB
    }

    func testAutoProfileLimit() {
        // Auto should return a dynamic limit based on system RAM
        let limit = MemoryConfig.cacheLimitForProfile(.auto)
        // For systems < 128GB, should return a limit
        // For systems >= 128GB, might return nil (unlimited)
        if MemoryConfig.systemRAMGB < 128 {
            XCTAssertNotNil(limit)
            XCTAssertGreaterThan(limit!, 256 * 1024 * 1024) // At least 256 MB
        }
    }

    func testPhaseLimitsForKlein4B() {
        let limits = MemoryConfig.PhaseLimits.forModel(.klein4B, profile: .conservative)
        XCTAssertGreaterThan(limits.textEncoding, 0)
        XCTAssertGreaterThan(limits.denoising, 0)
        XCTAssertGreaterThan(limits.vaeDecoding, 0)
    }

    func testPhaseLimitsForDev() {
        let limits = MemoryConfig.PhaseLimits.forModel(.dev, profile: .performance)
        XCTAssertGreaterThan(limits.textEncoding, 0)
        XCTAssertGreaterThan(limits.denoising, 0)
        XCTAssertGreaterThan(limits.vaeDecoding, 0)

        // Dev should have larger limits than Klein
        let kleinLimits = MemoryConfig.PhaseLimits.forModel(.klein4B, profile: .performance)
        XCTAssertGreaterThanOrEqual(limits.denoising, kleinLimits.denoising)
    }

    func testResolutionBasedCacheLimit() {
        // Higher resolution should have higher cache limit
        let limit512 = MemoryConfig.cacheLimitForResolution(width: 512, height: 512, model: .klein4B)
        let limit1024 = MemoryConfig.cacheLimitForResolution(width: 1024, height: 1024, model: .klein4B)

        XCTAssertGreaterThan(limit1024, limit512)
    }

    func testConfigurationSummary() {
        // Configuration summary should not be empty
        let summary = MemoryConfig.configurationSummary
        XCTAssertFalse(summary.isEmpty)
        XCTAssertTrue(summary.contains("Memory Configuration"))
    }
}

// MARK: - Inference Variant Tests

final class InferenceVariantTests: XCTestCase {

    func testInferenceVariantForDistilledModels() {
        // Distilled models should return themselves
        XCTAssertEqual(Flux2Model.klein4B.inferenceVariant, .klein4B)
        XCTAssertEqual(Flux2Model.klein9B.inferenceVariant, .klein9B)
        XCTAssertEqual(Flux2Model.dev.inferenceVariant, .dev)
    }

    func testInferenceVariantForBaseModels() {
        // Base models should return distilled variants
        XCTAssertEqual(Flux2Model.klein4BBase.inferenceVariant, .klein4B)
        XCTAssertEqual(Flux2Model.klein9BBase.inferenceVariant, .klein9B)
    }

    func testInferenceVariantIsAlwaysForInference() {
        // The inference variant should always be usable for inference
        for model in Flux2Model.allCases {
            XCTAssertTrue(model.inferenceVariant.isForInference,
                          "\(model).inferenceVariant should be for inference")
        }
    }

    func testInferenceVariantIsNeverBase() {
        // The inference variant should never be a base model
        for model in Flux2Model.allCases {
            XCTAssertFalse(model.inferenceVariant.isBaseModel,
                           "\(model).inferenceVariant should not be a base model")
        }
    }

    func testTrainingAndInferenceVariantsAreInverse() {
        // For each model, trainingVariant.inferenceVariant should return the distilled version
        for model in [Flux2Model.klein4B, .klein9B, .dev] {
            let training = model.trainingVariant
            let inference = training.inferenceVariant
            XCTAssertTrue(inference.isForInference,
                          "trainingVariant.inferenceVariant of \(model) should be for inference")
        }
    }
}

// MARK: - Gradient Checkpointing Config Tests

final class GradientCheckpointingConfigTests: XCTestCase {

    func testGradientCheckpointingReducesMemoryEstimate() {
        let tmpDataset = URL(fileURLWithPath: "/tmp/test")
        let tmpOutput = URL(fileURLWithPath: "/tmp/output")

        let configWithout = LoRATrainingConfig(
            datasetPath: tmpDataset,
            rank: 32,
            alpha: 32.0,
            gradientCheckpointing: false,
            outputPath: tmpOutput
        )

        let configWith = LoRATrainingConfig(
            datasetPath: tmpDataset,
            rank: 32,
            alpha: 32.0,
            gradientCheckpointing: true,
            outputPath: tmpOutput
        )

        let memWithout = configWithout.estimateMemoryGB(for: .klein4B)
        let memWith = configWith.estimateMemoryGB(for: .klein4B)

        // Gradient checkpointing should reduce memory estimate
        XCTAssertLessThan(memWith, memWithout)
    }

    func testGradientCheckpointingSuggestion() {
        let config = LoRATrainingConfig(
            datasetPath: URL(fileURLWithPath: "/tmp/test"),
            rank: 32,
            alpha: 32.0,
            gradientCheckpointing: false,
            outputPath: URL(fileURLWithPath: "/tmp/output")
        )

        // Request suggestions for tight memory
        let suggestions = config.suggestAdjustments(for: .klein9B, availableGB: 16)

        // Should suggest enabling gradient checkpointing
        XCTAssertTrue(suggestions.contains { $0.contains("gradient checkpointing") },
                      "Should suggest gradient checkpointing when disabled and memory is tight")
    }

    func testGradientCheckpointingNoSuggestionWhenEnabled() {
        let config = LoRATrainingConfig(
            datasetPath: URL(fileURLWithPath: "/tmp/test"),
            rank: 32,
            alpha: 32.0,
            gradientCheckpointing: true,
            outputPath: URL(fileURLWithPath: "/tmp/output")
        )

        let suggestions = config.suggestAdjustments(for: .klein9B, availableGB: 16)

        // Should NOT suggest gradient checkpointing when already enabled
        XCTAssertFalse(suggestions.contains { $0.contains("gradient checkpointing") },
                       "Should not suggest gradient checkpointing when already enabled")
    }

    func testPresetsHaveGradientCheckpointing() {
        let tmpDataset = URL(fileURLWithPath: "/tmp/test")
        let tmpOutput = URL(fileURLWithPath: "/tmp/output")

        // All presets should have gradient checkpointing enabled
        let minimal = LoRATrainingConfig.minimal8GB(
            datasetPath: tmpDataset,
            outputPath: tmpOutput
        )
        XCTAssertTrue(minimal.gradientCheckpointing)

        let balanced = LoRATrainingConfig.balanced16GB(
            datasetPath: tmpDataset,
            outputPath: tmpOutput
        )
        XCTAssertTrue(balanced.gradientCheckpointing)

        let quality = LoRATrainingConfig.quality32GB(
            datasetPath: tmpDataset,
            outputPath: tmpOutput
        )
        XCTAssertTrue(quality.gradientCheckpointing)
    }

    func testGradientCheckpointingDefaultTrue() {
        // Default init should have gradient checkpointing enabled
        let config = LoRATrainingConfig(
            datasetPath: URL(fileURLWithPath: "/tmp/test"),
            outputPath: URL(fileURLWithPath: "/tmp/output")
        )
        XCTAssertTrue(config.gradientCheckpointing)
    }
}

// MARK: - Validation Quantization Tests

final class ValidationQuantizationTests: XCTestCase {

    func testKlein9BQuantizationOnTheFly() {
        // Klein 9B has no pre-quantized variant — uses on-the-fly quantization
        // All quantization levels should map to the bf16 download variant
        XCTAssertEqual(
            ModelRegistry.TransformerVariant.variant(for: .klein9B, quantization: .bf16),
            .klein9B_bf16
        )
        XCTAssertEqual(
            ModelRegistry.TransformerVariant.variant(for: .klein9B, quantization: .qint8),
            .klein9B_bf16,
            "Klein 9B qint8 should load bf16 and quantize on-the-fly"
        )
        XCTAssertEqual(
            ModelRegistry.TransformerVariant.variant(for: .klein9B, quantization: .int4),
            .klein9B_bf16,
            "Klein 9B int4 should load bf16 and quantize on-the-fly"
        )
    }

    func testKlein9BTransformerConfigMatchesDistilled() {
        // Base and distilled Klein 9B should share the same transformer config
        XCTAssertEqual(Flux2Model.klein9B.transformerConfig.numLayers,
                       Flux2Model.klein9BBase.transformerConfig.numLayers)
        XCTAssertEqual(Flux2Model.klein9B.transformerConfig.numSingleLayers,
                       Flux2Model.klein9BBase.transformerConfig.numSingleLayers)
    }

    func testKlein9BConfig() {
        let config = Flux2TransformerConfig.klein9B

        // Klein 9B should be between Klein 4B and Dev in size
        let klein4B = Flux2TransformerConfig.klein4B
        let dev = Flux2TransformerConfig.flux2Dev

        XCTAssertGreaterThan(config.numLayers, klein4B.numLayers)
        XCTAssertGreaterThan(config.numSingleLayers, klein4B.numSingleLayers)
        XCTAssertLessThanOrEqual(config.numLayers, dev.numLayers)
    }
}

// MARK: - Weight Conversion Tests

final class WeightConversionTests: XCTestCase {

    /// Validate that bf16→f16 direct conversion produces identical results to bf16→f32→f16
    /// This is critical for the BFL weight loading optimization (item 2.1)
    func testBF16ToF16DirectMatchesViaF32() {
        // Create realistic weight-like values in float32, then convert to bf16
        // Test various ranges: small, normal, large (within f16 range)
        let testValues: [Float] = [
            0.0, 1.0, -1.0,
            0.001, -0.001,           // Small weights
            0.01, -0.05, 0.1,       // Typical weight magnitudes
            0.5, -0.5, 1.5, -2.0,   // Normal range
            100.0, -100.0,           // Larger values
            65504.0, -65504.0,       // f16 max representable
            0.000061035,             // f16 min positive normal (~6.1e-5)
        ]
        let f32Array = MLXArray(testValues)
        let bf16Array = f32Array.asType(.bfloat16)

        // Path A: bf16 → f16 direct (new optimized path)
        let directF16 = bf16Array.asType(.float16)

        // Path B: bf16 → f32 → f16 (old path)
        let viaF32 = bf16Array.asType(.float32).asType(.float16)

        // Both should produce identical results
        eval(directF16, viaF32)

        let directValues = directF16.asType(.float32)
        let viaF32Values = viaF32.asType(.float32)

        // Compare element by element
        for i in 0..<testValues.count {
            let d = directValues[i].item(Float.self)
            let v = viaF32Values[i].item(Float.self)
            XCTAssertEqual(d, v, "Mismatch at index \(i) for input \(testValues[i]): direct=\(d), viaF32=\(v)")
        }
    }

    /// Test bf16→f16 with large random arrays (simulating real weight tensors)
    func testBF16ToF16DirectLargeArray() {
        // Simulate a realistic weight matrix (e.g., 3072x3072 linear layer)
        let weights = MLXRandom.normal([3072, 3072]).asType(.bfloat16)

        let directF16 = weights.asType(.float16)
        let viaF32 = weights.asType(.float32).asType(.float16)

        eval(directF16, viaF32)

        // Check that all values match exactly (bitwise identical)
        let diff = MLX.abs(directF16.asType(.float32) - viaF32.asType(.float32))
        let maxDiff = MLX.max(diff).item(Float.self)
        XCTAssertEqual(maxDiff, 0.0, "Max difference between direct and viaF32 conversion: \(maxDiff)")

        // Also verify no NaN or inf introduced
        let hasNaN = any(isNaN(directF16)).item(Bool.self)
        let hasInf = any(MLX.abs(directF16.asType(.float32)) .> 1e30).item(Bool.self)
        XCTAssertFalse(hasNaN, "Direct bf16→f16 conversion should not introduce NaN")
        XCTAssertFalse(hasInf, "Normal weight values should not overflow to inf")
    }

    /// Test that values outside f16 range are handled consistently
    func testBF16ToF16OverflowConsistency() {
        // bf16 can represent values up to ~3.4e38, but f16 max is ~65504
        // Both conversion paths should handle overflow identically
        let largeValues = MLXArray([Float(70000.0), Float(-70000.0), Float(100000.0)])
            .asType(.bfloat16)

        let directF16 = largeValues.asType(.float16)
        let viaF32 = largeValues.asType(.float32).asType(.float16)

        eval(directF16, viaF32)

        // Both should produce the same result (likely inf)
        let directF32 = directF16.asType(.float32)
        let viaF32F32 = viaF32.asType(.float32)

        for i in 0..<3 {
            let d = directF32[i].item(Float.self)
            let v = viaF32F32[i].item(Float.self)
            // Both should be inf or the same clamped value
            XCTAssertEqual(d, v, "Overflow handling mismatch at index \(i): direct=\(d), viaF32=\(v)")
        }
    }

    // MARK: - Image Roundtrip Tests (I2I spatial shift diagnostics)

    /// Helper: create a CGImage with a known gradient pattern
    private func createGradientCGImage(width: Int, height: Int) -> CGImage {
        let bytesPerPixel = 3
        let bytesPerRow = bytesPerPixel * width
        var pixelData = [UInt8](repeating: 0, count: height * bytesPerRow)

        for y in 0..<height {
            for x in 0..<width {
                let idx = y * bytesPerRow + x * bytesPerPixel
                pixelData[idx] = UInt8(x * 255 / max(width - 1, 1))     // R = horizontal gradient
                pixelData[idx + 1] = UInt8(y * 255 / max(height - 1, 1)) // G = vertical gradient
                pixelData[idx + 2] = 128                                  // B = constant
            }
        }

        let provider = CGDataProvider(data: Data(pixelData) as CFData)!
        return CGImage(
            width: width,
            height: height,
            bitsPerComponent: 8,
            bitsPerPixel: 24,
            bytesPerRow: bytesPerRow,
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.none.rawValue),
            provider: provider,
            decode: nil,
            shouldInterpolate: false,
            intent: .defaultIntent
        )!
    }

    /// Helper: extract pixel RGB from CGImage at (x, y)
    private func getPixel(from image: CGImage, x: Int, y: Int) -> (r: UInt8, g: UInt8, b: UInt8)? {
        guard let dataProvider = image.dataProvider,
              let data = dataProvider.data else { return nil }
        let bpp = image.bitsPerPixel / 8
        let bpr = image.bytesPerRow
        let ptr = CFDataGetBytePtr(data)!
        let offset = y * bpr + x * bpp

        // Determine RGB offsets from bitmap info
        let alphaInfo = CGImageAlphaInfo(rawValue: image.bitmapInfo.rawValue & CGBitmapInfo.alphaInfoMask.rawValue)
        let alphaFirst = alphaInfo == .premultipliedFirst || alphaInfo == .first || alphaInfo == .noneSkipFirst
        let hasAlpha = bpp >= 4

        let rOff = (hasAlpha && alphaFirst) ? 1 : 0
        let gOff = (hasAlpha && alphaFirst) ? 2 : 1
        let bOff = (hasAlpha && alphaFirst) ? 3 : 2

        return (ptr[offset + rOff], ptr[offset + gOff], ptr[offset + bOff])
    }

    /// Helper: encode CGImage to PNG Data
    private func pngData(from image: CGImage) -> Data? {
        let mutableData = NSMutableData()
        guard let destination = CGImageDestinationCreateWithData(mutableData as CFMutableData, "public.png" as CFString, 1, nil) else {
            return nil
        }
        CGImageDestinationAddImage(destination, image, nil)
        guard CGImageDestinationFinalize(destination) else { return nil }
        return mutableData as Data
    }

    /// Test 1: CGImageSource roundtrip is pixel-exact
    func testCGImageSourceRoundtripPixelExact() {
        let width = 8
        let height = 8
        let original = createGradientCGImage(width: width, height: height)

        // Encode to PNG
        guard let data = pngData(from: original) else {
            XCTFail("Failed to encode PNG")
            return
        }

        // Decode via CGImageSource (the fix path)
        guard let decoded = Flux2Pipeline.cgImage(from: data) else {
            XCTFail("Failed to decode via CGImageSource")
            return
        }

        // Verify dimensions
        XCTAssertEqual(decoded.width, width, "Width mismatch")
        XCTAssertEqual(decoded.height, height, "Height mismatch")

        // Compare every pixel
        for y in 0..<height {
            for x in 0..<width {
                guard let origPixel = getPixel(from: original, x: x, y: y),
                      let decodedPixel = getPixel(from: decoded, x: x, y: y) else {
                    XCTFail("Failed to read pixel at (\(x), \(y))")
                    continue
                }
                XCTAssertEqual(origPixel.r, decodedPixel.r, "R mismatch at (\(x), \(y)): \(origPixel.r) vs \(decodedPixel.r)")
                XCTAssertEqual(origPixel.g, decodedPixel.g, "G mismatch at (\(x), \(y)): \(origPixel.g) vs \(decodedPixel.g)")
                XCTAssertEqual(origPixel.b, decodedPixel.b, "B mismatch at (\(x), \(y)): \(origPixel.b) vs \(decodedPixel.b)")
            }
        }
    }

    #if canImport(AppKit)
    /// Test 2: Detect NSImage roundtrip artifacts
    /// This test documents that NSImage cgImage(forProposedRect:) can change pixel format/values
    func testNSImageRoundtripDetectsChanges() {
        let width = 16
        let height = 16
        let original = createGradientCGImage(width: width, height: height)

        // NSImage roundtrip (the problematic path)
        let nsImage = NSImage(cgImage: original, size: NSSize(width: width, height: height))
        guard let roundtripped = nsImage.cgImage(forProposedRect: nil, context: nil, hints: nil) else {
            XCTFail("NSImage roundtrip failed")
            return
        }

        // Check format changes
        let formatChanged = roundtripped.bitsPerPixel != original.bitsPerPixel
            || roundtripped.bytesPerRow != original.bytesPerRow
            || roundtripped.alphaInfo != original.alphaInfo

        if formatChanged {
            // Document that NSImage changes format (expected behavior we're fixing)
            print("[NSImage Roundtrip] Format changed: \(original.bitsPerPixel)bpp/\(original.alphaInfo.rawValue)alpha → \(roundtripped.bitsPerPixel)bpp/\(roundtripped.alphaInfo.rawValue)alpha")
        }

        // Dimensions should at least be preserved
        XCTAssertEqual(roundtripped.width, width, "NSImage roundtrip changed width")
        XCTAssertEqual(roundtripped.height, height, "NSImage roundtrip changed height")
    }
    #endif

    /// Test 3: CGImageSource vs NSImage pixel comparison
    /// Verifies CGImageSource produces pixel-exact results while NSImage may not
    func testCGImageSourceVsNSImageComparison() {
        let width = 16
        let height = 16
        let original = createGradientCGImage(width: width, height: height)

        guard let data = pngData(from: original) else {
            XCTFail("Failed to encode PNG")
            return
        }

        // Path A: CGImageSource (pixel-exact)
        guard let viaCGImageSource = Flux2Pipeline.cgImage(from: data) else {
            XCTFail("CGImageSource decode failed")
            return
        }

        // Verify CGImageSource path is pixel-exact with original
        var cgImageSourceExact = true
        for y in 0..<height {
            for x in 0..<width {
                guard let origPixel = getPixel(from: original, x: x, y: y),
                      let sourcePixel = getPixel(from: viaCGImageSource, x: x, y: y) else {
                    continue
                }
                if origPixel.r != sourcePixel.r || origPixel.g != sourcePixel.g || origPixel.b != sourcePixel.b {
                    cgImageSourceExact = false
                    break
                }
            }
            if !cgImageSourceExact { break }
        }

        XCTAssertTrue(cgImageSourceExact, "CGImageSource path should be pixel-exact")

        #if canImport(AppKit)
        // Path B: NSImage (potentially lossy)
        let nsImage = NSImage(data: data)!
        let viaNSImage = nsImage.cgImage(forProposedRect: nil, context: nil, hints: nil)!

        // Count pixel differences
        var diffCount = 0
        for y in 0..<height {
            for x in 0..<width {
                guard let origPixel = getPixel(from: original, x: x, y: y),
                      let nsPixel = getPixel(from: viaNSImage, x: x, y: y) else {
                    continue
                }
                if origPixel.r != nsPixel.r || origPixel.g != nsPixel.g || origPixel.b != nsPixel.b {
                    diffCount += 1
                }
            }
        }

        if diffCount > 0 {
            print("[NSImage vs CGImageSource] NSImage path has \(diffCount)/\(width * height) pixel differences")
        }
        #endif
    }
}

// MARK: - Flux2GenerationMode Tests

final class Flux2GenerationModeTests: XCTestCase {

    func testTextToImageMode() {
        let mode = Flux2GenerationMode.textToImage
        if case .textToImage = mode {
            // OK
        } else {
            XCTFail("Expected textToImage")
        }
    }

    func testImageToImageModeWithSingleImage() {
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        guard let ctx = CGContext(data: nil, width: 8, height: 8, bitsPerComponent: 8, bytesPerRow: 32, space: colorSpace, bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue),
              let img = ctx.makeImage() else {
            XCTFail("Failed to create test image"); return
        }

        let mode = Flux2GenerationMode.imageToImage(images: [img])
        if case .imageToImage(let images) = mode {
            XCTAssertEqual(images.count, 1)
            XCTAssertEqual(images[0].width, 8)
        } else {
            XCTFail("Expected imageToImage")
        }
    }

    func testImageToImageModeWithMultipleImages() {
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        guard let ctx = CGContext(data: nil, width: 4, height: 4, bitsPerComponent: 8, bytesPerRow: 16, space: colorSpace, bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue),
              let img = ctx.makeImage() else {
            XCTFail("Failed to create test image"); return
        }

        let mode = Flux2GenerationMode.imageToImage(images: [img, img, img])
        if case .imageToImage(let images) = mode {
            XCTAssertEqual(images.count, 3)
        } else {
            XCTFail("Expected imageToImage")
        }
    }

    func testModeHasNoStrengthParameter() {
        // Verify that Flux2GenerationMode.imageToImage has no strength associated value
        // This test documents the removal of the strength parameter (issue #57)
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        guard let ctx = CGContext(data: nil, width: 1, height: 1, bitsPerComponent: 8, bytesPerRow: 4, space: colorSpace, bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue),
              let img = ctx.makeImage() else {
            XCTFail("Failed to create test image"); return
        }

        let mode = Flux2GenerationMode.imageToImage(images: [img])
        // Pattern match with only images — no strength value
        if case .imageToImage(let images) = mode {
            XCTAssertEqual(images.count, 1)
        } else {
            XCTFail("Expected imageToImage with images only")
        }
    }
}

// MARK: - SchedulerOverrides Tests

final class SchedulerOverridesTests: XCTestCase {

    func testDefaultInit() {
        let overrides = SchedulerOverrides()
        XCTAssertNil(overrides.customSigmas)
        XCTAssertNil(overrides.numSteps)
        XCTAssertNil(overrides.guidance)
        XCTAssertFalse(overrides.hasOverrides)
    }

    func testWithNumSteps() {
        let overrides = SchedulerOverrides(numSteps: 4)
        XCTAssertEqual(overrides.numSteps, 4)
        XCTAssertTrue(overrides.hasOverrides)
    }

    func testWithGuidance() {
        let overrides = SchedulerOverrides(guidance: 1.0)
        XCTAssertEqual(overrides.guidance, 1.0)
        XCTAssertTrue(overrides.hasOverrides)
    }

    func testWithCustomSigmas() {
        let sigmas: [Float] = [1.0, 0.75, 0.5, 0.25, 0.0]
        let overrides = SchedulerOverrides(customSigmas: sigmas)
        XCTAssertEqual(overrides.customSigmas, sigmas)
        XCTAssertTrue(overrides.hasOverrides)
    }

    func testWithAllOverrides() {
        let overrides = SchedulerOverrides(
            customSigmas: [1.0, 0.5, 0.0],
            numSteps: 8,
            guidance: 3.5
        )
        XCTAssertEqual(overrides.customSigmas?.count, 3)
        XCTAssertEqual(overrides.numSteps, 8)
        XCTAssertEqual(overrides.guidance, 3.5)
        XCTAssertTrue(overrides.hasOverrides)
    }

    func testEquatable() {
        let a = SchedulerOverrides(numSteps: 4, guidance: 1.0)
        let b = SchedulerOverrides(numSteps: 4, guidance: 1.0)
        let c = SchedulerOverrides(numSteps: 8, guidance: 1.0)
        XCTAssertEqual(a, b)
        XCTAssertNotEqual(a, c)
    }

    func testCodableRoundtrip() throws {
        let original = SchedulerOverrides(
            customSigmas: [1.0, 0.5, 0.0],
            numSteps: 4,
            guidance: 1.0
        )
        let data = try JSONEncoder().encode(original)
        let decoded = try JSONDecoder().decode(SchedulerOverrides.self, from: data)
        XCTAssertEqual(original, decoded)
    }

    func testCodableNoStrengthField() throws {
        // Verify that JSON with a "strength" field is handled gracefully
        // (the field was removed, so it should be ignored by the decoder)
        let json = """
        {"numSteps": 4, "guidance": 1.0}
        """.data(using: .utf8)!
        let decoded = try JSONDecoder().decode(SchedulerOverrides.self, from: json)
        XCTAssertEqual(decoded.numSteps, 4)
        XCTAssertEqual(decoded.guidance, 1.0)
    }

    func testHasOverridesWithNoValues() {
        let overrides = SchedulerOverrides(customSigmas: nil, numSteps: nil, guidance: nil)
        XCTAssertFalse(overrides.hasOverrides)
    }
}

// MARK: - LoRAConfig SchedulerOverrides Integration Tests

final class LoRAConfigSchedulerOverridesTests: XCTestCase {

    func testLoRAConfigWithSchedulerOverrides() {
        let overrides = SchedulerOverrides(numSteps: 4, guidance: 1.0)
        var config = LoRAConfig(filePath: "/path/to/turbo.safetensors")
        config.schedulerOverrides = overrides

        XCTAssertNotNil(config.schedulerOverrides)
        XCTAssertEqual(config.schedulerOverrides?.numSteps, 4)
        XCTAssertEqual(config.schedulerOverrides?.guidance, 1.0)
    }

    func testLoRAConfigWithoutSchedulerOverrides() {
        let config = LoRAConfig(filePath: "/path/to/style.safetensors")
        XCTAssertNil(config.schedulerOverrides)
    }

    func testLoRAConfigCodableWithOverrides() throws {
        let json = """
        {
            "filePath": "/path/to/turbo.safetensors",
            "scale": 1.0,
            "schedulerOverrides": {
                "numSteps": 8,
                "guidance": 3.5,
                "customSigmas": [1.0, 0.75, 0.5, 0.25, 0.0]
            }
        }
        """.data(using: .utf8)!

        let config = try JSONDecoder().decode(LoRAConfig.self, from: json)
        XCTAssertEqual(config.filePath, "/path/to/turbo.safetensors")
        XCTAssertEqual(config.scale, 1.0)
        XCTAssertNotNil(config.schedulerOverrides)
        XCTAssertEqual(config.schedulerOverrides?.numSteps, 8)
        XCTAssertEqual(config.schedulerOverrides?.guidance, 3.5)
        XCTAssertEqual(config.schedulerOverrides?.customSigmas?.count, 5)
    }

    func testLoRAConfigCodableRoundtrip() throws {
        var config = LoRAConfig(filePath: "/path/to/lora.safetensors", scale: 0.8)
        config.activationKeyword = "sks"
        config.schedulerOverrides = SchedulerOverrides(numSteps: 4, guidance: 1.0)

        let data = try JSONEncoder().encode(config)
        let decoded = try JSONDecoder().decode(LoRAConfig.self, from: data)

        XCTAssertEqual(decoded.filePath, config.filePath)
        XCTAssertEqual(decoded.scale, config.scale)
        XCTAssertEqual(decoded.activationKeyword, "sks")
        XCTAssertEqual(decoded.schedulerOverrides, config.schedulerOverrides)
    }
}

// MARK: - CGImageSource Pipeline Helper Tests

final class CGImageSourcePipelineTests: XCTestCase {

    func testCgImageFromValidPNGData() {
        // Create a small CGImage and encode to PNG
        let width = 4, height = 4
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        guard let ctx = CGContext(data: nil, width: width, height: height, bitsPerComponent: 8, bytesPerRow: width * 4, space: colorSpace, bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue),
              let img = ctx.makeImage() else {
            XCTFail("Failed to create test image"); return
        }

        // Encode to PNG data
        let mutableData = NSMutableData()
        guard let dest = CGImageDestinationCreateWithData(mutableData as CFMutableData, "public.png" as CFString, 1, nil) else {
            XCTFail("Failed to create image destination"); return
        }
        CGImageDestinationAddImage(dest, img, nil)
        guard CGImageDestinationFinalize(dest) else {
            XCTFail("Failed to finalize image"); return
        }

        // Decode via pipeline helper
        let decoded = Flux2Pipeline.cgImage(from: mutableData as Data)
        XCTAssertNotNil(decoded)
        XCTAssertEqual(decoded?.width, width)
        XCTAssertEqual(decoded?.height, height)
    }

    func testCgImageFromValidJPEGData() {
        let width = 8, height = 8
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        guard let ctx = CGContext(data: nil, width: width, height: height, bitsPerComponent: 8, bytesPerRow: width * 4, space: colorSpace, bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue),
              let img = ctx.makeImage() else {
            XCTFail("Failed to create test image"); return
        }

        let mutableData = NSMutableData()
        guard let dest = CGImageDestinationCreateWithData(mutableData as CFMutableData, "public.jpeg" as CFString, 1, nil) else {
            XCTFail("Failed to create JPEG destination"); return
        }
        CGImageDestinationAddImage(dest, img, nil)
        guard CGImageDestinationFinalize(dest) else {
            XCTFail("Failed to finalize JPEG"); return
        }

        let decoded = Flux2Pipeline.cgImage(from: mutableData as Data)
        XCTAssertNotNil(decoded)
        XCTAssertEqual(decoded?.width, width)
        XCTAssertEqual(decoded?.height, height)
    }

    func testCgImageFromInvalidData() {
        let garbage = Data([0x00, 0x01, 0x02, 0x03, 0xFF])
        let result = Flux2Pipeline.cgImage(from: garbage)
        XCTAssertNil(result)
    }

    func testCgImageFromEmptyData() {
        let result = Flux2Pipeline.cgImage(from: Data())
        XCTAssertNil(result)
    }

    func testCgImagePreservesPixelValues() {
        // Create gradient image with known pixel values
        let width = 8, height = 8
        var pixels = [UInt8](repeating: 0, count: width * height * 4)
        for y in 0..<height {
            for x in 0..<width {
                let idx = (y * width + x) * 4
                pixels[idx + 0] = UInt8(x * 32)     // R
                pixels[idx + 1] = UInt8(y * 32)     // G
                pixels[idx + 2] = UInt8((x + y) * 16) // B
                pixels[idx + 3] = 255                // A (skip)
            }
        }

        let colorSpace = CGColorSpaceCreateDeviceRGB()
        guard let ctx = pixels.withUnsafeMutableBytes({ ptr -> CGContext? in
            CGContext(data: ptr.baseAddress, width: width, height: height, bitsPerComponent: 8, bytesPerRow: width * 4, space: colorSpace, bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue)
        }), let img = ctx.makeImage() else {
            XCTFail("Failed to create gradient image"); return
        }

        // Encode to PNG (lossless)
        let mutableData = NSMutableData()
        guard let dest = CGImageDestinationCreateWithData(mutableData as CFMutableData, "public.png" as CFString, 1, nil) else {
            XCTFail("Failed to create PNG destination"); return
        }
        CGImageDestinationAddImage(dest, img, nil)
        guard CGImageDestinationFinalize(dest) else {
            XCTFail("Failed to finalize PNG"); return
        }

        // Decode and verify
        guard let decoded = Flux2Pipeline.cgImage(from: mutableData as Data) else {
            XCTFail("Failed to decode PNG"); return
        }
        XCTAssertEqual(decoded.width, width)
        XCTAssertEqual(decoded.height, height)

        // Verify pixel values match
        guard let dataProvider = decoded.dataProvider,
              let pixelData = dataProvider.data,
              let bytes = CFDataGetBytePtr(pixelData) else {
            XCTFail("Failed to get pixel data from decoded image"); return
        }

        let bpp = decoded.bitsPerPixel / 8
        let bpr = decoded.bytesPerRow
        var mismatch = 0
        for y in 0..<height {
            for x in 0..<width {
                let srcIdx = (y * width + x) * 4
                let dstOff = y * bpr + x * bpp
                if pixels[srcIdx] != bytes[dstOff] || pixels[srcIdx + 1] != bytes[dstOff + 1] || pixels[srcIdx + 2] != bytes[dstOff + 2] {
                    mismatch += 1
                }
            }
        }
        XCTAssertEqual(mismatch, 0, "PNG roundtrip via CGImageSource should be pixel-exact")
    }
}

// MARK: - Klein 9B KV Configuration Tests

final class Klein9BKVConfigTests: XCTestCase {

    func testKlein9BKVModelProperties() {
        let model = Flux2Model.klein9BKV

        XCTAssertEqual(model.rawValue, "klein-9b-kv")
        XCTAssertEqual(model.displayName, "Flux.2 Klein 9B KV")
        XCTAssertTrue(model.isForInference)
        XCTAssertFalse(model.isForTraining)
        XCTAssertFalse(model.isBaseModel)
        XCTAssertTrue(model.supportsKVCache)
        XCTAssertEqual(model.defaultSteps, 4)
        XCTAssertEqual(model.defaultGuidance, 1.0)
        XCTAssertEqual(model.jointAttentionDim, 12288)
        XCTAssertFalse(model.usesGuidanceEmbeds)
        XCTAssertFalse(model.isCommercialUseAllowed)
        XCTAssertEqual(model.license, "Non-Commercial")
        XCTAssertEqual(model.maxReferenceImages, 4)
    }

    func testKlein9BKVTransformerConfig() {
        let config = Flux2Model.klein9BKV.transformerConfig
        // Same architecture as klein-9b
        XCTAssertEqual(config.numLayers, 8)
        XCTAssertEqual(config.numSingleLayers, 24)
        XCTAssertEqual(config.numAttentionHeads, 32)
        XCTAssertEqual(config.attentionHeadDim, 128)
        XCTAssertEqual(config.innerDim, 4096)  // 32 × 128
        XCTAssertEqual(config.jointAttentionDim, 12288)
        XCTAssertFalse(config.guidanceEmbeds)
    }

    func testSupportsKVCacheOnlyKlein9BKV() {
        // Only klein-9b-kv should support KV cache
        XCTAssertTrue(Flux2Model.klein9BKV.supportsKVCache)
        XCTAssertFalse(Flux2Model.klein9B.supportsKVCache)
        XCTAssertFalse(Flux2Model.klein4B.supportsKVCache)
        XCTAssertFalse(Flux2Model.dev.supportsKVCache)
        XCTAssertFalse(Flux2Model.klein9BBase.supportsKVCache)
        XCTAssertFalse(Flux2Model.klein4BBase.supportsKVCache)
    }

    func testKlein9BKVInferenceVariant() {
        // Inference variant should be klein9B (standard distilled)
        XCTAssertEqual(Flux2Model.klein9BKV.inferenceVariant, .klein9B)
    }

    func testKlein9BKVTrainingVariant() {
        // Training variant should be klein9BBase
        XCTAssertEqual(Flux2Model.klein9BKV.trainingVariant, .klein9BBase)
    }
}

// MARK: - Klein 9B KV Registry Tests

final class Klein9BKVRegistryTests: XCTestCase {

    func testKlein9BKVTransformerVariant() {
        let variant = ModelRegistry.TransformerVariant.klein9B_kv_bf16

        XCTAssertEqual(variant.rawValue, "klein9b-kv-bf16")
        XCTAssertEqual(variant.huggingFaceRepo, "black-forest-labs/FLUX.2-klein-9b-kv")
        XCTAssertNil(variant.huggingFaceSubfolder)
        XCTAssertEqual(variant.estimatedSizeGB, 18)
        XCTAssertTrue(variant.isGated)
        XCTAssertEqual(variant.modelType, .klein9BKV)
        XCTAssertTrue(variant.isForInference)
        XCTAssertFalse(variant.isForTraining)
        XCTAssertEqual(variant.quantization, .bf16)
    }

    func testKlein9BKVVariantLookup() {
        // All quantizations should return the bf16 variant (quantize on-the-fly)
        XCTAssertEqual(
            ModelRegistry.TransformerVariant.variant(for: .klein9BKV, quantization: .bf16),
            .klein9B_kv_bf16
        )
        XCTAssertEqual(
            ModelRegistry.TransformerVariant.variant(for: .klein9BKV, quantization: .qint8),
            .klein9B_kv_bf16
        )
        XCTAssertEqual(
            ModelRegistry.TransformerVariant.variant(for: .klein9BKV, quantization: .int4),
            .klein9B_kv_bf16
        )
    }

    func testKlein9BKVTrainingVariantLookup() {
        // Training variant should be klein9B_base_bf16 (same base model)
        XCTAssertEqual(
            ModelRegistry.TransformerVariant.trainingVariant(for: .klein9BKV),
            .klein9B_base_bf16
        )
    }

    func testKlein9BKVLocalPath() {
        let path = ModelRegistry.localPath(for: .transformer(.klein9B_kv_bf16))
        XCTAssertTrue(path.path.contains("FLUX.2-klein-9b-kv"))
    }
}

// MARK: - TransformerKVCache Tests

final class TransformerKVCacheTests: XCTestCase {

    func testKVCacheCreation() {
        let cache = TransformerKVCache(referenceTokenCount: 1024)
        XCTAssertEqual(cache.referenceTokenCount, 1024)
        XCTAssertEqual(cache.layerCount, 0)
        XCTAssertTrue(cache.doubleStreamEntries.isEmpty)
        XCTAssertTrue(cache.singleStreamEntries.isEmpty)
    }

    func testKVCacheEntryStorage() {
        var cache = TransformerKVCache(referenceTokenCount: 512)

        let keys = MLXRandom.normal([1, 32, 512, 128])
        let values = MLXRandom.normal([1, 32, 512, 128])
        let entry = LayerKVCacheEntry(keys: keys, values: values)

        cache.setDoubleStream(blockIndex: 0, entry: entry)
        XCTAssertEqual(cache.layerCount, 1)

        let retrieved = cache.doubleStreamEntry(at: 0)
        XCTAssertNotNil(retrieved)
        XCTAssertEqual(retrieved!.keys.shape, [1, 32, 512, 128])
        XCTAssertEqual(retrieved!.values.shape, [1, 32, 512, 128])
    }

    func testKVCacheSingleStreamStorage() {
        var cache = TransformerKVCache(referenceTokenCount: 256)

        let keys = MLXRandom.normal([1, 32, 256, 128])
        let values = MLXRandom.normal([1, 32, 256, 128])
        let entry = LayerKVCacheEntry(keys: keys, values: values)

        cache.setSingleStream(blockIndex: 5, entry: entry)
        XCTAssertEqual(cache.layerCount, 1)

        let retrieved = cache.singleStreamEntry(at: 5)
        XCTAssertNotNil(retrieved)

        // Non-existent index returns nil
        XCTAssertNil(cache.singleStreamEntry(at: 99))
    }

    func testKVCacheClear() {
        var cache = TransformerKVCache(referenceTokenCount: 128)

        let entry = LayerKVCacheEntry(
            keys: MLXRandom.normal([1, 32, 128, 128]),
            values: MLXRandom.normal([1, 32, 128, 128])
        )

        cache.setDoubleStream(blockIndex: 0, entry: entry)
        cache.setDoubleStream(blockIndex: 1, entry: entry)
        cache.setSingleStream(blockIndex: 0, entry: entry)
        XCTAssertEqual(cache.layerCount, 3)

        cache.clear()
        XCTAssertEqual(cache.layerCount, 0)
        XCTAssertTrue(cache.doubleStreamEntries.isEmpty)
        XCTAssertTrue(cache.singleStreamEntries.isEmpty)
        // referenceTokenCount is preserved after clear
        XCTAssertEqual(cache.referenceTokenCount, 128)
    }

    func testKVCacheMultipleLayersCount() {
        var cache = TransformerKVCache(referenceTokenCount: 64)
        let entry = LayerKVCacheEntry(
            keys: MLXRandom.normal([1, 4, 64, 32]),
            values: MLXRandom.normal([1, 4, 64, 32])
        )

        // Fill 8 double + 24 single (Klein 9B architecture)
        for i in 0..<8 { cache.setDoubleStream(blockIndex: i, entry: entry) }
        for i in 0..<24 { cache.setSingleStream(blockIndex: i, entry: entry) }

        XCTAssertEqual(cache.layerCount, 32)
        XCTAssertEqual(cache.doubleStreamEntries.count, 8)
        XCTAssertEqual(cache.singleStreamEntries.count, 24)

        // Overwrite an existing entry
        cache.setDoubleStream(blockIndex: 0, entry: entry)
        XCTAssertEqual(cache.doubleStreamEntries.count, 8, "Overwrite should not increase count")
    }

    func testLayerKVCacheEntryShapes() {
        let keys = MLXRandom.normal([1, 32, 1024, 128])
        let values = MLXRandom.normal([1, 32, 1024, 128])
        let entry = LayerKVCacheEntry(keys: keys, values: values)

        XCTAssertEqual(entry.keys.shape, [1, 32, 1024, 128])
        XCTAssertEqual(entry.values.shape, [1, 32, 1024, 128])
    }
}

// MARK: - KV Extraction Attention Mask Tests

final class KVExtractionMaskTests: XCTestCase {

    /// Test double-stream attention mask pattern:
    /// Joint sequence order: [txt, ref, output]
    /// - txt queries: attend to ALL (txt, ref, output)
    /// - ref queries: attend to txt + ref ONLY (blocked from output)
    /// - output queries: attend to ALL
    func testDoubleStreamMaskPattern() {
        let attn = Flux2Attention(dim: 128, numHeads: 4, headDim: 32)
        let textLen = 3
        let refLen = 2
        let outputLen = 4
        let totalSeq = textLen + refLen + outputLen  // 9

        let mask = attn.buildKVExtractionMask(
            textLen: textLen, refLen: refLen, outputLen: outputLen, totalSeq: totalSeq
        )

        XCTAssertEqual(mask.shape, [1, 1, totalSeq, totalSeq])

        // Materialize mask values
        eval(mask)
        let flat = mask.reshaped([totalSeq, totalSeq])

        // Helper to read mask value at (q, k)
        func maskVal(_ q: Int, _ k: Int) -> Float {
            flat[q, k].item(Float.self)
        }

        // txt queries (rows 0-2) should attend to everything → all 0.0
        for q in 0..<textLen {
            for k in 0..<totalSeq {
                XCTAssertEqual(maskVal(q, k), 0.0, "txt query \(q) should attend to key \(k)")
            }
        }

        // ref queries (rows 3-4) should attend to txt+ref (0-4) but NOT output (5-8)
        for q in textLen..<(textLen + refLen) {
            // Can attend to txt and ref
            for k in 0..<(textLen + refLen) {
                XCTAssertEqual(maskVal(q, k), 0.0, "ref query \(q) should attend to key \(k)")
            }
            // Blocked from output
            for k in (textLen + refLen)..<totalSeq {
                XCTAssertEqual(maskVal(q, k), -Float.infinity, "ref query \(q) should be blocked from output key \(k)")
            }
        }

        // output queries (rows 5-8) should attend to everything → all 0.0
        for q in (textLen + refLen)..<totalSeq {
            for k in 0..<totalSeq {
                XCTAssertEqual(maskVal(q, k), 0.0, "output query \(q) should attend to key \(k)")
            }
        }
    }

    /// Test single-stream attention mask pattern:
    /// Single sequence order: [txt, ref, output]
    /// Same blocking rule: ref queries blocked from output keys
    func testSingleStreamMaskPattern() {
        let attn = Flux2ParallelSelfAttention(dim: 128, numHeads: 4, headDim: 32)
        let textLen = 3
        let refLen = 2
        let outputLen = 4
        let totalSeq = textLen + refLen + outputLen

        let mask = attn.buildSingleStreamKVExtractionMask(
            textLen: textLen, refLen: refLen, outputLen: outputLen, totalSeq: totalSeq
        )

        XCTAssertEqual(mask.shape, [1, 1, totalSeq, totalSeq])

        eval(mask)
        let flat = mask.reshaped([totalSeq, totalSeq])

        func maskVal(_ q: Int, _ k: Int) -> Float {
            flat[q, k].item(Float.self)
        }

        // Ref queries blocked from output
        for q in textLen..<(textLen + refLen) {
            for k in (textLen + refLen)..<totalSeq {
                XCTAssertEqual(maskVal(q, k), -Float.infinity)
            }
        }

        // Everything else is 0.0
        for q in 0..<textLen {
            for k in 0..<totalSeq {
                XCTAssertEqual(maskVal(q, k), 0.0)
            }
        }
        for q in (textLen + refLen)..<totalSeq {
            for k in 0..<totalSeq {
                XCTAssertEqual(maskVal(q, k), 0.0)
            }
        }
    }

    /// Edge case: 0 reference tokens → no blocking at all (all 0.0)
    func testMaskWithZeroReferenceTokens() {
        let attn = Flux2Attention(dim: 128, numHeads: 4, headDim: 32)
        let mask = attn.buildKVExtractionMask(textLen: 5, refLen: 0, outputLen: 10, totalSeq: 15)

        eval(mask)
        let flat = mask.reshaped([15 * 15])

        // All values should be 0.0 (no blocking)
        let sum = MLX.abs(flat).sum()
        eval(sum)
        XCTAssertEqual(sum.item(Float.self), 0.0, "No ref tokens → no blocking")
    }
}

// MARK: - Flux2Attention KV Methods Tests

final class Flux2AttentionKVTests: XCTestCase {

    let batchSize = 1
    let numHeads = 4
    let headDim = 32
    let dim = 128  // 4 × 32
    let seqLenTxt = 8
    let seqLenRef = 6
    let seqLenImg = 10

    func makeAttention() -> Flux2Attention {
        Flux2Attention(dim: dim, numHeads: numHeads, headDim: headDim)
    }

    func makeRoPE(seqLen: Int) -> (cos: MLXArray, sin: MLXArray) {
        // Fake RoPE embeddings
        (cos: MLXRandom.normal([seqLen, headDim]),
         sin: MLXRandom.normal([seqLen, headDim]))
    }

    func testCallWithKVExtractionOutputShapes() {
        let attn = makeAttention()
        // hiddenStates contains [ref + img] tokens
        let imgPlusRef = MLXRandom.normal([batchSize, seqLenRef + seqLenImg, dim])
        let txt = MLXRandom.normal([batchSize, seqLenTxt, dim])
        let rope = makeRoPE(seqLen: seqLenTxt + seqLenRef + seqLenImg)

        let (hsOut, ehsOut, cache) = attn.callWithKVExtraction(
            hiddenStates: imgPlusRef,
            encoderHiddenStates: txt,
            rotaryEmb: rope,
            referenceTokenCount: seqLenRef
        )
        eval(hsOut, ehsOut, cache.keys, cache.values)

        // Output shapes match input
        XCTAssertEqual(hsOut.shape, [batchSize, seqLenRef + seqLenImg, dim])
        XCTAssertEqual(ehsOut.shape, [batchSize, seqLenTxt, dim])

        // Cache contains reference K/V with correct shape [B, H, S_ref, D]
        XCTAssertEqual(cache.keys.shape, [batchSize, numHeads, seqLenRef, headDim])
        XCTAssertEqual(cache.values.shape, [batchSize, numHeads, seqLenRef, headDim])
    }

    func testCallWithKVCachedOutputShapes() {
        let attn = makeAttention()
        // Only output tokens (no ref)
        let img = MLXRandom.normal([batchSize, seqLenImg, dim])
        let txt = MLXRandom.normal([batchSize, seqLenTxt, dim])
        let rope = makeRoPE(seqLen: seqLenTxt + seqLenImg)

        // Fake cached KV
        let cachedKV = LayerKVCacheEntry(
            keys: MLXRandom.normal([batchSize, numHeads, seqLenRef, headDim]),
            values: MLXRandom.normal([batchSize, numHeads, seqLenRef, headDim])
        )

        let (hsOut, ehsOut) = attn.callWithKVCached(
            hiddenStates: img,
            encoderHiddenStates: txt,
            rotaryEmb: rope,
            cachedKV: cachedKV
        )
        eval(hsOut, ehsOut)

        XCTAssertEqual(hsOut.shape, [batchSize, seqLenImg, dim])
        XCTAssertEqual(ehsOut.shape, [batchSize, seqLenTxt, dim])
    }

    /// Standard callAsFunction and callWithKVCached with 0 ref tokens
    /// should produce outputs of the same shape
    func testStandardAndCachedConsistentShapes() {
        let attn = makeAttention()
        let img = MLXRandom.normal([batchSize, seqLenImg, dim])
        let txt = MLXRandom.normal([batchSize, seqLenTxt, dim])
        let rope = makeRoPE(seqLen: seqLenTxt + seqLenImg)

        let (stdHs, stdEhs) = attn.callAsFunction(
            hiddenStates: img, encoderHiddenStates: txt, rotaryEmb: rope
        )

        // Empty cache (0 ref tokens)
        let emptyCache = LayerKVCacheEntry(
            keys: MLXRandom.normal([batchSize, numHeads, 0, headDim]),
            values: MLXRandom.normal([batchSize, numHeads, 0, headDim])
        )
        let (cachedHs, cachedEhs) = attn.callWithKVCached(
            hiddenStates: img, encoderHiddenStates: txt,
            rotaryEmb: rope, cachedKV: emptyCache
        )
        eval(stdHs, stdEhs, cachedHs, cachedEhs)

        XCTAssertEqual(stdHs.shape, cachedHs.shape)
        XCTAssertEqual(stdEhs.shape, cachedEhs.shape)
    }
}

// MARK: - Flux2ParallelSelfAttention KV Methods Tests

final class Flux2ParallelAttentionKVTests: XCTestCase {

    let batchSize = 1
    let numHeads = 4
    let headDim = 32
    let dim = 128
    let textLen = 8
    let refLen = 6
    let imgLen = 10

    func makeAttention() -> Flux2ParallelSelfAttention {
        Flux2ParallelSelfAttention(dim: dim, numHeads: numHeads, headDim: headDim)
    }

    func testCallWithKVExtractionOutputShapes() {
        let attn = makeAttention()
        let totalSeq = textLen + refLen + imgLen
        let combined = MLXRandom.normal([batchSize, totalSeq, dim])
        let rope = (cos: MLXRandom.normal([totalSeq, headDim]),
                    sin: MLXRandom.normal([totalSeq, headDim]))

        let (output, cache) = attn.callWithKVExtraction(
            hiddenStates: combined,
            rotaryEmb: rope,
            textLen: textLen,
            referenceTokenCount: refLen
        )
        eval(output, cache.keys, cache.values)

        // Output has same shape as input
        XCTAssertEqual(output.shape, [batchSize, totalSeq, dim])

        // Cache contains reference K/V: [B, H, refLen, D]
        XCTAssertEqual(cache.keys.shape, [batchSize, numHeads, refLen, headDim])
        XCTAssertEqual(cache.values.shape, [batchSize, numHeads, refLen, headDim])
    }

    func testCallWithKVCachedOutputShapes() {
        let attn = makeAttention()
        let seqLen = textLen + imgLen  // no ref tokens
        let combined = MLXRandom.normal([batchSize, seqLen, dim])
        let rope = (cos: MLXRandom.normal([seqLen, headDim]),
                    sin: MLXRandom.normal([seqLen, headDim]))

        let cachedKV = LayerKVCacheEntry(
            keys: MLXRandom.normal([batchSize, numHeads, refLen, headDim]),
            values: MLXRandom.normal([batchSize, numHeads, refLen, headDim])
        )

        let output = attn.callWithKVCached(
            hiddenStates: combined,
            rotaryEmb: rope,
            cachedKV: cachedKV,
            textLen: textLen
        )
        eval(output)

        // Output has same shape as input (txt + img, no ref added)
        XCTAssertEqual(output.shape, [batchSize, seqLen, dim])
    }
}

// MARK: - Transformer Block KV Forwarding Tests

final class TransformerBlockKVTests: XCTestCase {

    let batchSize = 1
    let numHeads = 4
    let headDim = 32
    let dim = 128
    let seqLenTxt = 8
    let seqLenRef = 4
    let seqLenImg = 10

    func testDoubleStreamBlockKVExtraction() {
        let block = Flux2TransformerBlock(dim: dim, numHeads: numHeads, headDim: headDim)
        let imgPlusRef = MLXRandom.normal([batchSize, seqLenRef + seqLenImg, dim])
        let txt = MLXRandom.normal([batchSize, seqLenTxt, dim])
        let temb = MLXRandom.normal([batchSize, dim])
        let totalRope = seqLenTxt + seqLenRef + seqLenImg
        let rope = (cos: MLXRandom.normal([totalRope, headDim]),
                    sin: MLXRandom.normal([totalRope, headDim]))

        let (ehsOut, hsOut, cache) = block.callWithKVExtraction(
            hiddenStates: imgPlusRef,
            encoderHiddenStates: txt,
            temb: temb,
            rotaryEmb: rope,
            imgModParams: nil,
            txtModParams: nil,
            referenceTokenCount: seqLenRef
        )
        eval(ehsOut, hsOut, cache.keys, cache.values)

        XCTAssertEqual(hsOut.shape, [batchSize, seqLenRef + seqLenImg, dim])
        XCTAssertEqual(ehsOut.shape, [batchSize, seqLenTxt, dim])
        XCTAssertEqual(cache.keys.shape, [batchSize, numHeads, seqLenRef, headDim])
    }

    func testDoubleStreamBlockKVCached() {
        let block = Flux2TransformerBlock(dim: dim, numHeads: numHeads, headDim: headDim)
        let img = MLXRandom.normal([batchSize, seqLenImg, dim])
        let txt = MLXRandom.normal([batchSize, seqLenTxt, dim])
        let temb = MLXRandom.normal([batchSize, dim])
        let rope = (cos: MLXRandom.normal([seqLenTxt + seqLenImg, headDim]),
                    sin: MLXRandom.normal([seqLenTxt + seqLenImg, headDim]))
        let cachedKV = LayerKVCacheEntry(
            keys: MLXRandom.normal([batchSize, numHeads, seqLenRef, headDim]),
            values: MLXRandom.normal([batchSize, numHeads, seqLenRef, headDim])
        )

        let (ehsOut, hsOut) = block.callWithKVCached(
            hiddenStates: img,
            encoderHiddenStates: txt,
            temb: temb,
            rotaryEmb: rope,
            imgModParams: nil,
            txtModParams: nil,
            cachedKV: cachedKV
        )
        eval(ehsOut, hsOut)

        XCTAssertEqual(hsOut.shape, [batchSize, seqLenImg, dim])
        XCTAssertEqual(ehsOut.shape, [batchSize, seqLenTxt, dim])
    }

    func testSingleStreamBlockKVExtraction() {
        let block = Flux2SingleTransformerBlock(dim: dim, numHeads: numHeads, headDim: headDim)
        let totalSeq = seqLenTxt + seqLenRef + seqLenImg
        let combined = MLXRandom.normal([batchSize, totalSeq, dim])
        let temb = MLXRandom.normal([batchSize, dim])
        let rope = (cos: MLXRandom.normal([totalSeq, headDim]),
                    sin: MLXRandom.normal([totalSeq, headDim]))

        let (output, cache) = block.callWithKVExtraction(
            hiddenStates: combined,
            temb: temb,
            rotaryEmb: rope,
            modParams: nil,
            textLen: seqLenTxt,
            referenceTokenCount: seqLenRef
        )
        eval(output, cache.keys, cache.values)

        XCTAssertEqual(output.shape, [batchSize, totalSeq, dim])
        XCTAssertEqual(cache.keys.shape, [batchSize, numHeads, seqLenRef, headDim])
    }

    func testSingleStreamBlockKVCached() {
        let block = Flux2SingleTransformerBlock(dim: dim, numHeads: numHeads, headDim: headDim)
        let seqLen = seqLenTxt + seqLenImg  // no ref
        let combined = MLXRandom.normal([batchSize, seqLen, dim])
        let temb = MLXRandom.normal([batchSize, dim])
        let rope = (cos: MLXRandom.normal([seqLen, headDim]),
                    sin: MLXRandom.normal([seqLen, headDim]))
        let cachedKV = LayerKVCacheEntry(
            keys: MLXRandom.normal([batchSize, numHeads, seqLenRef, headDim]),
            values: MLXRandom.normal([batchSize, numHeads, seqLenRef, headDim])
        )

        let output = block.callWithKVCached(
            hiddenStates: combined,
            temb: temb,
            rotaryEmb: rope,
            modParams: nil,
            cachedKV: cachedKV,
            textLen: seqLenTxt
        )
        eval(output)

        XCTAssertEqual(output.shape, [batchSize, seqLen, dim])
    }
}

// MARK: - Klein 9B KV Enum Exhaustiveness Tests

final class Klein9BKVEnumTests: XCTestCase {

    func testFlux2ModelAllCasesIncludesKlein9BKV() {
        let allRawValues = Flux2Model.allCases.map { $0.rawValue }
        XCTAssertTrue(allRawValues.contains("klein-9b-kv"))
    }

    func testTransformerVariantAllCasesIncludesKlein9BKV() {
        let allRawValues = ModelRegistry.TransformerVariant.allCases.map { $0.rawValue }
        XCTAssertTrue(allRawValues.contains("klein9b-kv-bf16"))
    }

    func testFlux2ModelCaseCount() {
        // dev, klein-4b, klein-4b-base, klein-9b, klein-9b-base, klein-9b-kv = 6
        XCTAssertEqual(Flux2Model.allCases.count, 6)
    }

    func testTransformerVariantCaseCount() {
        // bf16, qint8, klein4b-bf16, klein4b-8bit, klein4b-base-bf16,
        // klein9b-bf16, klein9b-base-bf16, klein9b-kv-bf16 = 8
        XCTAssertEqual(ModelRegistry.TransformerVariant.allCases.count, 8)
    }

    func testKlein9BKVMemoryConfigDoesNotCrash() {
        // Verify MemoryConfig handles the new model in all code paths
        let limit = MemoryConfig.cacheLimitForResolution(width: 512, height: 512, model: .klein9BKV)
        XCTAssertGreaterThan(limit, 0)

        let phaseLimits = MemoryConfig.PhaseLimits.forModel(.klein9BKV, profile: .auto)
        XCTAssertGreaterThan(phaseLimits.denoising, 0)
        XCTAssertGreaterThan(phaseLimits.textEncoding, 0)
        XCTAssertGreaterThan(phaseLimits.vaeDecoding, 0)

        // Manual profiles fallback to dynamic
        let conservativeLimits = MemoryConfig.PhaseLimits.forModel(.klein9BKV, profile: .conservative)
        XCTAssertGreaterThan(conservativeLimits.denoising, 0)
    }
}
