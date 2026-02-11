// Flux2CoreTests.swift - Unit tests for Flux2Core
// Copyright 2025 Vincent Gourbin

import XCTest
@testable import Flux2Core
import MLX

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
}
