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
        // Low RAM should recommend minimal config
        let lowRamConfig = ModelRegistry.recommendedConfig(forRAMGB: 32)
        XCTAssertEqual(lowRamConfig.transformer, .qint8)

        // High RAM can use bf16
        let highRamConfig = ModelRegistry.recommendedConfig(forRAMGB: 128)
        XCTAssertEqual(highRamConfig.transformer, .bf16)
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

        XCTAssertGreaterThan(bf16, qint8)
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
