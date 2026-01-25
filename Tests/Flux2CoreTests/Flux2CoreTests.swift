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
        XCTAssertEqual(scheduler.timesteps.first!, 1.0, accuracy: 0.01)
        XCTAssertEqual(scheduler.timesteps.last!, 0.0, accuracy: 0.01)
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

        let timestep = MLXArray([0.5])
        let guidance = MLXArray([4.0])

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
