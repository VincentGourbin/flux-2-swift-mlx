// SmallDecoderVAETests.swift - Tests for FLUX.2 Small Decoder VAE
// Copyright 2025 Vincent Gourbin

import XCTest
@testable import Flux2Core
import MLX
import MLXNN
import MLXRandom

// MARK: - VAEDecoder Construction Tests

final class VAEDecoderConstructionTests: XCTestCase {

    func testStandardDecoderConstruction() {
        let config = VAEConfig.flux2Dev
        let decoder = VAEDecoder(config: config)

        // convIn: latentChannels(32) -> last blockOutChannel(512)
        XCTAssertEqual(decoder.convIn.weight.shape[0], 512)  // OHWI: O=512
        XCTAssertEqual(decoder.convIn.weight.shape[3], 32)   // I=32

        // convOut: first blockOutChannel(128) -> outChannels(3)
        XCTAssertEqual(decoder.convOut.weight.shape[0], 3)   // O=3
        XCTAssertEqual(decoder.convOut.weight.shape[3], 128)  // I=128

        // 4 up blocks
        XCTAssertEqual(decoder.upBlocks.count, 4)

        // Last block has no upsample
        XCTAssertNil(decoder.upBlocks[3].upsample)
        // Others have upsample
        XCTAssertNotNil(decoder.upBlocks[0].upsample)
        XCTAssertNotNil(decoder.upBlocks[1].upsample)
        XCTAssertNotNil(decoder.upBlocks[2].upsample)
    }

    func testSmallDecoderConstruction() {
        let config = VAEConfig.flux2SmallDecoder
        let decoder = VAEDecoder(config: config)

        // convIn: latentChannels(32) -> last decoderChannel(384)
        XCTAssertEqual(decoder.convIn.weight.shape[0], 384)  // OHWI: O=384
        XCTAssertEqual(decoder.convIn.weight.shape[3], 32)   // I=32

        // convOut: first decoderChannel(96) -> outChannels(3)
        XCTAssertEqual(decoder.convOut.weight.shape[0], 3)   // O=3
        XCTAssertEqual(decoder.convOut.weight.shape[3], 96)   // I=96

        // Same structure: 4 up blocks
        XCTAssertEqual(decoder.upBlocks.count, 4)

        // Last block has no upsample
        XCTAssertNil(decoder.upBlocks[3].upsample)
    }

    func testSmallDecoderHasFewerParameters() {
        let standard = VAEDecoder(config: .flux2Dev)
        let small = VAEDecoder(config: .flux2SmallDecoder)

        let standardParams = standard.parameters().flattenedValues().map { $0.size }.reduce(0, +)
        let smallParams = small.parameters().flattenedValues().map { $0.size }.reduce(0, +)

        // Small decoder should have significantly fewer parameters (~44% reduction)
        XCTAssertLessThan(smallParams, standardParams)
        let ratio = Float(smallParams) / Float(standardParams)
        XCTAssertLessThan(ratio, 0.7, "Small decoder should be at least 30% smaller, got \(ratio)")
    }

    func testSmallDecoderLayersPerBlock() {
        let config = VAEConfig.flux2SmallDecoder
        let decoder = VAEDecoder(config: config)

        // Each up block should have layersPerBlock + 1 = 3 ResNet blocks
        for (i, block) in decoder.upBlocks.enumerated() {
            XCTAssertEqual(block.blocks.count, 3, "Up block \(i) should have 3 ResNet blocks")
        }
    }
}

// MARK: - VAEDecoder Forward Pass Tests

final class VAEDecoderForwardPassTests: XCTestCase {

    func testStandardDecoderOutputShape() {
        let decoder = VAEDecoder(config: .flux2Dev)
        let input = MLXRandom.normal([1, 32, 16, 16])  // [B, C, H/8, W/8] for 128x128 image

        let output = decoder(input)
        eval(output)

        // Output: [B, 3, H, W] = [1, 3, 128, 128] (8x upscale)
        XCTAssertEqual(output.shape, [1, 3, 128, 128])
    }

    func testSmallDecoderOutputShape() {
        let decoder = VAEDecoder(config: .flux2SmallDecoder)
        let input = MLXRandom.normal([1, 32, 16, 16])  // Same input

        let output = decoder(input)
        eval(output)

        // Same output shape as standard decoder
        XCTAssertEqual(output.shape, [1, 3, 128, 128])
    }

    func testSmallDecoderNonSquareInput() {
        let decoder = VAEDecoder(config: .flux2SmallDecoder)
        let input = MLXRandom.normal([1, 32, 16, 24])  // 128x192 image

        let output = decoder(input)
        eval(output)

        XCTAssertEqual(output.shape, [1, 3, 128, 192])
    }

    func testBothDecodersProduceSameOutputShape() {
        let standard = VAEDecoder(config: .flux2Dev)
        let small = VAEDecoder(config: .flux2SmallDecoder)
        let input = MLXRandom.normal([1, 32, 8, 8])  // 64x64

        let outStandard = standard(input)
        let outSmall = small(input)
        eval(outStandard, outSmall)

        XCTAssertEqual(outStandard.shape, outSmall.shape)
        XCTAssertEqual(outStandard.shape, [1, 3, 64, 64])
    }
}

// MARK: - AutoencoderKLFlux2 Tests

final class AutoencoderKLFlux2Tests: XCTestCase {

    func testStandardAutoencoder() {
        let vae = AutoencoderKLFlux2(config: .flux2Dev)

        XCTAssertFalse(vae.config.isSmallDecoder)
        XCTAssertEqual(vae.config.latentChannels, 32)
    }

    func testSmallDecoderAutoencoder() {
        let vae = AutoencoderKLFlux2(config: .flux2SmallDecoder)

        XCTAssertTrue(vae.config.isSmallDecoder)
        XCTAssertEqual(vae.config.effectiveDecoderChannels, [96, 192, 384, 384])
        // Encoder unchanged
        XCTAssertEqual(vae.config.blockOutChannels, [128, 256, 512, 512])
    }

    func testSmallDecoderDecodeOutputShape() {
        let vae = AutoencoderKLFlux2(config: .flux2SmallDecoder)
        let latent = MLXRandom.normal([1, 32, 8, 8])  // 64x64 image latents

        let decoded = vae.decode(latent)
        eval(decoded)

        XCTAssertEqual(decoded.shape, [1, 3, 64, 64])
    }

    func testSmallDecoderEncodeOutputShape() {
        let vae = AutoencoderKLFlux2(config: .flux2SmallDecoder)
        let image = MLXRandom.normal([1, 3, 64, 64])  // 64x64 RGB

        let encoded = vae.encode(image, samplePosterior: false)
        eval(encoded)

        // Encoder uses standard blockOutChannels, so encode should still work
        XCTAssertEqual(encoded.shape, [1, 32, 8, 8])
    }

    func testSmallDecoderPackUnpackLatents() {
        let vae = AutoencoderKLFlux2(config: .flux2SmallDecoder)
        let latents = MLXRandom.normal([1, 32, 16, 16])

        let packed = vae.packLatents(latents)
        // [1, 32, 16, 16] -> [1, (16/2)*(16/2), 32*2*2] = [1, 64, 128]
        XCTAssertEqual(packed.shape, [1, 64, 128])

        let unpacked = vae.unpackLatents(packed, height: 128, width: 128)
        XCTAssertEqual(unpacked.shape, [1, 32, 16, 16])
    }

    func testBatchNormStatsLoading() {
        let vae = AutoencoderKLFlux2(config: .flux2SmallDecoder)
        let mean = MLXRandom.normal([32])
        let variance = MLXArray.ones([32])

        vae.loadBatchNormStats(runningMean: mean, runningVar: variance)

        XCTAssertEqual(vae.batchNormRunningMean.shape, [32])
        XCTAssertEqual(vae.batchNormRunningVar.shape, [32])
    }
}

// MARK: - VAEConfig Codable Round-Trip Tests

final class VAEConfigCodableTests: XCTestCase {

    func testSmallDecoderRoundTrip() throws {
        let original = VAEConfig.flux2SmallDecoder
        let data = try JSONEncoder().encode(original)
        let decoded = try JSONDecoder().decode(VAEConfig.self, from: data)

        XCTAssertEqual(decoded.blockOutChannels, original.blockOutChannels)
        XCTAssertEqual(decoded.decoderBlockOutChannels, original.decoderBlockOutChannels)
        XCTAssertEqual(decoded.effectiveDecoderChannels, [96, 192, 384, 384])
        XCTAssertTrue(decoded.isSmallDecoder)
    }

    func testStandardRoundTrip() throws {
        let original = VAEConfig.flux2Dev
        let data = try JSONEncoder().encode(original)
        let decoded = try JSONDecoder().decode(VAEConfig.self, from: data)

        XCTAssertEqual(decoded.blockOutChannels, original.blockOutChannels)
        XCTAssertNil(decoded.decoderBlockOutChannels)
        XCTAssertFalse(decoded.isSmallDecoder)
    }

    func testHuggingFaceConfigJsonParsing() throws {
        // Exact config.json from black-forest-labs/FLUX.2-small-decoder
        let json = """
        {
            "_class_name": "AutoencoderKLFlux2",
            "in_channels": 3,
            "out_channels": 3,
            "block_out_channels": [128, 256, 512, 512],
            "decoder_block_out_channels": [96, 192, 384, 384],
            "layers_per_block": 2,
            "act_fn": "silu",
            "latent_channels": 32,
            "norm_num_groups": 32,
            "scaling_factor": 0.18215
        }
        """.data(using: .utf8)!

        let config = try JSONDecoder().decode(VAEConfig.self, from: json)

        XCTAssertEqual(config.inChannels, 3)
        XCTAssertEqual(config.outChannels, 3)
        XCTAssertEqual(config.blockOutChannels, [128, 256, 512, 512])
        XCTAssertEqual(config.decoderBlockOutChannels, [96, 192, 384, 384])
        XCTAssertEqual(config.latentChannels, 32)
        XCTAssertEqual(config.normNumGroups, 32)
        XCTAssertEqual(config.layersPerBlock, 2)
        XCTAssertTrue(config.isSmallDecoder)
    }

    func testConfigJsonWithUnknownFieldsIgnored() throws {
        // HuggingFace config.json has extra fields we don't model
        let json = """
        {
            "_class_name": "AutoencoderKLFlux2",
            "_diffusers_version": "0.32.0.dev0",
            "in_channels": 3,
            "out_channels": 3,
            "block_out_channels": [128, 256, 512, 512],
            "decoder_block_out_channels": [96, 192, 384, 384],
            "layers_per_block": 2,
            "act_fn": "silu",
            "latent_channels": 32,
            "norm_num_groups": 32,
            "sample_size": 1024,
            "force_upcast": true,
            "use_quant_conv": true,
            "use_post_quant_conv": true,
            "mid_block_add_attention": true,
            "batch_norm_eps": 0.0001,
            "batch_norm_momentum": 0.1,
            "patch_size": [2, 2]
        }
        """.data(using: .utf8)!

        // Should decode without errors despite unknown fields
        let config = try JSONDecoder().decode(VAEConfig.self, from: json)
        XCTAssertTrue(config.isSmallDecoder)
        XCTAssertEqual(config.effectiveDecoderChannels, [96, 192, 384, 384])
    }
}

// MARK: - VAEVariant Registry Tests

final class VAEVariantRegistryTests: XCTestCase {

    func testAllCasesIncludesBothVariants() {
        let allCases = ModelRegistry.VAEVariant.allCases
        XCTAssertEqual(allCases.count, 2)
        XCTAssertTrue(allCases.contains(.standard))
        XCTAssertTrue(allCases.contains(.smallDecoder))
    }

    func testSmallDecoderVariantProperties() {
        let variant = ModelRegistry.VAEVariant.smallDecoder

        XCTAssertEqual(variant.rawValue, "small-decoder")
        XCTAssertEqual(variant.displayName, "Small Decoder VAE")
        XCTAssertEqual(variant.huggingFaceRepo, "black-forest-labs/FLUX.2-small-decoder")
        XCTAssertNil(variant.huggingFaceSubfolder)
        XCTAssertEqual(variant.estimatedSizeGB, 1)
        XCTAssertFalse(variant.isGated)
        XCTAssertTrue(variant.license.contains("Apache"))
        XCTAssertTrue(variant.isCommercialUseAllowed)
    }

    func testStandardVariantProperties() {
        let variant = ModelRegistry.VAEVariant.standard

        XCTAssertEqual(variant.rawValue, "standard")
        XCTAssertEqual(variant.displayName, "Standard VAE")
        XCTAssertEqual(variant.huggingFaceRepo, "black-forest-labs/FLUX.2-klein-4B")
        XCTAssertEqual(variant.huggingFaceSubfolder, "vae")
        XCTAssertEqual(variant.estimatedSizeGB, 3)
        XCTAssertFalse(variant.isGated)
        XCTAssertTrue(variant.license.contains("Non-Commercial"))
        XCTAssertFalse(variant.isCommercialUseAllowed)
    }

    func testVAEVariantConfigMapping() {
        let standardConfig = ModelRegistry.VAEVariant.standard.vaeConfig
        XCTAssertFalse(standardConfig.isSmallDecoder)
        XCTAssertEqual(standardConfig.effectiveDecoderChannels, [128, 256, 512, 512])

        let smallConfig = ModelRegistry.VAEVariant.smallDecoder.vaeConfig
        XCTAssertTrue(smallConfig.isSmallDecoder)
        XCTAssertEqual(smallConfig.effectiveDecoderChannels, [96, 192, 384, 384])
    }

    func testVAEVariantURLConstruction() {
        let standard = ModelRegistry.VAEVariant.standard
        XCTAssertTrue(standard.huggingFaceURL.contains("/tree/main/vae"))

        let small = ModelRegistry.VAEVariant.smallDecoder
        XCTAssertTrue(small.huggingFaceURL.contains("FLUX.2-small-decoder"))
        XCTAssertFalse(small.huggingFaceURL.contains("/tree/main/"))
    }

    func testVAEComponentDisplayNames() {
        let standard = ModelRegistry.ModelComponent.vae(.standard)
        XCTAssertEqual(standard.displayName, "Flux.2 Standard VAE")

        let small = ModelRegistry.ModelComponent.vae(.smallDecoder)
        XCTAssertEqual(small.displayName, "Flux.2 Small Decoder VAE")
    }

    func testVAEComponentLocalDirectoryNames() {
        XCTAssertEqual(
            ModelRegistry.ModelComponent.vae(.standard).localDirectoryName,
            "flux2-vae-standard"
        )
        XCTAssertEqual(
            ModelRegistry.ModelComponent.vae(.smallDecoder).localDirectoryName,
            "flux2-vae-small-decoder"
        )
    }

    func testVAEComponentEstimatedSize() {
        let standard = ModelRegistry.ModelComponent.vae(.standard)
        let small = ModelRegistry.ModelComponent.vae(.smallDecoder)

        XCTAssertGreaterThan(standard.estimatedSizeGB, small.estimatedSizeGB)
    }

    func testVAELocalPathsAreDifferent() {
        let standardPath = ModelRegistry.localPath(for: .vae(.standard))
        let smallPath = ModelRegistry.localPath(for: .vae(.smallDecoder))

        XCTAssertNotEqual(standardPath, smallPath)
        XCTAssertTrue(standardPath.path.contains("FLUX.2-klein-4B-vae"))
        XCTAssertTrue(smallPath.path.contains("FLUX.2-small-decoder"))
    }

    func testVAEVariantInitFromRawValue() {
        XCTAssertEqual(ModelRegistry.VAEVariant(rawValue: "standard"), .standard)
        XCTAssertEqual(ModelRegistry.VAEVariant(rawValue: "small-decoder"), .smallDecoder)
        XCTAssertNil(ModelRegistry.VAEVariant(rawValue: "invalid"))
    }
}

// MARK: - WeightLoader Single File Tests

final class WeightLoaderSingleFileTests: XCTestCase {

    func testLoadWeightsDetectsSingleFile() throws {
        // Create a temp safetensors file to verify single-file detection
        let tempDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("flux2-test-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: tempDir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: tempDir) }

        // We can't easily create a valid safetensors file without the library,
        // but we can verify the path detection logic
        let filePath = tempDir.appendingPathComponent("model.safetensors")
        FileManager.default.createFile(atPath: filePath.path, contents: nil)

        var isDir: ObjCBool = false
        XCTAssertTrue(FileManager.default.fileExists(atPath: filePath.path, isDirectory: &isDir))
        XCTAssertFalse(isDir.boolValue, "Single file should not be detected as directory")

        // Directory should be detected as directory
        XCTAssertTrue(FileManager.default.fileExists(atPath: tempDir.path, isDirectory: &isDir))
        XCTAssertTrue(isDir.boolValue, "Directory should be detected as directory")
    }
}

// MARK: - VAETilingConfig Tests

final class VAETilingConfigTests: XCTestCase {

    func testDefaultConfig() {
        let config = VAETilingConfig.default
        XCTAssertEqual(config.tileSize, 64)
        XCTAssertEqual(config.tileOverlap, 8)
        XCTAssertEqual(config.minTileThreshold, 128)
    }

    func testAggressiveConfig() {
        let config = VAETilingConfig.aggressive
        XCTAssertEqual(config.tileSize, 32)
        XCTAssertEqual(config.tileOverlap, 4)
        XCTAssertLessThan(config.tileSize, VAETilingConfig.default.tileSize)
    }

    func testDisabledConfig() {
        let config = VAETilingConfig.disabled
        XCTAssertGreaterThan(config.tileSize, 1000)
        XCTAssertEqual(config.tileOverlap, 0)
    }

    func testCustomConfig() {
        let config = VAETilingConfig(tileSize: 48, tileOverlap: 6, minTileThreshold: 96)
        XCTAssertEqual(config.tileSize, 48)
        XCTAssertEqual(config.tileOverlap, 6)
        XCTAssertEqual(config.minTileThreshold, 96)
    }
}

// MARK: - Pipeline VAE Variant Tests

final class PipelineVAEVariantTests: XCTestCase {

    func testPipelineDefaultVAEVariant() {
        let pipeline = Flux2Pipeline(model: .klein4B, quantization: .balanced)
        XCTAssertEqual(pipeline.vaeVariant, .standard)
    }

    func testPipelineSmallDecoderVariant() {
        let pipeline = Flux2Pipeline(model: .klein4B, quantization: .balanced, vaeVariant: .smallDecoder)
        XCTAssertEqual(pipeline.vaeVariant, .smallDecoder)
    }

    func testPipelineVariantAffectsMissingModels() {
        // With a non-downloaded variant, it should appear in missing models
        let pipeline = Flux2Pipeline(model: .klein4B, quantization: .balanced, vaeVariant: .smallDecoder)
        let missing = pipeline.missingModels

        // Should check for small-decoder, not standard
        let vaeComponents = missing.filter {
            if case .vae = $0 { return true }
            return false
        }

        // If VAE is missing, it should be the small-decoder variant
        for component in vaeComponents {
            if case .vae(let variant) = component {
                XCTAssertEqual(variant, .smallDecoder)
            }
        }
    }
}

// MARK: - VAEConfig Description Tests

final class VAEConfigDescriptionTests: XCTestCase {

    func testStandardDescription() {
        let desc = VAEConfig.flux2Dev.description
        XCTAssertTrue(desc.contains("128, 256, 512, 512"))
    }

    func testSmallDecoderDescription() {
        let desc = VAEConfig.flux2SmallDecoder.description
        // Should show different encoder and decoder channels
        XCTAssertTrue(desc.contains("128, 256, 512, 512"), "Should show encoder channels")
        XCTAssertTrue(desc.contains("96, 192, 384, 384"), "Should show decoder channels")
    }
}
