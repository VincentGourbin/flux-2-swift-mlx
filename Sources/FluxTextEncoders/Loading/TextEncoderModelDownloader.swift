/**
 * TextEncoderModelDownloader.swift
 * Downloads Mistral and Qwen3 models from HuggingFace Hub for text encoding
 */

import Foundation
import Hub

/// Progress callback for download updates
public typealias TextEncoderDownloadProgressCallback = @Sendable (Double, String) -> Void

/// Model downloader with HuggingFace Hub integration
public class TextEncoderModelDownloader {

    /// HuggingFace token for private/gated models
    private var hfToken: String?

    /// Hub API instance
    nonisolated(unsafe) private static var hubApi: HubApi = {
        setenv("CI_DISABLE_NETWORK_MONITOR", "1", 1)
        return HubApi(
            downloadBase: FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask).first
        )
    }()

    /// Default models directory
    public static var modelsDirectory: URL {
        let homeDir = FileManager.default.homeDirectoryForCurrentUser
        return homeDir.appendingPathComponent(".mistral").appendingPathComponent("models")
    }

    public init(hfToken: String? = nil) {
        self.hfToken = hfToken
        if let token = hfToken {
            setenv("HF_TOKEN", token, 1)
        }
    }

    /// Check if a model is already downloaded
    public static func isModelDownloaded(_ model: ModelInfo) -> Bool {
        return findModelPath(for: model) != nil
    }

    /// Get the HuggingFace Hub cache path for a model
    public static func hubCachePath(for model: ModelInfo) -> URL? {
        // Check new location: ~/Library/Caches/models/{org}/{repo}
        if let cacheDir = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask).first {
            // Split repoId (e.g., "lmstudio-community/Model-Name") and append each part
            // to avoid URL encoding of the slash
            var newPath = cacheDir.appendingPathComponent("models")
            for component in model.repoId.split(separator: "/") {
                newPath = newPath.appendingPathComponent(String(component))
            }

            if FileManager.default.fileExists(atPath: newPath.appendingPathComponent("config.json").path) {
                return newPath
            }
        }

        // Check legacy location: ~/.cache/huggingface/hub/models--{org}--{repo}/snapshots/...
        let homeDir = FileManager.default.homeDirectoryForCurrentUser
        let hubCache = homeDir
            .appendingPathComponent(".cache")
            .appendingPathComponent("huggingface")
            .appendingPathComponent("hub")

        let modelFolder = "models--\(model.repoId.replacingOccurrences(of: "/", with: "--"))"
        let snapshotsDir = hubCache.appendingPathComponent(modelFolder).appendingPathComponent("snapshots")

        guard let contents = try? FileManager.default.contentsOfDirectory(atPath: snapshotsDir.path),
              let latestSnapshot = contents.sorted().last else {
            return nil
        }

        let modelPath = snapshotsDir.appendingPathComponent(latestSnapshot)
        let configPath = modelPath.appendingPathComponent("config.json")

        if FileManager.default.fileExists(atPath: configPath.path) {
            return modelPath
        }

        return nil
    }

    /// Find a model path (checks Hub cache first, then local directory)
    public static func findModelPath(for model: ModelInfo) -> URL? {
        // Check Hub cache first
        if let hubPath = hubCachePath(for: model) {
            let verification = verifyShardedModel(at: hubPath)
            if verification.complete {
                return hubPath
            }
        }

        // Check local models directory
        let localDir = modelsDirectory.appendingPathComponent(model.repoId.replacingOccurrences(of: "/", with: "--"))
        if FileManager.default.fileExists(atPath: localDir.appendingPathComponent("config.json").path) {
            let verification = verifyShardedModel(at: localDir)
            if verification.complete {
                return localDir
            }
        }

        return nil
    }

    /// Verify that a sharded model has all required safetensors files
    /// Note: Does NOT trust index.json as some HF repos have mismatched index files
    /// Instead, detects safetensors files and verifies the series is complete
    public static func verifyShardedModel(at path: URL) -> (complete: Bool, missing: [String]) {
        let contents = (try? FileManager.default.contentsOfDirectory(atPath: path.path)) ?? []
        let safetensorsFiles = contents.filter { $0.hasSuffix(".safetensors") }

        // Single file model
        if safetensorsFiles.contains("model.safetensors") {
            return (true, [])
        }

        // No safetensors files at all
        guard !safetensorsFiles.isEmpty else {
            return (false, ["No safetensors files found"])
        }

        // Parse sharded file pattern: model-XXXXX-of-YYYYY.safetensors
        // Example: model-00001-of-00003.safetensors
        var totalShards: Int?
        var foundIndices: Set<Int> = []

        for file in safetensorsFiles {
            // Parse filename like "model-00001-of-00003.safetensors"
            let name = file.replacingOccurrences(of: ".safetensors", with: "")
            let parts = name.split(separator: "-")
            // Expected: ["model", "00001", "of", "00003"]
            guard parts.count == 4,
                  parts[0] == "model",
                  parts[2] == "of",
                  let index = Int(parts[1]),
                  let total = Int(parts[3]) else {
                continue
            }

            if totalShards == nil {
                totalShards = total
            } else if totalShards != total {
                // Inconsistent totals - mixed files
                return (false, ["Inconsistent shard totals: \(totalShards!) vs \(total)"])
            }

            foundIndices.insert(index)
        }

        // If we found sharded files, verify all parts are present
        if let total = totalShards {
            let expectedIndices = Set(1...total)
            let missing = expectedIndices.subtracting(foundIndices)

            if missing.isEmpty {
                return (true, [])
            } else {
                let missingFiles = missing.sorted().map { "model-\(String(format: "%05d", $0))-of-\(String(format: "%05d", total)).safetensors" }
                return (false, missingFiles)
            }
        }

        // Has some safetensors files but not in standard sharded format
        // Consider it complete if there are any safetensors files
        return (true, [])
    }

    /// Download a model using Hub API
    public func download(
        _ model: ModelInfo,
        progress: TextEncoderDownloadProgressCallback? = nil
    ) async throws -> URL {
        // Check if already downloaded
        if let existingPath = Self.findModelPath(for: model) {
            let verification = Self.verifyShardedModel(at: existingPath)
            if verification.complete {
                // Also ensure tekken.json exists
                await ensureTekkenJson(at: existingPath, progress: progress)
                progress?(1.0, "Model already downloaded")
                return existingPath
            } else {
                print("Warning: Incomplete download detected. Missing files: \(verification.missing)")
                print("Re-downloading...")
            }
        }

        progress?(0.0, "Starting download of \(model.name)...")
        print("\nDownloading \(model.name) from HuggingFace...")
        print("Repository: \(model.repoId)")
        print()

        let modelUrl = try await Self.hubApi.snapshot(
            from: model.repoId,
            matching: ["*.json", "*.safetensors"]
        ) { downloadProgress in
            let completed = downloadProgress.completedUnitCount
            let total = downloadProgress.totalUnitCount
            let fraction = downloadProgress.fractionCompleted

            let message = "Downloading file \(completed)/\(total)"
            progress?(fraction, message)
        }

        let verification = Self.verifyShardedModel(at: modelUrl)
        if !verification.complete {
            print("\nWarning: Download may be incomplete. Missing files: \(verification.missing)")
        }

        // Download tekken.json from original Mistral repo if not present
        await ensureTekkenJson(at: modelUrl, progress: progress)

        progress?(1.0, "Download complete!")
        print("\nDownload complete: \(modelUrl.path)")

        return modelUrl
    }

    /// Ensure tekken.json exists in the model directory
    /// Downloads from original Mistral repo if not present
    private func ensureTekkenJson(at modelPath: URL, progress: TextEncoderDownloadProgressCallback? = nil) async {
        let tekkenPath = modelPath.appendingPathComponent("tekken.json")

        // Check if already exists
        if FileManager.default.fileExists(atPath: tekkenPath.path) {
            // Verify it's not a Git LFS pointer
            if let data = try? Data(contentsOf: tekkenPath),
               data.count > 1000 {  // Real file is ~19MB, pointer is < 200 bytes
                return
            }
        }

        progress?(0.9, "Downloading tekken.json tokenizer...")
        print("Downloading tekken.json from Mistral AI repository...")

        let tekkenUrl = URL(string: "https://huggingface.co/mistralai/Mistral-Small-3.2-24B-Instruct-2506/resolve/main/tekken.json")!

        do {
            let (data, response) = try await URLSession.shared.data(from: tekkenUrl)

            if let httpResponse = response as? HTTPURLResponse,
               httpResponse.statusCode == 200,
               data.count > 1000 {  // Sanity check - real file is ~19MB
                try data.write(to: tekkenPath)
                print("tekken.json downloaded successfully (\(data.count / 1_000_000)MB)")
            } else {
                print("Warning: Failed to download tekken.json - response invalid")
            }
        } catch {
            print("Warning: Could not download tekken.json: \(error.localizedDescription)")
        }
    }

    /// Download a model by variant
    public func download(
        variant: ModelVariant,
        progress: TextEncoderDownloadProgressCallback? = nil
    ) async throws -> URL {
        guard let model = await TextEncoderModelRegistry.shared.model(withVariant: variant) else {
            throw TextEncoderModelDownloaderError.modelNotFound
        }
        return try await download(model, progress: progress)
    }

    // MARK: - Qwen3 Downloads

    /// Download a Qwen3 model for Klein embeddings
    public func downloadQwen3(
        _ model: Qwen3ModelInfo,
        progress: TextEncoderDownloadProgressCallback? = nil
    ) async throws -> URL {
        // Check if already downloaded
        if let existingPath = Self.findQwen3ModelPath(for: model) {
            let verification = Self.verifyShardedModel(at: existingPath)
            if verification.complete {
                progress?(1.0, "Qwen3 model already downloaded")
                return existingPath
            } else {
                print("Warning: Incomplete Qwen3 download detected. Missing files: \(verification.missing)")
                print("Re-downloading...")
            }
        }

        progress?(0.0, "Starting download of \(model.name)...")
        print("\nDownloading \(model.name) from HuggingFace...")
        print("Repository: \(model.repoId)")
        print()

        let modelUrl = try await Self.hubApi.snapshot(
            from: model.repoId,
            matching: ["*.json", "*.safetensors", "tokenizer.model"]
        ) { downloadProgress in
            let completed = downloadProgress.completedUnitCount
            let total = downloadProgress.totalUnitCount
            let fraction = downloadProgress.fractionCompleted

            let message = "Downloading file \(completed)/\(total)"
            progress?(fraction, message)
        }

        let verification = Self.verifyShardedModel(at: modelUrl)
        if !verification.complete {
            print("\nWarning: Qwen3 download may be incomplete. Missing files: \(verification.missing)")
        }

        progress?(1.0, "Download complete!")
        print("\nQwen3 download complete: \(modelUrl.path)")

        return modelUrl
    }

    /// Download a Qwen3 model by variant
    public func downloadQwen3(
        variant: Qwen3Variant,
        progress: TextEncoderDownloadProgressCallback? = nil
    ) async throws -> URL {
        guard let model = await TextEncoderModelRegistry.shared.qwen3Model(withVariant: variant) else {
            throw TextEncoderModelDownloaderError.qwen3ModelNotFound
        }
        return try await downloadQwen3(model, progress: progress)
    }

    /// Find a Qwen3 model path (checks Hub cache)
    public static func findQwen3ModelPath(for model: Qwen3ModelInfo) -> URL? {
        // Check new location: ~/Library/Caches/models/{org}/{repo}
        if let cacheDir = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask).first {
            var newPath = cacheDir.appendingPathComponent("models")
            for component in model.repoId.split(separator: "/") {
                newPath = newPath.appendingPathComponent(String(component))
            }

            if FileManager.default.fileExists(atPath: newPath.appendingPathComponent("config.json").path) {
                return newPath
            }
        }

        // Check legacy location
        let homeDir = FileManager.default.homeDirectoryForCurrentUser
        let hubCache = homeDir
            .appendingPathComponent(".cache")
            .appendingPathComponent("huggingface")
            .appendingPathComponent("hub")

        let modelFolder = "models--\(model.repoId.replacingOccurrences(of: "/", with: "--"))"
        let snapshotsDir = hubCache.appendingPathComponent(modelFolder).appendingPathComponent("snapshots")

        guard let contents = try? FileManager.default.contentsOfDirectory(atPath: snapshotsDir.path),
              let latestSnapshot = contents.sorted().last else {
            return nil
        }

        let modelPath = snapshotsDir.appendingPathComponent(latestSnapshot)
        let configPath = modelPath.appendingPathComponent("config.json")

        if FileManager.default.fileExists(atPath: configPath.path) {
            return modelPath
        }

        return nil
    }

    /// Check if a Qwen3 model is already downloaded
    public static func isQwen3ModelDownloaded(_ model: Qwen3ModelInfo) -> Bool {
        return findQwen3ModelPath(for: model) != nil
    }

    /// Find a Qwen3 model path by variant
    public static func findQwen3ModelPath(for variant: Qwen3Variant) -> URL? {
        let repoId = variant.repoId

        // Check new location: ~/Library/Caches/models/{org}/{repo}
        if let cacheDir = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask).first {
            var newPath = cacheDir.appendingPathComponent("models")
            for component in repoId.split(separator: "/") {
                newPath = newPath.appendingPathComponent(String(component))
            }

            if FileManager.default.fileExists(atPath: newPath.appendingPathComponent("config.json").path) {
                return newPath
            }
        }

        // Check legacy location
        let homeDir = FileManager.default.homeDirectoryForCurrentUser
        let hubCache = homeDir
            .appendingPathComponent(".cache")
            .appendingPathComponent("huggingface")
            .appendingPathComponent("hub")

        let modelFolder = "models--\(repoId.replacingOccurrences(of: "/", with: "--"))"
        let snapshotsDir = hubCache.appendingPathComponent(modelFolder).appendingPathComponent("snapshots")

        guard let contents = try? FileManager.default.contentsOfDirectory(atPath: snapshotsDir.path),
              let latestSnapshot = contents.sorted().last else {
            return nil
        }

        let modelPath = snapshotsDir.appendingPathComponent(latestSnapshot)
        let configPath = modelPath.appendingPathComponent("config.json")

        if FileManager.default.fileExists(atPath: configPath.path) {
            return modelPath
        }

        return nil
    }

    /// Check if a Qwen3 model is already downloaded by variant
    public static func isQwen3ModelDownloaded(variant: Qwen3Variant) -> Bool {
        return findQwen3ModelPath(for: variant) != nil
    }

    /// Download a model by repo ID directly
    public func downloadByRepoId(
        _ repoId: String,
        progress: TextEncoderDownloadProgressCallback? = nil
    ) async throws -> URL {
        progress?(0.0, "Starting download...")
        print("\nDownloading from HuggingFace: \(repoId)")

        let modelUrl = try await Self.hubApi.snapshot(
            from: repoId,
            matching: ["*.json", "*.safetensors"]
        )

        progress?(1.0, "Download complete!")
        print("Model available at: \(modelUrl.path)")

        return modelUrl
    }

    /// Resolve a model identifier to a local path, downloading if necessary
    public func resolveModel(
        _ identifier: String,
        progress: TextEncoderDownloadProgressCallback? = nil
    ) async throws -> URL {
        // Try to find by ID
        if let model = await TextEncoderModelRegistry.shared.model(withId: identifier) {
            if let existingPath = Self.findModelPath(for: model) {
                return existingPath
            }
            return try await download(model, progress: progress)
        }

        // Try variant matching
        if let variant = ModelVariant(rawValue: identifier),
           let model = await TextEncoderModelRegistry.shared.model(withVariant: variant) {
            if let existingPath = Self.findModelPath(for: model) {
                return existingPath
            }
            return try await download(model, progress: progress)
        }

        // Check if it's a local path
        let localURL = URL(fileURLWithPath: identifier)
        if FileManager.default.fileExists(atPath: localURL.appendingPathComponent("config.json").path) {
            return localURL
        }

        // Try as a direct HuggingFace repo ID
        return try await downloadByRepoId(identifier, progress: progress)
    }

    /// Format bytes as human-readable string
    public static func formatSize(_ bytes: Int64) -> String {
        let formatter = ByteCountFormatter()
        formatter.allowedUnits = [.useGB, .useMB]
        formatter.countStyle = .file
        return formatter.string(fromByteCount: bytes)
    }
}

/// Errors for model downloading
public enum TextEncoderModelDownloaderError: LocalizedError {
    case modelNotFound
    case qwen3ModelNotFound
    case downloadFailed(String)
    case invalidToken

    public var errorDescription: String? {
        switch self {
        case .modelNotFound:
            return "Model not found"
        case .qwen3ModelNotFound:
            return "Qwen3 model not found"
        case .downloadFailed(let reason):
            return "Download failed: \(reason)"
        case .invalidToken:
            return "Invalid HuggingFace token"
        }
    }
}
