// ModelDownloader.swift - Download Flux.2 models from HuggingFace
// Copyright 2025 Vincent Gourbin

import Foundation

/// Progress callback for download updates
public typealias Flux2DownloadProgressCallback = @Sendable (Double, String) -> Void

/// Downloads Flux.2 models from HuggingFace Hub
public class Flux2ModelDownloader: @unchecked Sendable {

    /// HuggingFace token for gated models
    private var hfToken: String?

    /// URLSession for downloads
    private let session: URLSession

    public init(hfToken: String? = nil) {
        self.hfToken = hfToken
        if let token = hfToken {
            setenv("HF_TOKEN", token, 1)
        }

        let config = URLSessionConfiguration.default
        config.timeoutIntervalForResource = 3600  // 1 hour for large models
        self.session = URLSession(configuration: config)
    }

    // MARK: - Model Paths

    /// Check if a model component is downloaded
    public static func isDownloaded(_ component: ModelRegistry.ModelComponent) -> Bool {
        findModelPath(for: component) != nil
    }

    /// Find local path for a model component
    public static func findModelPath(for component: ModelRegistry.ModelComponent) -> URL? {
        // Check our local models directory (localPath(for:) already returns a
        // pathOverrides entry unconditionally when one is set for this component)
        let localPath = ModelRegistry.localPath(for: component)

        // Check for config.json OR model_index.json (Klein models use the latter)
        let hasConfig = FileManager.default.fileExists(atPath: localPath.appendingPathComponent("config.json").path)
        let hasModelIndex = FileManager.default.fileExists(atPath: localPath.appendingPathComponent("model_index.json").path)

        if hasConfig || hasModelIndex {
            let verification = verifyModel(at: localPath)
            if verification.complete {
                return localPath
            }
        }

        // An explicit per-component override is authoritative: never fall back
        // to the legacy cache-search locations below just because they happen
        // to hold an unrelated, independently-valid copy of the same component.
        if ModelRegistry.pathOverrides[component] != nil {
            return nil
        }

        // Check configured models directory
        let repoId = repoId(for: component)
        var path = ModelRegistry.modelsDirectory

        for part in repoId.split(separator: "/") {
            path = path.appendingPathComponent(String(part))
        }

        let cacheHasConfig = FileManager.default.fileExists(atPath: path.appendingPathComponent("config.json").path)
        let cacheHasModelIndex = FileManager.default.fileExists(atPath: path.appendingPathComponent("model_index.json").path)

        if cacheHasConfig || cacheHasModelIndex {
            let verification = verifyModel(at: path)
            if verification.complete {
                return path
            }
        }

        // Check legacy HuggingFace cache
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
        let modelIndexPath = modelPath.appendingPathComponent("model_index.json")

        if FileManager.default.fileExists(atPath: configPath.path) ||
           FileManager.default.fileExists(atPath: modelIndexPath.path) {
            // Verify safetensors files are complete
            let verification = verifyModel(at: modelPath)
            if verification.complete {
                return modelPath
            }
        }

        return nil
    }

    /// Get HuggingFace repo ID for a component
    private static func repoId(for component: ModelRegistry.ModelComponent) -> String {
        switch component {
        case .transformer(let variant):
            return variant.huggingFaceRepo
        case .textEncoder:
            // Text encoder uses MistralCore's download system
            return "mistralai/Mistral-Small-3.2-24B-Instruct-2506"
        case .vae(let variant):
            return variant.huggingFaceRepo
        }
    }

    /// Verify model files are complete
    public static func verifyModel(at path: URL) -> (complete: Bool, missing: [String]) {
        let contents = (try? FileManager.default.contentsOfDirectory(atPath: path.path)) ?? []
        let safetensorsFiles = contents.filter { $0.hasSuffix(".safetensors") }

        // Single file model (various naming conventions)
        if safetensorsFiles.contains("model.safetensors") ||
           safetensorsFiles.contains("diffusion_pytorch_model.safetensors") {
            return (true, [])
        }

        // Klein bf16 models use flux-2-klein-*.safetensors naming
        if safetensorsFiles.contains(where: { $0.hasPrefix("flux-2-klein") }) {
            return (true, [])
        }

        // Check for sharded model
        guard !safetensorsFiles.isEmpty else {
            return (false, ["No safetensors files found"])
        }

        // Parse sharded pattern
        var totalShards: Int?
        var foundIndices: Set<Int> = []

        for file in safetensorsFiles {
            let name = file.replacingOccurrences(of: ".safetensors", with: "")
            let parts = name.split(separator: "-")

            guard parts.count == 4,
                  parts[0] == "model",
                  parts[2] == "of",
                  let index = Int(parts[1]),
                  let total = Int(parts[3]) else {
                continue
            }

            if totalShards == nil {
                totalShards = total
            }
            foundIndices.insert(index)
        }

        if let total = totalShards {
            let expectedIndices = Set(1...total)
            let missing = expectedIndices.subtracting(foundIndices)

            if missing.isEmpty {
                return (true, [])
            } else {
                let missingFiles = missing.sorted().map {
                    "model-\(String(format: "%05d", $0))-of-\(String(format: "%05d", total)).safetensors"
                }
                return (false, missingFiles)
            }
        }

        return (true, [])
    }

    // MARK: - Download

    /// Download a model component from HuggingFace
    public func download(
        _ component: ModelRegistry.ModelComponent,
        progress: Flux2DownloadProgressCallback? = nil
    ) async throws -> URL {
        // Check if already downloaded
        if let existingPath = Self.findModelPath(for: component) {
            let verification = Self.verifyModel(at: existingPath)
            if verification.complete {
                progress?(1.0, "Model already downloaded")
                return existingPath
            }
        }

        // A path override typically points at removable/external storage. Fail
        // fast, before any network activity, if its parent directory is
        // missing: only the override's own leaf directory gets created below,
        // never its missing parents — silently recreating a whole missing tree
        // would just as happily build it on the boot volume if the external
        // disk isn't actually connected, masking that mistake instead of
        // surfacing it.
        let destDir = ModelRegistry.localPath(for: component)
        if ModelRegistry.pathOverrides[component] != nil,
           !FileManager.default.fileExists(atPath: destDir.deletingLastPathComponent().path) {
            throw Flux2DownloadError.downloadFailed(
                "\(component.displayName)'s path override (\(destDir.path)) has a missing parent directory — is the external disk connected?"
            )
        }

        let repoId = Self.repoId(for: component)
        let subfolder = Self.subfolder(for: component)
        progress?(0.0, "Fetching file list for \(component.displayName)...")

        Flux2Debug.log("Downloading \(component.displayName) from \(repoId)")

        // Get file list from HuggingFace API
        let files = try await fetchFileList(repoId: repoId, subfolder: subfolder)

        // Filter to only necessary files
        let filesToDownload = files.filter { file in
            file.hasSuffix(".safetensors") ||
            file.hasSuffix(".json") ||
            file == "tokenizer.model"
        }

        guard !filesToDownload.isEmpty else {
            throw Flux2DownloadError.modelNotFound("No model files found in \(repoId)")
        }

        // Create destination directory
        try FileManager.default.createDirectory(at: destDir, withIntermediateDirectories: true)

        // Download each file
        var downloadedBytes: Int64 = 0
        let totalFiles = filesToDownload.count

        for (index, file) in filesToDownload.enumerated() {
            let fileName = URL(fileURLWithPath: file).lastPathComponent
            progress?(Double(index) / Double(totalFiles), "Downloading \(fileName)...")

            let fileURL = try await downloadFile(
                repoId: repoId,
                filePath: file,
                to: destDir.appendingPathComponent(fileName)
            )

            if let attrs = try? FileManager.default.attributesOfItem(atPath: fileURL.path),
               let size = attrs[.size] as? Int64 {
                downloadedBytes += size
            }

            Flux2Debug.log("Downloaded \(fileName) (\(Self.formatSize(downloadedBytes)) total)")
        }

        progress?(1.0, "Download complete: \(Self.formatSize(downloadedBytes))")
        return destDir
    }

    /// Get subfolder path for component within repo
    private static func subfolder(for component: ModelRegistry.ModelComponent) -> String? {
        switch component {
        case .transformer(let variant):
            return variant.huggingFaceSubfolder
        case .vae(let variant):
            return variant.huggingFaceSubfolder
        case .textEncoder:
            return nil
        }
    }

    /// Fetch file list from HuggingFace API
    private func fetchFileList(repoId: String, subfolder: String?) async throws -> [String] {
        var urlString = "https://huggingface.co/api/models/\(repoId)/tree/main"
        if let subfolder = subfolder {
            urlString += "/\(subfolder)"
        }

        guard let url = URL(string: urlString) else {
            throw Flux2DownloadError.downloadFailed("Invalid URL: \(urlString)")
        }

        var request = URLRequest(url: url)
        if let token = hfToken {
            request.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization")
        }

        let (data, response) = try await session.data(for: request)

        guard let httpResponse = response as? HTTPURLResponse else {
            throw Flux2DownloadError.downloadFailed("Invalid response")
        }

        if httpResponse.statusCode == 401 {
            throw Flux2DownloadError.downloadFailed(
                "Authentication required. Set HF_TOKEN environment variable or pass token to downloader."
            )
        }

        if httpResponse.statusCode == 403 {
            throw Flux2DownloadError.downloadFailed(
                "Access denied. You may need to accept the model's license at https://huggingface.co/\(repoId)"
            )
        }

        guard httpResponse.statusCode == 200 else {
            throw Flux2DownloadError.downloadFailed("HTTP \(httpResponse.statusCode)")
        }

        // Parse JSON response
        guard let json = try? JSONSerialization.jsonObject(with: data) as? [[String: Any]] else {
            throw Flux2DownloadError.downloadFailed("Invalid JSON response")
        }

        var files: [String] = []
        for item in json {
            if let type = item["type"] as? String, type == "file",
               let path = item["path"] as? String {
                files.append(path)
            }
        }

        return files
    }

    /// Download a single file from HuggingFace
    private func downloadFile(repoId: String, filePath: String, to destination: URL) async throws -> URL {
        let urlString = "https://huggingface.co/\(repoId)/resolve/main/\(filePath)"

        guard let url = URL(string: urlString.addingPercentEncoding(withAllowedCharacters: .urlQueryAllowed) ?? urlString) else {
            throw Flux2DownloadError.downloadFailed("Invalid URL: \(urlString)")
        }

        var request = URLRequest(url: url)
        if let token = hfToken {
            request.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization")
        }

        let (tempURL, response) = try await session.download(for: request)

        guard let httpResponse = response as? HTTPURLResponse,
              httpResponse.statusCode == 200 else {
            throw Flux2DownloadError.downloadFailed("Failed to download \(filePath)")
        }

        // Move to destination
        if FileManager.default.fileExists(atPath: destination.path) {
            try FileManager.default.removeItem(at: destination)
        }
        try FileManager.default.moveItem(at: tempURL, to: destination)

        return destination
    }

    /// Download all models for a quantization configuration
    public func downloadAll(
        for config: Flux2QuantizationConfig,
        progress: Flux2DownloadProgressCallback? = nil
    ) async throws {
        let components: [ModelRegistry.ModelComponent] = [
            .transformer(ModelRegistry.TransformerVariant(rawValue: config.transformer.rawValue)!),
            .vae(.standard)
        ]

        let totalComponents = Float(components.count + 1)  // +1 for text encoder

        // Download transformer and VAE
        for (index, component) in components.enumerated() {
            let completedComponents = Float(index)
            let componentProgress: Flux2DownloadProgressCallback = { p, msg in
                let overall = (completedComponents + Float(p)) / totalComponents
                progress?(Double(overall), msg)
            }

            _ = try await download(component, progress: componentProgress)
        }

        // Text encoder is handled by MistralCore
        progress?(1.0, "All models downloaded")
    }

    // MARK: - Utilities

    /// Format bytes as human-readable string
    public static func formatSize(_ bytes: Int64) -> String {
        let formatter = ByteCountFormatter()
        formatter.allowedUnits = [.useGB, .useMB]
        formatter.countStyle = .file
        return formatter.string(fromByteCount: bytes)
    }

    /// Delete a downloaded model
    ///
    /// Refuses to delete a component that has an active `ModelRegistry.pathOverrides`
    /// entry: unlike the default cache locations (always a re-downloadable copy),
    /// an override may point directly at a component's only copy — e.g. one
    /// relocated wholesale to an external disk rather than left behind a local
    /// symlink. Deleting it here would be permanent data loss, not cache cleanup.
    public static func delete(_ component: ModelRegistry.ModelComponent) throws {
        if ModelRegistry.pathOverrides[component] != nil {
            throw Flux2DownloadError.deletionRefusedForOverride(component.displayName)
        }

        guard let path = findModelPath(for: component) else {
            return
        }

        try FileManager.default.removeItem(at: path)
        Flux2Debug.log("Deleted \(component.displayName)")
    }

    /// Get total size of downloaded models
    public static func downloadedSize() -> Int64 {
        var total: Int64 = 0

        let components: [ModelRegistry.ModelComponent] = [
            .transformer(.qint8),
            .transformer(.bf16),
            .vae(.standard)
        ]

        for component in components {
            if let path = findModelPath(for: component) {
                total += directorySize(at: path)
            }
        }

        return total
    }

    /// Calculate directory size recursively.
    ///
    /// Walks with `atPath:` APIs (not the `URL`-based family) because a relocated
    /// model's large weight files are replaced with file symlinks to an external
    /// disk: the `URL`-based enumerator/`resourceValues`/`attributesOfItem(atPath:)`
    /// combo reports a symlink's own size (a few bytes), not its target's.
    ///
    /// Shared by every "how big is this model on disk" call site in the package
    /// (`downloadedSize()` here, and Flux2App's Model Manager screen) so the fix
    /// below lives in exactly one place.
    public static func directorySize(at url: URL) -> Int64 {
        sizeOfItem(atPath: url.path, symlinkHopsRemaining: 40)
    }

    /// Size of a single filesystem entry, following symlinks (including chains
    /// and symlinked subdirectories) to their real target rather than reporting
    /// the link's own few-byte size.
    ///
    /// Each symlink is resolved via `destinationOfSymbolicLink(atPath:)` (a raw
    /// `readlink`) rather than `resolvingSymlinksInPath()`, which silently no-ops
    /// and leaks the symlink's own near-zero size when the target is missing
    /// (e.g. an unmounted external disk); a broken symlink contributes 0.
    /// `symlinkHopsRemaining` bounds symlink-chain traversal (not directory
    /// recursion depth) so a cyclic chain can't loop forever.
    private static func sizeOfItem(atPath path: String, symlinkHopsRemaining: Int) -> Int64 {
        let fm = FileManager.default
        guard let attrs = try? fm.attributesOfItem(atPath: path) else { return 0 }

        switch attrs[.type] as? FileAttributeType {
        case .typeSymbolicLink:
            guard symlinkHopsRemaining > 0,
                  let rawTarget = try? fm.destinationOfSymbolicLink(atPath: path) else {
                return 0
            }
            let targetPath = rawTarget.hasPrefix("/")
                ? rawTarget
                : URL(fileURLWithPath: path).deletingLastPathComponent().appendingPathComponent(rawTarget).path
            return sizeOfItem(atPath: targetPath, symlinkHopsRemaining: symlinkHopsRemaining - 1)

        case .typeDirectory:
            guard let children = try? fm.contentsOfDirectory(atPath: path) else { return 0 }
            return children.reduce(Int64(0)) { total, name in
                total + sizeOfItem(
                    atPath: (path as NSString).appendingPathComponent(name),
                    symlinkHopsRemaining: symlinkHopsRemaining
                )
            }

        default:
            return (attrs[.size] as? Int64) ?? 0
        }
    }
}

// MARK: - Errors

public enum Flux2DownloadError: LocalizedError {
    case modelNotFound(String)
    case downloadFailed(String)
    case verificationFailed([String])
    case insufficientSpace(required: Int64, available: Int64)
    case deletionRefusedForOverride(String)

    public var errorDescription: String? {
        switch self {
        case .modelNotFound(let id):
            return "Model not found: \(id)"
        case .downloadFailed(let reason):
            return "Download failed: \(reason)"
        case .verificationFailed(let missing):
            return "Verification failed, missing files: \(missing.joined(separator: ", "))"
        case .insufficientSpace(let required, let available):
            return "Insufficient disk space: need \(Flux2ModelDownloader.formatSize(required)), have \(Flux2ModelDownloader.formatSize(available))"
        case .deletionRefusedForOverride(let name):
            return "\(name) has a path override in place and was not deleted — it may be its only copy (e.g. relocated to an external disk). Remove the override or delete the files manually if you're sure."
        }
    }
}
