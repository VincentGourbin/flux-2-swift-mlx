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
        // Check our local models directory
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

        // Check legacy HuggingFace cache (macOS only - ~/.cache/huggingface/hub)
        #if os(macOS)
        let homeDir = FileManager.default.homeDirectoryForCurrentUser
        let hubCache = homeDir
            .appendingPathComponent(".cache")
            .appendingPathComponent("huggingface")
            .appendingPathComponent("hub")

        let modelFolder = "models--\(repoId.replacingOccurrences(of: "/", with: "--"))"
        let snapshotsDir = hubCache.appendingPathComponent(modelFolder).appendingPathComponent("snapshots")

        if let contents = try? FileManager.default.contentsOfDirectory(atPath: snapshotsDir.path),
           let latestSnapshot = contents.sorted().last {
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
        }
        #endif

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
        case .vae:
            // Use Klein 4B distilled repo (not gated, same VAE as all Flux.2 models)
            return "black-forest-labs/FLUX.2-klein-4B"
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

    /// Download a model component from CDN or HuggingFace
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

        // Try CDN first if configured
        if let cdnBase = ModelRegistry.cdnBaseURL {
            do {
                return try await downloadFromCDN(
                    component,
                    cdnBase: cdnBase,
                    progress: progress
                )
            } catch {
                Flux2Debug.log("CDN download failed for \(component.displayName): \(error.localizedDescription). Falling back to HuggingFace.")
            }
        }

        let repoId = Self.repoId(for: component)
        let subfolder = Self.subfolder(for: component)
        progress?(0.0, "Fetching file list for \(component.displayName)...")

        Flux2Debug.log("Downloading \(component.displayName) from \(repoId)")

        // Get file list from HuggingFace API
        let allFiles = try await fetchFileList(repoId: repoId, subfolder: subfolder)

        // Filter to only necessary files
        let filesToDownload = allFiles.filter { file in
            file.path.hasSuffix(".safetensors") ||
            file.path.hasSuffix(".json") ||
            file.path == "tokenizer.model"
        }

        guard !filesToDownload.isEmpty else {
            throw Flux2DownloadError.modelNotFound("No model files found in \(repoId)")
        }

        // Create destination directory
        let destDir = ModelRegistry.localPath(for: component)
        try FileManager.default.createDirectory(at: destDir, withIntermediateDirectories: true)

        // Download each file with byte-level progress
        var completedBytes: Int64 = 0
        let totalBytes = filesToDownload.reduce(Int64(0)) { $0 + $1.size }

        for file in filesToDownload {
            let fileName = URL(fileURLWithPath: file.path).lastPathComponent
            let baseBytes = completedBytes

            let fileProgress: ((Int64, Int64) -> Void)? = totalBytes > 0 ? { bytesWritten, _ in
                let overall = Double(baseBytes + bytesWritten) / Double(totalBytes)
                let downloaded = Self.formatSize(baseBytes + bytesWritten)
                let total = Self.formatSize(totalBytes)
                progress?(min(overall, 0.99), "Downloading \(fileName): \(downloaded) / \(total)")
            } : nil

            let fileURL = try await downloadFile(
                repoId: repoId,
                filePath: file.path,
                to: destDir.appendingPathComponent(fileName),
                progress: fileProgress
            )

            if let attrs = try? FileManager.default.attributesOfItem(atPath: fileURL.path),
               let size = attrs[.size] as? Int64 {
                completedBytes += size
            }

            Flux2Debug.log("Downloaded \(fileName) (\(Self.formatSize(completedBytes)) total)")
        }

        progress?(1.0, "Download complete: \(Self.formatSize(completedBytes))")
        return destDir
    }

    // MARK: - CDN Download

    /// Download a model component from a CDN using manifest.json
    private func downloadFromCDN(
        _ component: ModelRegistry.ModelComponent,
        cdnBase: URL,
        progress: Flux2DownloadProgressCallback? = nil
    ) async throws -> URL {
        let cdnDir = component.cdnDirectoryName
        let manifestURL = cdnBase
            .appendingPathComponent("models")
            .appendingPathComponent(cdnDir)
            .appendingPathComponent("manifest.json")

        progress?(0.0, "Fetching manifest for \(component.displayName)...")
        Flux2Debug.log("Downloading \(component.displayName) from CDN: \(cdnBase)")

        // Fetch manifest
        let (data, response) = try await session.data(from: manifestURL)
        guard let httpResponse = response as? HTTPURLResponse,
              httpResponse.statusCode == 200 else {
            throw Flux2DownloadError.downloadFailed(
                "CDN manifest not available for \(cdnDir)"
            )
        }

        guard let manifest = try? JSONDecoder().decode(CDNManifest.self, from: data) else {
            throw Flux2DownloadError.downloadFailed("Invalid CDN manifest for \(cdnDir)")
        }

        let filesToDownload = manifest.files.filter { file in
            file.name.hasSuffix(".safetensors") ||
            file.name.hasSuffix(".json") ||
            file.name == "tokenizer.model"
        }

        guard !filesToDownload.isEmpty else {
            throw Flux2DownloadError.modelNotFound("No model files in CDN manifest for \(cdnDir)")
        }

        // Create destination directory
        let destDir = ModelRegistry.localPath(for: component)
        try FileManager.default.createDirectory(at: destDir, withIntermediateDirectories: true)

        // Download each file
        var completedBytes: Int64 = 0
        let totalBytes = filesToDownload.reduce(Int64(0)) { $0 + $1.size }

        for file in filesToDownload {
            let fileURL = cdnBase
                .appendingPathComponent("models")
                .appendingPathComponent(cdnDir)
                .appendingPathComponent(file.name)
            let destFile = destDir.appendingPathComponent(file.name)

            let baseBytes = completedBytes
            let fileProgress: ((Int64, Int64) -> Void)? = totalBytes > 0 ? { bytesWritten, _ in
                let overall = Double(baseBytes + bytesWritten) / Double(totalBytes)
                let downloaded = Self.formatSize(baseBytes + bytesWritten)
                let total = Self.formatSize(totalBytes)
                progress?(min(overall, 0.99), "Downloading \(file.name): \(downloaded) / \(total)")
            } : nil

            let downloadedFile = try await downloadDirectURL(
                fileURL,
                to: destFile,
                progress: fileProgress
            )

            if let attrs = try? FileManager.default.attributesOfItem(atPath: downloadedFile.path),
               let size = attrs[.size] as? Int64 {
                completedBytes += size
            }

            Flux2Debug.log("Downloaded \(file.name) from CDN (\(Self.formatSize(completedBytes)) total)")
        }

        progress?(1.0, "Download complete: \(Self.formatSize(completedBytes))")
        return destDir
    }

    /// Download a file from a direct URL (no auth headers)
    private func downloadDirectURL(
        _ url: URL,
        to destination: URL,
        progress: ((Int64, Int64) -> Void)? = nil
    ) async throws -> URL {
        let request = URLRequest(url: url)

        let tempURL: URL
        if let progress = progress {
            tempURL = try await withCheckedThrowingContinuation { continuation in
                let delegate = FileDownloadDelegate(
                    progressHandler: progress,
                    continuation: continuation
                )
                let config = URLSessionConfiguration.default
                config.timeoutIntervalForResource = 3600
                let delegateSession = URLSession(
                    configuration: config,
                    delegate: delegate,
                    delegateQueue: nil
                )
                delegateSession.downloadTask(with: request).resume()
            }
        } else {
            let (url, response) = try await session.download(for: request)
            guard let httpResponse = response as? HTTPURLResponse,
                  httpResponse.statusCode == 200 else {
                throw Flux2DownloadError.downloadFailed("Failed to download from CDN")
            }
            tempURL = url
        }

        if FileManager.default.fileExists(atPath: destination.path) {
            try FileManager.default.removeItem(at: destination)
        }
        try FileManager.default.moveItem(at: tempURL, to: destination)
        return destination
    }

    // MARK: - HuggingFace Download Helpers

    /// Get subfolder path for component within repo
    private static func subfolder(for component: ModelRegistry.ModelComponent) -> String? {
        switch component {
        case .transformer(let variant):
            return variant.huggingFaceSubfolder
        case .vae:
            return "vae"
        case .textEncoder:
            return nil
        }
    }

    /// Fetch file list with sizes from HuggingFace API
    private func fetchFileList(repoId: String, subfolder: String?) async throws -> [(path: String, size: Int64)] {
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

        var files: [(path: String, size: Int64)] = []
        for item in json {
            if let type = item["type"] as? String, type == "file",
               let path = item["path"] as? String {
                let size = (item["size"] as? Int64) ?? (item["size"] as? Int).map(Int64.init) ?? 0
                files.append((path: path, size: size))
            }
        }

        return files
    }

    /// Download a single file from HuggingFace with optional byte-level progress
    private func downloadFile(
        repoId: String,
        filePath: String,
        to destination: URL,
        progress: ((Int64, Int64) -> Void)? = nil
    ) async throws -> URL {
        let urlString = "https://huggingface.co/\(repoId)/resolve/main/\(filePath)"

        guard let url = URL(string: urlString.addingPercentEncoding(withAllowedCharacters: .urlQueryAllowed) ?? urlString) else {
            throw Flux2DownloadError.downloadFailed("Invalid URL: \(urlString)")
        }

        var request = URLRequest(url: url)
        if let token = hfToken {
            request.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization")
        }

        let tempURL: URL

        if let progress = progress {
            tempURL = try await withCheckedThrowingContinuation { continuation in
                let delegate = FileDownloadDelegate(
                    progressHandler: progress,
                    continuation: continuation
                )
                let config = URLSessionConfiguration.default
                config.timeoutIntervalForResource = 3600
                let delegateSession = URLSession(
                    configuration: config,
                    delegate: delegate,
                    delegateQueue: nil
                )
                delegateSession.downloadTask(with: request).resume()
            }
        } else {
            let (url, response) = try await session.download(for: request)
            guard let httpResponse = response as? HTTPURLResponse,
                  httpResponse.statusCode == 200 else {
                throw Flux2DownloadError.downloadFailed("Failed to download \(filePath)")
            }
            tempURL = url
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
    public static func delete(_ component: ModelRegistry.ModelComponent) throws {
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

    private static func directorySize(at url: URL) -> Int64 {
        let fm = FileManager.default
        guard let enumerator = fm.enumerator(at: url, includingPropertiesForKeys: [.fileSizeKey]) else {
            return 0
        }

        var total: Int64 = 0
        for case let fileURL as URL in enumerator {
            if let attrs = try? fm.attributesOfItem(atPath: fileURL.path),
               let size = attrs[.size] as? Int64 {
                total += size
            }
        }

        return total
    }
}

// MARK: - Download Delegate

/// Bridges URLSessionDownloadDelegate callbacks to async/await with byte-level progress
private final class FileDownloadDelegate: NSObject, URLSessionDownloadDelegate, @unchecked Sendable {
    private let progressHandler: (Int64, Int64) -> Void
    private let continuation: CheckedContinuation<URL, Error>
    private var resumed = false

    init(
        progressHandler: @escaping (Int64, Int64) -> Void,
        continuation: CheckedContinuation<URL, Error>
    ) {
        self.progressHandler = progressHandler
        self.continuation = continuation
    }

    func urlSession(
        _ session: URLSession,
        downloadTask: URLSessionDownloadTask,
        didFinishDownloadingTo location: URL
    ) {
        guard !resumed else { return }

        // Check HTTP status
        if let httpResponse = downloadTask.response as? HTTPURLResponse,
           httpResponse.statusCode != 200 {
            resumed = true
            continuation.resume(throwing: Flux2DownloadError.downloadFailed(
                "HTTP \(httpResponse.statusCode)"
            ))
            session.finishTasksAndInvalidate()
            return
        }

        // Copy to a stable temp path (original is deleted after this method returns)
        let tempFile = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString)
        do {
            try FileManager.default.copyItem(at: location, to: tempFile)
            resumed = true
            continuation.resume(returning: tempFile)
        } catch {
            resumed = true
            continuation.resume(throwing: error)
        }
        session.finishTasksAndInvalidate()
    }

    func urlSession(
        _ session: URLSession,
        downloadTask: URLSessionDownloadTask,
        didWriteData bytesWritten: Int64,
        totalBytesWritten: Int64,
        totalBytesExpectedToWrite: Int64
    ) {
        progressHandler(totalBytesWritten, totalBytesExpectedToWrite)
    }

    func urlSession(
        _ session: URLSession,
        task: URLSessionTask,
        didCompleteWithError error: Error?
    ) {
        guard !resumed, let error = error else { return }
        resumed = true
        continuation.resume(throwing: Flux2DownloadError.downloadFailed(error.localizedDescription))
        session.finishTasksAndInvalidate()
    }
}

// MARK: - CDN Manifest

/// Manifest file structure for CDN-hosted models.
/// Generated by CI and uploaded alongside model files.
struct CDNManifest: Codable {
    struct FileEntry: Codable {
        let name: String
        let size: Int64
    }

    let files: [FileEntry]
}

// MARK: - Errors

public enum Flux2DownloadError: LocalizedError {
    case modelNotFound(String)
    case downloadFailed(String)
    case verificationFailed([String])
    case insufficientSpace(required: Int64, available: Int64)

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
        }
    }
}
