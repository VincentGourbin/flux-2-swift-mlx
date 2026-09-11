// ModelDownloader.swift - Download Flux.2 models from HuggingFace
// Copyright 2025 Vincent Gourbin

import Foundation
import FluxTextEncoders

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

    /// Where a component lives, captured from exactly one read of
    /// `ModelRegistry.pathOverride(for:)`.
    ///
    /// Every public operation resolves this once up front and threads it
    /// through, so a guard ("is this an override?") and the action it guards
    /// ("delete what findModelPath returns") can never observe two different
    /// override states within the same call.
    struct Location {
        let url: URL
        let isOverride: Bool
    }

    static func location(for component: ModelRegistry.ModelComponent) -> Location {
        if let override = ModelRegistry.pathOverride(forComponent: component) {
            return Location(url: override, isOverride: true)
        }
        return Location(url: ModelRegistry.localPath(for: component), isOverride: false)
    }

    /// Why `findModelPath(for:)` is `nil` when there is more to it than "never
    /// downloaded" — an override is set, or the weights are dangling symlinks
    /// (relocated to a disk that isn't connected). `nil` when a plain download
    /// would fix it. Meant for error messages: "run `flux2 download`" is wrong
    /// advice in both cases.
    public static func unavailableReason(for component: ModelRegistry.ModelComponent) -> String? {
        let location = location(for: component)
        guard findModelPath(for: component, at: location) == nil else { return nil }

        // Same classification `download()` throws on, so the explanation a user
        // reads and the error a caller catches can never name different causes.
        if let problem = destinationProblem(component, at: location.url) {
            return problem.errorDescription
        }
        if location.isOverride {
            return "\(component.displayName)'s path override (\(location.url.path)) holds no complete model — restore the files there or clear the override."
        }
        return nil
    }

    /// Everything that makes a destination directory unusable, resolved in one
    /// place and in one precedence order.
    ///
    /// `download()` throws this before touching the network; `unavailableReason`
    /// renders it. Keeping both on one helper is what stops a fifth check (or a
    /// reordering) from landing in only one of them.
    static func destinationProblem(
        _ component: ModelRegistry.ModelComponent, at directory: URL
    ) -> Flux2DownloadError? {
        let name = component.displayName

        // Under App Sandbox, `stat` works everywhere but listing a directory
        // outside an active security scope fails with EPERM; verifyModel
        // swallows that into "no weights", which must not become a download.
        if isDirectoryUnreadable(directory) {
            return .destinationUnreadable(name, directory)
        }
        // An unmounted volume: `createDirectory(withIntermediateDirectories:)`
        // would rebuild the missing tree on the boot volume and download into
        // it. Applies to a catalog root on removable storage as much as to an
        // override, and precedes the weight checks — an absent directory has
        // no weights to inspect.
        if isOnUnmountedVolume(directory) {
            return .destinationVolumeUnavailable(name, directory)
        }
        // Relocated weights: writing here replaces the links with local copies
        // and orphans the external payload. Dangling ones mean the disk is
        // gone; live ones mean the series is incomplete at the target.
        let links = SafetensorsDirectory.symlinkedWeights(at: directory)
        guard !links.isEmpty else { return nil }
        let unreachable = SafetensorsDirectory.unreachableWeights(at: directory)
        if !unreachable.isEmpty {
            return .weightsUnreachable(name, directory, unreachable)
        }
        return .weightsRelocated(name, directory, missing: verifyModel(at: directory).missing)
    }

    /// The path exists (`stat` works) but can't be listed as a directory —
    /// under App Sandbox a missing security scope, otherwise a plain file
    /// sitting where the model directory belongs.
    static func isDirectoryUnreadable(_ directory: URL) -> Bool {
        let fm = FileManager.default
        var isDir: ObjCBool = false
        guard fm.fileExists(atPath: directory.path, isDirectory: &isDir), isDir.boolValue else {
            return false
        }
        return (try? fm.contentsOfDirectory(atPath: directory.path)) == nil
    }

    /// `.safetensors` entries that are symlinks, live or dangling: the model
    /// was relocated, and `download()` must not write into it.
    public static func symlinkedWeights(at directory: URL) -> [String] {
        SafetensorsDirectory.symlinkedWeights(at: directory)
    }

    /// Symlinked `.safetensors` entries whose target can't be reached — a model
    /// relocated to an external disk that is currently unplugged. Neither
    /// loadable nor safely re-downloadable.
    public static func unreachableWeights(at directory: URL) -> [String] {
        SafetensorsDirectory.unreachableWeights(at: directory)
    }

    /// The legacy `huggingface_hub`-style cache searched as a last resort
    /// (`~/.cache/huggingface/hub`). Overridable so tests don't depend on
    /// whatever snapshots the current user has lying around.
    nonisolated(unsafe) public static var legacyHubCacheDirectory: URL =
        FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent(".cache")
            .appendingPathComponent("huggingface")
            .appendingPathComponent("hub")

    /// Find local path for a model component
    public static func findModelPath(for component: ModelRegistry.ModelComponent) -> URL? {
        findModelPath(for: component, at: location(for: component))
    }

    static func findModelPath(for component: ModelRegistry.ModelComponent, at location: Location) -> URL? {
        // An override is authoritative: it's the only place we look. Never fall
        // back to the cache-search tiers below just because one of them happens
        // to hold an unrelated, independently-valid copy of the same component.
        if location.isOverride {
            return isCompleteModel(at: location.url) ? location.url : nil
        }

        // Check our local models directory
        if isCompleteModel(at: location.url) {
            return location.url
        }

        // Check configured models directory
        let repoId = repoId(for: component)
        var path = ModelRegistry.modelsDirectory

        for part in repoId.split(separator: "/") {
            path = path.appendingPathComponent(String(part))
        }

        if isCompleteModel(at: path) {
            return path
        }

        // Check legacy HuggingFace cache
        let hubCache = legacyHubCacheDirectory

        let modelFolder = "models--\(repoId.replacingOccurrences(of: "/", with: "--"))"
        let snapshotsDir = hubCache.appendingPathComponent(modelFolder).appendingPathComponent("snapshots")

        guard let contents = try? FileManager.default.contentsOfDirectory(atPath: snapshotsDir.path),
              let latestSnapshot = contents.sorted().last else {
            return nil
        }

        let modelPath = snapshotsDir.appendingPathComponent(latestSnapshot)
        return isCompleteModel(at: modelPath) ? modelPath : nil
    }

    /// True when `url` lives under `/Volumes` but not on a mounted volume: its
    /// nearest existing ancestor is on the *boot* volume (same volume
    /// identifier as `/`). That covers both an absent mount point (ancestor is
    /// `/Volumes`) and a ghost mount-point directory left behind on the boot
    /// volume after an unclean unmount. A missing subfolder on a mounted disk
    /// resolves to an ancestor on that disk and is fine.
    ///
    /// Mounts outside `/Volumes` (`hdiutil -mountpoint`, sshfs under `$HOME`)
    /// can't be told apart from a plain directory this way and are not
    /// detected; the guard's job is a clear error *before* any network
    /// transfer, not a complete mount oracle.
    static func isOnUnmountedVolume(_ url: URL) -> Bool {
        let fm = FileManager.default
        var ancestor = url.standardizedFileURL
        while !fm.fileExists(atPath: ancestor.path) {
            let parent = ancestor.deletingLastPathComponent()
            if parent.path == ancestor.path { return false }
            ancestor = parent
        }
        // The ancestor exists, so resolving is safe (the pitfall of
        // resolvingSymlinksInPath only concerns dangling links).
        let resolved = ancestor.resolvingSymlinksInPath()
        let components = resolved.pathComponents
        guard components.count >= 2, components[1].caseInsensitiveCompare("Volumes") == .orderedSame else {
            return false
        }
        guard let ancestorVolume = try? resolved.resourceValues(forKeys: [.volumeIdentifierKey]).volumeIdentifier,
              let rootVolume = try? URL(fileURLWithPath: "/").resourceValues(forKeys: [.volumeIdentifierKey]).volumeIdentifier
        else { return components.count == 2 }
        return ancestorVolume.isEqual(rootVolume)
    }

    /// A model directory counts as present when it has a `config.json` or
    /// `model_index.json` (Klein models use the latter) and `verifyModel`
    /// finds its weights complete.
    private static func isCompleteModel(at path: URL) -> Bool {
        let fm = FileManager.default
        let hasConfig = fm.fileExists(atPath: path.appendingPathComponent("config.json").path)
        let hasModelIndex = fm.fileExists(atPath: path.appendingPathComponent("model_index.json").path)
        guard hasConfig || hasModelIndex else { return false }
        return verifyModel(at: path).complete
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

    /// Verify model files are complete.
    ///
    /// A `.safetensors` entry only counts if its bytes are actually reachable:
    /// the directory listing is filtered through `fileExists(atPath:)`, which is
    /// `stat`-based and follows symlinks, so a weight file relocated to an
    /// external disk that is currently unmounted (a dangling symlink) is
    /// reported as missing rather than counted by name. `._*` entries are
    /// AppleDouble sidecars (exFAT/NTFS) and never weights.
    ///
    /// Sharded weights are recognised by their `-NNNNN-of-MMMMM` suffix
    /// whatever the stem (`model-…`, `diffusion_pytorch_model-…`), and the
    /// whole `1…M` series must be reachable.
    public static func verifyModel(at path: URL) -> (complete: Bool, missing: [String]) {
        SafetensorsDirectory.verifySeries(
            at: path,
            // Klein bf16 models ship a single flux-2-klein-*.safetensors.
            singleFilePrefixes: ["flux-2-klein"])
    }

    // MARK: - Download

    /// Download a model component from HuggingFace
    public func download(
        _ component: ModelRegistry.ModelComponent,
        progress: Flux2DownloadProgressCallback? = nil
    ) async throws -> URL {
        // One override snapshot for the whole operation: the already-downloaded
        // check and the write destination below must agree on where this
        // component lives, even if an override is set/cleared mid-download.
        let location = Self.location(for: component)

        // Check if already downloaded
        if let existingPath = Self.findModelPath(for: component, at: location) {
            progress?(1.0, "Model already downloaded")
            return existingPath
        }

        // Everything that can be known about the destination is checked here,
        // before any network activity, so a doomed download fails in
        // milliseconds with a precise reason rather than after gigabytes.
        let destDir = location.url
        let fm = FileManager.default

        // Everything knowable about the destination — sandbox scope, unmounted
        // volume, relocation symlinks — decided before any network activity, so
        // a doomed download fails in milliseconds with a precise reason instead
        // of after gigabytes. Shared with `unavailableReason`.
        if let problem = Self.destinationProblem(component, at: destDir) {
            throw problem
        }

        // Read-only volumes (NTFS by default, a dirty exFAT remounted read-only)
        // pass every check above; the only way to know is to create the
        // directory and ask. Anything that throws before the first file lands
        // then removes a directory we created ourselves, so a failed attempt
        // (403 without a licence, offline, not writable) leaves no empty tree.
        // The emptiness check also keeps a concurrent download of the same
        // component — which callers must serialize anyway, since both would
        // write the same files — from having its directory pulled away.
        let createdDestination = !fm.fileExists(atPath: destDir.path)
        try fm.createDirectory(at: destDir, withIntermediateDirectories: true)
        var transferStarted = false
        defer {
            if createdDestination, !transferStarted,
               ((try? fm.contentsOfDirectory(atPath: destDir.path)) ?? []).isEmpty {
                try? fm.removeItem(at: destDir)
            }
        }
        guard fm.isWritableFile(atPath: destDir.path) else {
            throw Flux2DownloadError.destinationNotWritable(component.displayName, destDir)
        }

        let repoId = Self.repoId(for: component)
        let subfolder = Self.subfolder(for: component)
        progress?(0.0, "Fetching file list for \(component.displayName)...")

        Flux2Debug.log("Downloading \(component.displayName) from \(repoId)")

        // Get file list from HuggingFace API
        let files = try await fetchFileList(repoId: repoId, subfolder: subfolder)

        // Filter to only necessary files. Small metadata first: if a write is
        // still going to fail for a reason no probe can catch, it fails on a
        // kilobyte of JSON rather than after a multi-gigabyte weight.
        let filesToDownload = files.filter { file in
            file.hasSuffix(".safetensors") ||
            file.hasSuffix(".json") ||
            file == "tokenizer.model"
        }.sorted { a, b in
            let aWeight = a.hasSuffix(".safetensors"), bWeight = b.hasSuffix(".safetensors")
            return aWeight == bWeight ? a < b : !aWeight
        }

        guard !filesToDownload.isEmpty else {
            throw Flux2DownloadError.modelNotFound("No model files found in \(repoId)")
        }
        transferStarted = true

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

        // Move to destination. `attributesOfItem` is lstat-based, so a dangling
        // symlink counts as "something is there" and gets removed — otherwise
        // `moveItem` fails with "already exists" against the stale link.
        if (try? FileManager.default.attributesOfItem(atPath: destination.path)) != nil {
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
    /// Refuses to delete a component that has a path override: the default
    /// locations are re-downloadable cache copies, but an override may be the
    /// component's only copy (relocated wholesale to an external disk), and
    /// nothing marks it as disposable the way the framework's own download
    /// locations are. The guard and the lookup share one override snapshot, so
    /// the guard can't be bypassed by an override set between the two.
    ///
    /// Without an override this removes whatever `findModelPath` resolves,
    /// including a legacy cache-tier directory — unchanged, pre-existing
    /// behavior.
    public static func delete(_ component: ModelRegistry.ModelComponent) throws {
        let location = location(for: component)
        if location.isOverride {
            throw Flux2DownloadError.deletionRefusedForOverride(component.displayName)
        }
        guard let path = findModelPath(for: component, at: location) else {
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
        sizeOfItem(atPath: url.path, symlinkHopsRemaining: 40, visitedDirectories: [])
    }

    /// Size of a single filesystem entry, following symlinks (including chains
    /// and symlinked subdirectories) to their real target rather than reporting
    /// the link's own few-byte size.
    ///
    /// Each symlink is resolved via `destinationOfSymbolicLink(atPath:)` (a raw
    /// `readlink`) rather than `resolvingSymlinksInPath()`, which silently no-ops
    /// and leaks the symlink's own near-zero size when the target is missing
    /// (e.g. an unmounted external disk); a broken symlink — including one with
    /// an empty stored target, which makes a naive relative-path join a no-op —
    /// contributes 0. `symlinkHopsRemaining` bounds symlink-chain length.
    /// `visitedDirectories` (keyed by device+inode, not path string, so it
    /// still works across symlink indirection) additionally guards against a
    /// symlink pointing at an ancestor directory, which would otherwise
    /// re-sum that ancestor's contents on every hop.
    private static func sizeOfItem(atPath path: String, symlinkHopsRemaining: Int, visitedDirectories: Set<String>) -> Int64 {
        let fm = FileManager.default
        guard let attrs = try? fm.attributesOfItem(atPath: path) else { return 0 }

        switch attrs[.type] as? FileAttributeType {
        case .typeSymbolicLink:
            guard symlinkHopsRemaining > 0,
                  let rawTarget = try? fm.destinationOfSymbolicLink(atPath: path),
                  !rawTarget.isEmpty else {
                return 0
            }
            let targetPath = rawTarget.hasPrefix("/")
                ? rawTarget
                : URL(fileURLWithPath: path).deletingLastPathComponent().appendingPathComponent(rawTarget).path
            return sizeOfItem(
                atPath: targetPath,
                symlinkHopsRemaining: symlinkHopsRemaining - 1,
                visitedDirectories: visitedDirectories
            )

        case .typeDirectory:
            let inode = (attrs[.systemFileNumber] as? Int) ?? 0
            let device = (attrs[.systemNumber] as? Int) ?? 0
            let identity = "\(device):\(inode)"
            guard !visitedDirectories.contains(identity) else { return 0 }

            guard let children = try? fm.contentsOfDirectory(atPath: path) else { return 0 }
            let nextVisited = visitedDirectories.union([identity])
            return children.reduce(Int64(0)) { total, name in
                total + sizeOfItem(
                    atPath: (path as NSString).appendingPathComponent(name),
                    symlinkHopsRemaining: symlinkHopsRemaining,
                    visitedDirectories: nextVisited
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
    case invalidPathOverride(URL)
    case deletionRefusedForOverride(String)
    case destinationVolumeUnavailable(String, URL)
    case weightsUnreachable(String, URL, [String])
    case weightsRelocated(String, URL, missing: [String])
    case destinationUnreadable(String, URL)
    case destinationNotWritable(String, URL)

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
        case .invalidPathOverride(let url):
            return "A path override must be a file URL, got: \(url.absoluteString)"
        case .weightsUnreachable(let name, let url, let files):
            return "\(name) at \(url.path) has weight files that are symlinks to a disk that isn't connected (\(files.prefix(3).joined(separator: ", "))) — connect it and retry; re-downloading would silently undo the relocation."
        case .destinationUnreadable(let name, let url):
            return "\(name)'s directory (\(url.path)) exists but can't be listed — under App Sandbox it must lie inside an active security-scoped resource."
        case .destinationNotWritable(let name, let url):
            return "\(name)'s directory (\(url.path)) is not writable — is the volume mounted read-only?"
        case .deletionRefusedForOverride(let name):
            return "\(name) has a path override and was not deleted — that location may be its only copy (e.g. an external disk). Clear the override or remove the files manually."
        case .destinationVolumeUnavailable(let name, let url):
            return "\(name)'s directory (\(url.path)) is on a volume that isn't mounted — connect the external disk and retry."
        case .weightsRelocated(let name, let url, let missing):
            return "\(name) at \(url.path) holds relocated weight files (symlinks) but is incomplete (\(missing.prefix(3).joined(separator: ", "))) — restore the missing files at the relocation target; downloading here would replace the live links with local copies."
        }
    }
}
