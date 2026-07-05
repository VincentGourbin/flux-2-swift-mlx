// Flux2ModelPaths.swift - Path resolution helpers for Flux.2 model components.
//
// Replaces the path-lookup surface of the deleted `Flux2ModelDownloader` class.
// All actual downloads are owned by `SwiftAcervo` (`Acervo.ensureAvailable`);
// this file only resolves on-disk locations and reports sizes.

import Foundation
import SwiftAcervo

/// Path-lookup and size helpers for Flux.2 model components.
///
/// These helpers wrap `Acervo.modelDirectory(for:)` with the Flux-specific
/// repoId/subfolder mapping carried by `ModelRegistry.ModelComponent`. The
/// `available` predicate forwards to `Acervo.isModelAvailable(_:)`, which is
/// the canonical strict-on-disk check as of SwiftAcervo 0.14.
public enum Flux2ModelPaths {

  /// Whether the component is fully present on disk according to Acervo's
  /// manifest. This is the strict check — partial downloads return `false`.
  public static func isDownloaded(_ component: ModelRegistry.ModelComponent) -> Bool {
    Acervo.isModelAvailable(component.repoId)
  }

  /// Resolve the on-disk directory for a model component.
  ///
  /// Returns the deepest directory whose contents the weight loaders read
  /// non-recursively: for components shipped under a subfolder (e.g.
  /// `transformer/qint8/`, `vae/`) the subfolder is appended when present.
  ///
  /// Returns `nil` only when the component is not on disk at all. The
  /// "is it complete?" question is delegated to `Acervo.isModelAvailable`.
  public static func findModelPath(for component: ModelRegistry.ModelComponent) -> URL? {
    guard let modelDir = try? Acervo.modelDirectory(for: component.repoId),
      FileManager.default.fileExists(atPath: modelDir.path)
    else {
      return nil
    }
    var path = modelDir
    if let sub = component.repoSubfolder {
      let subPath = path.appendingPathComponent(sub)
      if FileManager.default.fileExists(atPath: subPath.path) {
        path = subPath
      }
    }
    // Require either config.json or model_index.json at the resolved depth so
    // we don't hand callers an empty directory. The full manifest check lives
    // in `Acervo.isModelAvailable` — use that for the "ready to load" gate.
    let hasConfig = FileManager.default.fileExists(
      atPath: path.appendingPathComponent("config.json").path)
    let hasModelIndex = FileManager.default.fileExists(
      atPath: path.appendingPathComponent("model_index.json").path)
    guard hasConfig || hasModelIndex else {
      return nil
    }
    return path
  }

  /// Total on-disk size of a downloaded component in bytes.
  public static func diskSize(for component: ModelRegistry.ModelComponent) -> Int64? {
    guard let path = findModelPath(for: component) else { return nil }
    return directorySize(at: path)
  }

  /// Total on-disk size across all CDN-provisioned transformer variants plus
  /// the VAE, counting each shared Acervo model directory once.
  public static func downloadedSize() -> Int64 {
    var total: Int64 = 0
    var visitedRepoIds: Set<String> = []

    let provisionedTransformerVariants =
      ModelRegistry.TransformerVariant.allCases.filter(\.isProvisionedOnCDN)
    var components: [ModelRegistry.ModelComponent] = provisionedTransformerVariants.map {
      .transformer($0)
    }
    components.append(.vae(.standard))

    for component in components {
      // Multiple components (klein4B_bf16 + vae) can share a model directory.
      // Count each directory once.
      if !visitedRepoIds.insert(component.repoId).inserted {
        continue
      }
      if let path = findModelPath(for: component) {
        total += directorySize(at: path)
      }
    }
    return total
  }

  /// Delete the Acervo model directory that hosts the given component.
  ///
  /// Idempotent: returns silently if the directory is not present.
  ///
  /// Important: the Acervo model directory may host more than one logical
  /// component — e.g., `.vae(.standard)` and `.transformer(.klein4B_bf16)`
  /// both ship under `black-forest-labs/FLUX.2-klein-4B`. Deleting via the
  /// component-level entry point removes the entire shared directory.
  public static func delete(_ component: ModelRegistry.ModelComponent) throws {
    let repoId = component.repoId
    guard let modelDir = try? Acervo.modelDirectory(for: repoId),
      FileManager.default.fileExists(atPath: modelDir.path)
    else {
      return
    }
    do {
      try Acervo.deleteModel(repoId)
      Flux2Debug.log("Deleted \(component.displayName)")
    } catch {
      Flux2Debug.log(
        "Acervo.deleteModel(\(repoId)) raised: \(error.localizedDescription) — treating as deleted")
    }
  }

  /// Format bytes as a human-readable string.
  public static func formatSize(_ bytes: Int64) -> String {
    let formatter = ByteCountFormatter()
    formatter.allowedUnits = [.useGB, .useMB]
    formatter.countStyle = .file
    return formatter.string(fromByteCount: bytes)
  }

  private static func directorySize(at url: URL) -> Int64 {
    let fm = FileManager.default
    guard
      let enumerator = fm.enumerator(
        at: url, includingPropertiesForKeys: [.fileSizeKey, .isRegularFileKey])
    else {
      return 0
    }
    var total: Int64 = 0
    for case let fileURL as URL in enumerator {
      guard let values = try? fileURL.resourceValues(forKeys: [.isRegularFileKey, .fileSizeKey]),
        values.isRegularFile == true,
        let size = values.fileSize
      else { continue }
      total += Int64(size)
    }
    return total
  }
}

// MARK: - CDN Provisioning

extension ModelRegistry.TransformerVariant {
  /// Whether this variant is currently provisioned on the Acervo CDN.
  ///
  /// Returns `false` for the four cut variants (Sortie 19, Task 3):
  /// - `.bf16` — 64 GB FLUX.2-dev BF16; qint8 is the canonical production
  ///   quantization, license redistribution complications cut bf16.
  /// - `.klein4B_base_bf16` — base (non-distilled) for LoRA training, deferred
  ///   until LoRA-training is a v1 surface.
  /// - `.klein9B_base_bf16` — base (non-distilled), LoRA-training only, gated.
  /// - `.klein9B_kv_bf16` — multi-reference I2I specialty variant, gated.
  ///
  /// UI surfaces iterating `allCases` should consult this property to gray
  /// out (or hide) variants that would otherwise throw
  /// `Flux2DownloadError.notProvisionedOnCDN(variant:)` from a download
  /// attempt.
  public var isProvisionedOnCDN: Bool {
    switch self {
    case .bf16, .klein4B_base_bf16, .klein9B_base_bf16, .klein9B_kv_bf16:
      return false
    case .qint8, .klein4B_bf16, .klein4B_8bit, .klein4B_4bit, .klein9B_bf16:
      // .klein4B_4bit confirmed .available on the Acervo CDN (Sortie B1 —
      // themindstudio/flux2-klein-4b-mlx-4bit manifest resolves; see
      // EXECUTION_PLAN.md B1).
      return true
    }
  }
}

// MARK: - Progress callback

/// Progress callback for download updates. Preserved as a top-level typealias
/// so callers that previously imported it via `Flux2ModelDownloader` keep
/// working.
public typealias Flux2DownloadProgressCallback = @Sendable (Double, String) -> Void

// MARK: - Errors

/// Errors emitted by Flux.2 download flows. After Sortie 19's CDN cutover the
/// only remaining Flux-specific case is `.notProvisionedOnCDN`; raw network /
/// disk failures propagate from `SwiftAcervo` directly.
public enum Flux2DownloadError: LocalizedError {
  /// Variant has been intentionally cut from CDN provisioning. The follow-up
  /// CDN provisioning mission tracked under `docs/missions/` will re-enable
  /// it.
  case notProvisionedOnCDN(variant: ModelRegistry.TransformerVariant)

  public var errorDescription: String? {
    switch self {
    case .notProvisionedOnCDN(let variant):
      return
        "Variant \(variant.rawValue) is not yet provisioned on the Acervo CDN. "
        + "Track follow-up CDN provisioning mission in docs/missions/."
    }
  }
}
