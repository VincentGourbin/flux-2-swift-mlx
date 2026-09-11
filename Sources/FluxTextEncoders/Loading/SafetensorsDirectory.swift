// SafetensorsDirectory.swift - Shared inspection of a model weight directory
// Copyright 2025 Vincent Gourbin

import Foundation

/// How a model directory's `.safetensors` files look on disk.
///
/// Lives in `FluxTextEncoders` because `Flux2Core` depends on it: both
/// downloaders (and their weight loaders) need the exact same answers about
/// reachability, relocation symlinks and shard series. They used to carry
/// separate copies, which is how the shard-series fix landed in one verifier
/// and not the other.
public enum SafetensorsDirectory {

    /// `.safetensors` entries whose bytes are reachable right now.
    ///
    /// `fileExists` is `stat`-based and follows symlinks, so a weight relocated
    /// to an external disk that is unplugged (a dangling link) is *not*
    /// reachable. `._*` entries are AppleDouble sidecars written by macOS on
    /// exFAT/NTFS volumes and are never weights — feeding one to
    /// `loadArrays` yields an opaque "invalid json header length" error.
    public static func reachableWeights(at directory: URL) -> [String] {
        let fm = FileManager.default
        let contents = (try? fm.contentsOfDirectory(atPath: directory.path)) ?? []
        return contents.filter { name in
            isWeightName(name) && fm.fileExists(atPath: directory.appendingPathComponent(name).path)
        }.sorted()
    }

    /// `.safetensors` entries that are symlinks, live or dangling: the model
    /// was relocated per-file to another disk. Writing into such a directory
    /// replaces the links with local copies and orphans the external payload.
    public static func symlinkedWeights(at directory: URL) -> [String] {
        let fm = FileManager.default
        let contents = (try? fm.contentsOfDirectory(atPath: directory.path)) ?? []
        return contents.filter { name in
            guard isWeightName(name) else { return false }
            let attrs = try? fm.attributesOfItem(atPath: directory.appendingPathComponent(name).path)
            return (attrs?[.type] as? FileAttributeType) == .typeSymbolicLink
        }.sorted()
    }

    /// Symlinked weights whose target can't be reached — the relocation disk
    /// isn't connected.
    public static func unreachableWeights(at directory: URL) -> [String] {
        let fm = FileManager.default
        return symlinkedWeights(at: directory).filter {
            !fm.fileExists(atPath: directory.appendingPathComponent($0).path)
        }
    }

    /// Whether the reachable weights form a complete model, and which shards
    /// are missing if not.
    ///
    /// Shards are recognised by their `-NNNNN-of-MMMMM` suffix whatever the
    /// stem (`model-…`, `diffusion_pytorch_model-…`) and grouped by
    /// (stem, total), so a leftover shard of another series can neither
    /// complete nor break the real one, whatever order the listing comes in.
    /// `singleFileNames` are stems accepted on their own (a one-file model).
    public static func verifySeries(
        at directory: URL,
        singleFileNames: Set<String> = ["model.safetensors", "diffusion_pytorch_model.safetensors"],
        singleFilePrefixes: [String] = []
    ) -> (complete: Bool, missing: [String]) {
        let weights = reachableWeights(at: directory)

        if weights.contains(where: { singleFileNames.contains($0) }) { return (true, []) }
        if !singleFilePrefixes.isEmpty,
           weights.contains(where: { name in singleFilePrefixes.contains { name.hasPrefix($0) } }) {
            return (true, [])
        }

        guard !weights.isEmpty else { return (false, ["No safetensors files found"]) }

        struct Series: Hashable { let stem: String; let total: Int }
        var found: [Series: Set<Int>] = [:]
        for file in weights {
            guard let shard = shardComponents(of: file) else { continue }
            found[Series(stem: shard.stem, total: shard.total), default: []].insert(shard.index)
        }
        guard !found.isEmpty else { return (true, []) }

        // Complete if any one series is whole; otherwise report the missing
        // shards of the series closest to completion.
        var best: (series: Series, missing: [Int])?
        for (series, indices) in found {
            let missing = Set(1...series.total).subtracting(indices).sorted()
            if missing.isEmpty { return (true, []) }
            if best == nil || missing.count < best!.missing.count { best = (series, missing) }
        }
        let closest = best!
        return (false, closest.missing.map {
            "\(closest.series.stem)-\(String(format: "%05d", $0))-of-\(String(format: "%05d", closest.series.total)).safetensors"
        })
    }

    /// The exact `.safetensors` file names that make up "the" model in
    /// `directory` — the ones `verifySeries` found complete, not every
    /// reachable `.safetensors` file.
    ///
    /// A directory can accumulate a stray leftover from an earlier download
    /// or a different HF revision (e.g. a different shard count). Loading
    /// every reachable file regardless of which series it belongs to would
    /// silently merge that leftover's tensors into the result by key, with
    /// no error. Falls back to every reachable file when no recognisable
    /// series exists at all (an unusual layout) or none is complete —
    /// callers only reach here after `verifySeries`/`findModelPath` already
    /// confirmed completeness, so this is a defensive fallback, not the
    /// expected path.
    public static func filesToLoad(
        at directory: URL,
        singleFileNames: Set<String> = ["model.safetensors", "diffusion_pytorch_model.safetensors"],
        singleFilePrefixes: [String] = []
    ) -> [String] {
        let weights = reachableWeights(at: directory)

        if let single = weights.first(where: { singleFileNames.contains($0) }) {
            return [single]
        }
        if !singleFilePrefixes.isEmpty {
            let matches = weights.filter { name in singleFilePrefixes.contains { name.hasPrefix($0) } }
            if !matches.isEmpty { return matches }
        }

        struct Series: Hashable { let stem: String; let total: Int }
        var found: [Series: [Int: String]] = [:]
        for file in weights {
            guard let shard = shardComponents(of: file) else { continue }
            found[Series(stem: shard.stem, total: shard.total), default: [:]][shard.index] = file
        }
        if let complete = found.first(where: { $0.key.total == $0.value.count }) {
            return complete.value.sorted { $0.key < $1.key }.map { $0.value }
        }

        return weights
    }

    /// Splits `<stem>-NNNNN-of-MMMMM.safetensors` into its parts; `nil` for any
    /// other name (or an implausible series — no real checkpoint has thousands
    /// of shards, and the index set built from it must stay small).
    public static func shardComponents(of file: String) -> (stem: String, index: Int, total: Int)? {
        guard file.hasSuffix(".safetensors") else { return nil }
        let name = String(file.dropLast(".safetensors".count))
        let parts = name.split(separator: "-", omittingEmptySubsequences: false)
        guard parts.count >= 4,
              parts[parts.count - 2] == "of",
              let index = Int(parts[parts.count - 3]),
              let total = Int(parts[parts.count - 1]),
              total > 0, total <= 10_000, index >= 1, index <= total else {
            return nil
        }
        return (parts.dropLast(3).joined(separator: "-"), index, total)
    }

    private static func isWeightName(_ name: String) -> Bool {
        name.hasSuffix(".safetensors") && !name.hasPrefix("._")
    }

    /// The path exists (`stat` works) but can't be listed as a directory —
    /// under App Sandbox a missing security scope, otherwise a plain file
    /// sitting where a model directory belongs.
    public static func isDirectoryUnreadable(_ directory: URL) -> Bool {
        let fm = FileManager.default
        var isDir: ObjCBool = false
        guard fm.fileExists(atPath: directory.path, isDirectory: &isDir), isDir.boolValue else {
            return false
        }
        return (try? fm.contentsOfDirectory(atPath: directory.path)) == nil
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
    public static func isOnUnmountedVolume(_ url: URL) -> Bool {
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
}
