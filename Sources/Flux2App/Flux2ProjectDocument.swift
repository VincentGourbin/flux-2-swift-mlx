import Flux2Core
import UniformTypeIdentifiers

extension UTType {
    /// `.flux2project` document package (v3 bundle).
    static var flux2ProjectBundle: UTType {
        UTType(exportedAs: "com.realnotsteve.flux2-project", conformingTo: .package)
    }
}

enum Flux2ProjectDocument {
    static var allowedOpenContentTypes: [UTType] {
        [.json, .flux2ProjectBundle]
    }

    static var saveContentTypes: [UTType] {
        [.flux2ProjectBundle]
    }

    static var defaultSaveName: String {
        "Untitled.\(FluxGenerationProjectBundle.packageExtension)"
    }

    static func normalizedBundleURL(from url: URL) -> URL {
        if url.pathExtension == FluxGenerationProjectBundle.packageExtension {
            return url
        }
        return url.deletingPathExtension().appendingPathExtension(FluxGenerationProjectBundle.packageExtension)
    }

    static func isLegacyJSONProjectURL(_ url: URL) -> Bool {
        url.pathExtension == "json"
    }
}
