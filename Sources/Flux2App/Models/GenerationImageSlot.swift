/**
 * GenerationImageSlot.swift
 * In-memory image tab model for the Images palette.
 */

import CoreGraphics
import Flux2Core
import Foundation

#if canImport(AppKit)
import AppKit
#endif

struct GenerationImageSlot: Identifiable, Equatable {
    let id: UUID
    var role: GenerationImageRole
    var isPrimary: Bool
    var url: URL?
    var image: CGImage?
    var thumbnail: NSImage?
    var sizingFavor: ImageSizingFavor
    var sizingMethod: ImageSizingMethod
    var preparationScale: Double

    var hasImage: Bool { image != nil }

    static func empty(id: UUID = UUID()) -> GenerationImageSlot {
        GenerationImageSlot(
            id: id,
            role: .unassigned,
            isPrimary: false,
            url: nil,
            image: nil,
            thumbnail: nil,
            sizingFavor: .original,
            sizingMethod: .crop,
            preparationScale: 1.0
        )
    }

    var tabTitle: String {
        if isPrimary { return "Primary" }
        if let badge = role.tabBadge { return badge }
        return "Image"
    }

    var roleBadge: String? {
        if isPrimary { return "Primary" }
        return role.tabBadge
    }
}

extension GenerationImageSlot {
    func toReferenceImage() -> ReferenceImage? {
        guard let image, let thumbnail else { return nil }
        return ReferenceImage(id: id, url: url, image: image, thumbnail: thumbnail)
    }

    func toProjectRecord(pngBase64: String?) -> GenerationImageRecord {
        GenerationImageRecord(
            id: id,
            role: role,
            isPrimary: isPrimary,
            sourcePath: url?.path,
            pngBase64: pngBase64,
            formatting: ImageSlotFormatting(
                sizingFavor: sizingFavor.rawValue,
                sizingMethod: sizingMethod.rawValue,
                preparationScale: preparationScale
            )
        )
    }

    static func fromProjectRecord(
        _ record: GenerationImageRecord,
        cgImage: CGImage?,
        thumbnail: NSImage?
    ) -> GenerationImageSlot {
        let formatting = record.formatting
        return GenerationImageSlot(
            id: record.id,
            role: record.role,
            isPrimary: record.isPrimary,
            url: record.sourcePath.map { URL(fileURLWithPath: $0) },
            image: cgImage,
            thumbnail: thumbnail,
            sizingFavor: ImageSizingFavor(rawValue: formatting.sizingFavor) ?? .original,
            sizingMethod: ImageSizingMethod(rawValue: formatting.sizingMethod) ?? .crop,
            preparationScale: max(0.1, min(1.0, formatting.preparationScale ?? 1.0))
        )
    }
}
