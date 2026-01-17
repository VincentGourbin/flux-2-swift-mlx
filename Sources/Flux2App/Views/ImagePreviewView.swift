// ImagePreviewView.swift - Generated image preview
// Copyright 2025 Vincent Gourbin

import SwiftUI
import UniformTypeIdentifiers

struct ImagePreviewView: View {
    let image: NSImage?

    @State private var zoomLevel: Double = 1.0
    @State private var showingSaveDialog = false

    var body: some View {
        VStack {
            if let image = image {
                imageView(image)
            } else {
                placeholderView
            }
        }
        .background(Color(NSColor.windowBackgroundColor))
    }

    // MARK: - Image View

    private func imageView(_ image: NSImage) -> some View {
        VStack(spacing: 0) {
            // Toolbar
            HStack {
                // Zoom controls
                HStack(spacing: 8) {
                    Button(action: { zoomLevel = max(0.25, zoomLevel - 0.25) }) {
                        Image(systemName: "minus.magnifyingglass")
                    }
                    .buttonStyle(.borderless)

                    Text("\(Int(zoomLevel * 100))%")
                        .font(.caption)
                        .frame(width: 40)

                    Button(action: { zoomLevel = min(4.0, zoomLevel + 0.25) }) {
                        Image(systemName: "plus.magnifyingglass")
                    }
                    .buttonStyle(.borderless)

                    Button(action: { zoomLevel = 1.0 }) {
                        Text("100%")
                            .font(.caption)
                    }
                    .buttonStyle(.bordered)
                    .controlSize(.small)

                    Button(action: { fitToWindow() }) {
                        Text("Fit")
                            .font(.caption)
                    }
                    .buttonStyle(.bordered)
                    .controlSize(.small)
                }

                Spacer()

                // Image info
                Text("\(Int(image.size.width)) × \(Int(image.size.height))")
                    .font(.caption)
                    .foregroundColor(.secondary)

                Spacer()

                // Actions
                Button(action: copyToClipboard) {
                    Image(systemName: "doc.on.clipboard")
                }
                .buttonStyle(.borderless)
                .help("Copy to clipboard")

                Button(action: { showingSaveDialog = true }) {
                    Image(systemName: "square.and.arrow.down")
                }
                .buttonStyle(.borderless)
                .help("Save image")
            }
            .padding(.horizontal)
            .padding(.vertical, 8)
            .background(Color(NSColor.controlBackgroundColor))

            Divider()

            // Image
            ScrollView([.horizontal, .vertical]) {
                Image(nsImage: image)
                    .resizable()
                    .aspectRatio(contentMode: .fit)
                    .frame(
                        width: image.size.width * zoomLevel,
                        height: image.size.height * zoomLevel
                    )
                    .padding()
            }
        }
        .fileExporter(
            isPresented: $showingSaveDialog,
            document: ImageDocument(image: image),
            contentType: .png,
            defaultFilename: "flux2_output"
        ) { result in
            // Handle result if needed
        }
    }

    // MARK: - Placeholder

    private var placeholderView: some View {
        VStack(spacing: 16) {
            Image(systemName: "photo")
                .font(.system(size: 64))
                .foregroundColor(.secondary)

            Text("Generated image will appear here")
                .font(.headline)
                .foregroundColor(.secondary)

            Text("Enter a prompt and click Generate")
                .font(.subheadline)
                .foregroundColor(.secondary)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }

    // MARK: - Actions

    private func copyToClipboard() {
        guard let image = image else { return }
        NSPasteboard.general.clearContents()
        NSPasteboard.general.writeObjects([image])
    }

    private func fitToWindow() {
        // Reset zoom to fit
        zoomLevel = 1.0
    }
}

// MARK: - Image Document for Export

struct ImageDocument: FileDocument {
    static var readableContentTypes: [UTType] { [.png, .jpeg] }

    let image: NSImage

    init(image: NSImage) {
        self.image = image
    }

    init(configuration: ReadConfiguration) throws {
        guard let data = configuration.file.regularFileContents,
              let image = NSImage(data: data) else {
            throw CocoaError(.fileReadCorruptFile)
        }
        self.image = image
    }

    func fileWrapper(configuration: WriteConfiguration) throws -> FileWrapper {
        guard let tiffData = image.tiffRepresentation,
              let bitmap = NSBitmapImageRep(data: tiffData),
              let data = bitmap.representation(using: .png, properties: [:]) else {
            throw CocoaError(.fileWriteUnknown)
        }
        return FileWrapper(regularFileWithContents: data)
    }
}

#Preview {
    ImagePreviewView(image: nil)
}
