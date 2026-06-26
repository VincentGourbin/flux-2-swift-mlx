import Flux2Core
import SwiftUI

#if canImport(AppKit)
import AppKit
#endif

/// History block for the app-shell sidebar `List`, directly under the Mode section.
struct EditHistorySidebarSection: View {
    @ObservedObject var viewModel: ImageGenerationViewModel
    @ObservedObject var historyStore: EditHistoryStore
    @State private var showClearConfirmation = false

    var body: some View {
        Section {
            if historyStore.entries.isEmpty {
                Text("Import or generate to build history. Click a step to restore preview and primary reference.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
                    .listRowSeparator(.hidden)
            } else {
                ForEach(Array(historyStore.entries.enumerated()), id: \.element.id) { index, entry in
                    EditHistoryRow(
                        index: index,
                        entry: entry,
                        isCurrent: historyStore.currentIndex == index,
                        thumbnail: historyStore.thumbNSImage(for: entry)
                    ) {
                        viewModel.jumpToHistory(at: index)
                    }
                    .listRowInsets(EdgeInsets(top: 2, leading: 6, bottom: 2, trailing: 6))
                    .listRowSeparator(.hidden)
                    .listRowBackground(
                        historyStore.currentIndex == index
                            ? Color.accentColor.opacity(0.12)
                            : Color.clear
                    )
                }

                if historyStore.entries.count >= EditHistoryStore.maxEntryCount {
                    Text("Oldest steps drop off at \(EditHistoryStore.maxEntryCount) entries.")
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                        .listRowSeparator(.hidden)
                }
            }
        } header: {
            HStack(spacing: 6) {
                Text("History")
                Spacer(minLength: 0)
                if !historyStore.entries.isEmpty {
                    Text("\(historyStore.entries.count)")
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                    Button("Clear…") {
                        showClearConfirmation = true
                    }
                    .buttonStyle(.borderless)
                    .controlSize(.small)
                    .font(.caption)
                }
            }
        }
        .confirmationDialog(
            "Clear all history steps?",
            isPresented: $showClearConfirmation,
            titleVisibility: .visible
        ) {
            Button("Clear History", role: .destructive) {
                viewModel.clearEditHistory()
            }
            Button("Cancel", role: .cancel) {}
        } message: {
            Text("This removes every saved step from the project. The current canvas is unchanged until you save.")
        }
    }
}

private struct EditHistoryRow: View {
    let index: Int
    let entry: EditHistoryEntry
    let isCurrent: Bool
    let thumbnail: NSImage?
    let onSelect: () -> Void

    var body: some View {
        Button(action: onSelect) {
            HStack(spacing: 8) {
                Image(systemName: isCurrent ? "largecircle.fill.circle" : "circle")
                    .font(.caption)
                    .foregroundStyle(isCurrent ? Color.accentColor : .secondary)

                if let thumbnail {
                    Image(nsImage: thumbnail)
                        .resizable()
                        .aspectRatio(contentMode: .fill)
                        .frame(width: 44, height: 44)
                        .clipShape(RoundedRectangle(cornerRadius: 4))
                } else {
                    RoundedRectangle(cornerRadius: 4)
                        .fill(Color.secondary.opacity(0.15))
                        .frame(width: 44, height: 44)
                        .overlay {
                            Image(systemName: "photo")
                                .font(.caption)
                                .foregroundStyle(.secondary)
                        }
                }

                VStack(alignment: .leading, spacing: 2) {
                    Text("Step \(index + 1) · \(entry.label)")
                        .font(.caption.weight(.medium))
                        .lineLimit(1)
                    Text(entry.kind.displayName)
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                }

                Spacer(minLength: 0)
            }
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
    }
}

#if DEBUG
#Preview {
    List {
        Section("Mode") {
            Label("Image to Image", systemImage: "photo.on.rectangle.angled")
        }
        EditHistorySidebarSection(
            viewModel: ImageGenerationViewModel(workflow: .imageToImage),
            historyStore: EditHistoryStore()
        )
    }
    .listStyle(.sidebar)
    .frame(width: 240, height: 360)
}
#endif
