/**
 * PalettePanel.swift
 * Collapsible, content-sized, optionally detached sidebar palettes.
 */

import SwiftUI

#if canImport(AppKit)
import AppKit
#endif

// MARK: - Detach coordinator

@MainActor
final class PaletteDetachCoordinator: ObservableObject {
    @Published private(set) var detachedIDs: Set<String> = []
    @Published var positions: [String: CGPoint] = [:]

    func isDetached(_ id: String) -> Bool {
        detachedIDs.contains(id)
    }

    func detach(_ id: String) {
        detachedIDs.insert(id)
        if positions[id] == nil {
            let offset = CGFloat(detachedIDs.count) * 28
            positions[id] = CGPoint(x: 96 + offset, y: 96 + offset)
        }
    }

    func dock(_ id: String) {
        detachedIDs.remove(id)
    }

    func positionBinding(for id: String) -> Binding<CGPoint> {
        Binding(
            get: { self.positions[id] ?? CGPoint(x: 96, y: 96) },
            set: { self.positions[id] = $0 }
        )
    }
}

// MARK: - Column

struct PaletteColumn<Content: View>: View {
    @ViewBuilder var content: () -> Content

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 10) {
                content()
            }
            .padding(10)
            .frame(maxWidth: .infinity, alignment: .leading)
        }
    }
}

// MARK: - Dock placeholder

struct PaletteDockPlaceholder: View {
    let title: String
    let systemImage: String
    var onDock: () -> Void

    var body: some View {
        HStack(spacing: 8) {
            Image(systemName: systemImage)
                .foregroundStyle(.secondary)
            Text(title)
                .font(.subheadline.weight(.semibold))
            Text("— floating")
                .font(.caption)
                .foregroundStyle(.secondary)
            Spacer()
            Button("Dock", action: onDock)
                .buttonStyle(.bordered)
                .controlSize(.mini)
        }
        .padding(.horizontal, 10)
        .padding(.vertical, 8)
        .background(paletteChrome)
        .clipShape(RoundedRectangle(cornerRadius: 10, style: .continuous))
        .overlay {
            RoundedRectangle(cornerRadius: 10, style: .continuous)
                .strokeBorder(Color.primary.opacity(0.08), lineWidth: 1)
        }
    }
}

// MARK: - Docked panel

struct PalettePanel<Content: View, HeaderTrailing: View>: View {
    let storageKey: String
    let title: String
    let systemImage: String
    @ObservedObject var coordinator: PaletteDetachCoordinator
    @ViewBuilder var headerTrailing: () -> HeaderTrailing
    @ViewBuilder var content: () -> Content

    @AppStorage private var isCollapsed: Bool

    init(
        storageKey: String,
        title: String,
        systemImage: String,
        coordinator: PaletteDetachCoordinator,
        @ViewBuilder headerTrailing: @escaping () -> HeaderTrailing = { EmptyView() },
        @ViewBuilder content: @escaping () -> Content
    ) {
        self.storageKey = storageKey
        self.title = title
        self.systemImage = systemImage
        self.coordinator = coordinator
        self.headerTrailing = headerTrailing
        self.content = content
        _isCollapsed = AppStorage(wrappedValue: false, "palette.\(storageKey).collapsed")
    }

    var body: some View {
        Group {
            if coordinator.isDetached(storageKey) {
                PaletteDockPlaceholder(title: title, systemImage: systemImage) {
                    coordinator.dock(storageKey)
                }
            } else {
                VStack(spacing: 0) {
                    paletteHeader(onDetach: { coordinator.detach(storageKey) })

                    if !isCollapsed {
                        content()
                            .frame(maxWidth: .infinity, alignment: .leading)
                            .padding(10)
                    }
                }
                .background(paletteChrome)
                .clipShape(RoundedRectangle(cornerRadius: 10, style: .continuous))
                .overlay {
                    RoundedRectangle(cornerRadius: 10, style: .continuous)
                        .strokeBorder(Color.primary.opacity(0.08), lineWidth: 1)
                }
                .shadow(color: .black.opacity(0.06), radius: 3, y: 1)
            }
        }
    }

    private func paletteHeader(onDetach: @escaping () -> Void) -> some View {
        HStack(spacing: 8) {
            Button {
                withAnimation(.easeInOut(duration: 0.18)) {
                    isCollapsed.toggle()
                }
            } label: {
                Image(systemName: isCollapsed ? "chevron.right" : "chevron.down")
                    .font(.caption.weight(.semibold))
                    .foregroundStyle(.secondary)
                    .frame(width: 14)
            }
            .buttonStyle(.plain)

            Image(systemName: systemImage)
                .font(.subheadline)
                .foregroundStyle(.secondary)
            Text(title)
                .font(.subheadline.weight(.semibold))

            headerTrailing()

            Spacer(minLength: 0)

            Button(action: onDetach) {
                Image(systemName: "rectangle.portrait.topthird.inset.filled")
                    .font(.caption)
            }
            .buttonStyle(.plain)
            .foregroundStyle(.secondary)
            .help("Float this palette")
        }
        .padding(.horizontal, 10)
        .padding(.vertical, 8)
        .background(Color(nsColor: .controlBackgroundColor).opacity(0.65))
    }
}

// MARK: - Floating panel

struct PaletteFloatingPanel<Content: View, HeaderTrailing: View>: View {
    let storageKey: String
    let title: String
    let systemImage: String
    @Binding var position: CGPoint
    @ObservedObject var coordinator: PaletteDetachCoordinator
    @ViewBuilder var headerTrailing: () -> HeaderTrailing
    @ViewBuilder var content: () -> Content

    @AppStorage private var isCollapsed: Bool
    @State private var dragOrigin: CGPoint?

    init(
        storageKey: String,
        title: String,
        systemImage: String,
        position: Binding<CGPoint>,
        coordinator: PaletteDetachCoordinator,
        @ViewBuilder headerTrailing: @escaping () -> HeaderTrailing = { EmptyView() },
        @ViewBuilder content: @escaping () -> Content
    ) {
        self.storageKey = storageKey
        self.title = title
        self.systemImage = systemImage
        _position = position
        self.coordinator = coordinator
        self.headerTrailing = headerTrailing
        self.content = content
        _isCollapsed = AppStorage(wrappedValue: false, "palette.\(storageKey).collapsed")
    }

    var body: some View {
        let width: CGFloat = 380
        VStack(spacing: 0) {
            floatingHeader

            if !isCollapsed {
                content()
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding(10)
            }
        }
        .frame(width: width)
        .background(paletteChrome)
        .clipShape(RoundedRectangle(cornerRadius: 10, style: .continuous))
        .overlay {
            RoundedRectangle(cornerRadius: 10, style: .continuous)
                .strokeBorder(Color.primary.opacity(0.12), lineWidth: 1)
        }
        .shadow(color: .black.opacity(0.18), radius: 10, y: 4)
        .offset(x: position.x, y: position.y)
    }

    private var floatingHeader: some View {
        HStack(spacing: 8) {
            Button {
                withAnimation(.easeInOut(duration: 0.18)) {
                    isCollapsed.toggle()
                }
            } label: {
                Image(systemName: isCollapsed ? "chevron.right" : "chevron.down")
                    .font(.caption.weight(.semibold))
                    .foregroundStyle(.secondary)
                    .frame(width: 14)
            }
            .buttonStyle(.plain)

            Image(systemName: systemImage)
                .font(.subheadline)
                .foregroundStyle(.secondary)
            Text(title)
                .font(.subheadline.weight(.semibold))

            headerTrailing()

            Spacer(minLength: 0)

            Button {
                coordinator.dock(storageKey)
            } label: {
                Image(systemName: "arrow.down.right.and.arrow.up.left")
                    .font(.caption)
            }
            .buttonStyle(.plain)
            .foregroundStyle(.secondary)
            .help("Dock back into the sidebar")
        }
        .padding(.horizontal, 10)
        .padding(.vertical, 8)
        .background(Color(nsColor: .controlBackgroundColor).opacity(0.8))
        .gesture(
            DragGesture()
                .onChanged { value in
                    if dragOrigin == nil {
                        dragOrigin = position
                    }
                    let origin = dragOrigin ?? position
                    position = CGPoint(
                        x: origin.x + value.translation.width,
                        y: origin.y + value.translation.height
                    )
                }
                .onEnded { _ in
                    dragOrigin = nil
                }
        )
    }
}

private var paletteChrome: some View {
    RoundedRectangle(cornerRadius: 10, style: .continuous)
        .fill(Color(nsColor: .windowBackgroundColor).opacity(0.92))
}
