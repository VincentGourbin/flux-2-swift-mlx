/**
 * PreviewZoomableImageView.swift
 * Scrollable preview with keyboard (+/−) and trackpad pinch zoom.
 */

import SwiftUI

#if canImport(AppKit)
import AppKit
#endif

struct PreviewZoomableImageView: View {
    let image: CGImage
    @Binding var zoomScale: CGFloat

    @FocusState private var isFocused: Bool
    @State private var gestureBaseZoom: CGFloat = 1.0

    private static let minZoom: CGFloat = 0.25
    private static let maxZoom: CGFloat = 8.0
    private static let zoomStep: CGFloat = 0.25

    var body: some View {
        GeometryReader { geometry in
            let fitted = fittedSize(for: image, in: geometry.size)
            let displayWidth = max(1, fitted.width * zoomScale)
            let displayHeight = max(1, fitted.height * zoomScale)

            ScrollView([.horizontal, .vertical]) {
                Image(decorative: image, scale: 1.0)
                    .resizable()
                    .interpolation(.high)
                    .antialiased(true)
                    .frame(width: displayWidth, height: displayHeight)
            }
            .frame(width: geometry.size.width, height: geometry.size.height)
            .background(Color(nsColor: .windowBackgroundColor))
            .focusable()
            .focused($isFocused)
            .onAppear {
                isFocused = true
                gestureBaseZoom = zoomScale
            }
            .onChange(of: zoomScale) { _, newValue in
                gestureBaseZoom = newValue
            }
            .onTapGesture {
                isFocused = true
            }
            .gesture(magnificationGesture)
            .onKeyPress(keys: [.init("+"), .init("=")]) { _ in
                adjustZoom(by: Self.zoomStep)
                return .handled
            }
            .onKeyPress(keys: [.init("-"), .init("_")]) { _ in
                adjustZoom(by: -Self.zoomStep)
                return .handled
            }
            .onKeyPress(keys: [.init("0")]) { _ in
                zoomScale = 1.0
                gestureBaseZoom = 1.0
                return .handled
            }
        }
    }

    private var magnificationGesture: some Gesture {
        MagnificationGesture()
            .onChanged { value in
                zoomScale = clampZoom(gestureBaseZoom * value)
            }
            .onEnded { value in
                zoomScale = clampZoom(gestureBaseZoom * value)
                gestureBaseZoom = zoomScale
            }
    }

    private func adjustZoom(by delta: CGFloat) {
        zoomScale = clampZoom(zoomScale + delta)
        gestureBaseZoom = zoomScale
    }

    private func clampZoom(_ value: CGFloat) -> CGFloat {
        min(max(value, Self.minZoom), Self.maxZoom)
    }

    private func fittedSize(for image: CGImage, in container: CGSize) -> CGSize {
        guard image.width > 0, image.height > 0, container.width > 0, container.height > 0 else {
            return CGSize(width: 1, height: 1)
        }

        let imageAspect = CGFloat(image.width) / CGFloat(image.height)
        let containerAspect = container.width / container.height

        if imageAspect > containerAspect {
            let width = container.width
            return CGSize(width: width, height: width / imageAspect)
        }

        let height = container.height
        return CGSize(width: height * imageAspect, height: height)
    }
}
