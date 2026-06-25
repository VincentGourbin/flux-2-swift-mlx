import Flux2Core
import SwiftUI

/// Photoshop-style icon row for canvas tools (always visible when editing).
struct SelectionToolBar: View {
    @Binding var selectedTool: InpaintMaskTool
    var onSelect: (InpaintMaskTool) -> Void
    var isToolEnabled: (InpaintMaskTool) -> Bool = { _ in true }

    private let toolSize: CGFloat = 36

    var body: some View {
        HStack(spacing: 6) {
            ForEach(InpaintMaskTool.toolbarCases) { tool in
                let enabled = isToolEnabled(tool)
                Button {
                    if selectedTool == tool {
                        onSelect(.pointer)
                        selectedTool = .pointer
                    } else {
                        onSelect(tool)
                        selectedTool = tool
                    }
                } label: {
                    Image(systemName: tool.systemImage)
                        .font(.system(size: 15, weight: .medium))
                        .frame(width: toolSize, height: toolSize)
                        .background(
                            RoundedRectangle(cornerRadius: 6)
                                .fill(selectedTool == tool ? Color.accentColor.opacity(0.22) : Color(nsColor: .controlBackgroundColor))
                        )
                        .overlay(
                            RoundedRectangle(cornerRadius: 6)
                                .stroke(
                                    (selectedTool == tool) ? Color.accentColor : Color.secondary.opacity(0.35),
                                    lineWidth: (selectedTool == tool) ? 1.5 : 1
                                )
                        )
                }
                .buttonStyle(.plain)
                .disabled(!enabled)
                .opacity(enabled ? 1 : 0.38)
                .help(tool.displayName)
            }
        }
    }
}
