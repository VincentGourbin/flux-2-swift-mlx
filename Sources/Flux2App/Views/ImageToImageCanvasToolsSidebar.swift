import Flux2Core
import SwiftUI

/// Canvas tools shown in the app sidebar while Image to Image is active.
struct ImageToImageCanvasToolsSidebar: View {
    @ObservedObject var viewModel: ImageGenerationViewModel

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("Canvas Tools")
                .font(.caption.bold())
                .foregroundStyle(.secondary)

            SelectionToolBar(
                selectedTool: $viewModel.inpaintMaskTool,
                onSelect: { viewModel.selectMaskTool($0) },
                isToolEnabled: { viewModel.isMaskToolEnabled($0) },
                layout: .vertical
            )

            if !viewModel.hasPrimaryReference {
                Text("Add a primary reference image to enable the tools.")
                    .font(.caption2)
                    .foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            } else if !viewModel.isSpatialEditingActive {
                Text("Select the Primary reference tab to use Live Area and selections.")
                    .font(.caption2)
                    .foregroundStyle(.orange)
                    .fixedSize(horizontal: false, vertical: true)
            } else {
                Text("Live Area, selections, and crop.")
                    .font(.caption2)
                    .foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }
}
