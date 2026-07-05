import Flux2Core
import SwiftUI

/// Canvas tools fixed at the top of the I2I palette column.
struct ImageToImageCanvasToolsSidebar: View {
    @ObservedObject var viewModel: ImageGenerationViewModel

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            SelectionToolBar(
                selectedTool: $viewModel.inpaintMaskTool,
                onSelect: { viewModel.selectMaskTool($0) },
                isToolEnabled: { viewModel.isMaskToolEnabled($0) },
                layout: .horizontal
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
