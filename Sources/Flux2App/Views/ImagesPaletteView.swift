/**
 * ImagesPaletteView.swift
 * Tabbed Images palette — roles, primary, per-tab formatting.
 */

import SwiftUI
import Flux2Core
import UniformTypeIdentifiers
import CoreGraphics

#if canImport(AppKit)
import AppKit
#endif

struct ImagesPaletteView: View {
    @ObservedObject var viewModel: ImageGenerationViewModel
    @State private var isTargetedForDrop = false
    @State private var renameSlotID: UUID?
    @State private var renameText = ""

    private static let imagePreviewMaxHeight: CGFloat = 256

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("One image per tab. Assign Reference or Interpret (VLM) roles before generating. Live Area, selections, and crop apply to the Primary reference only.")
                .font(.caption)
                .foregroundStyle(.secondary)

            imageTabBar

            if let slot = viewModel.activeImageSlot {
                activeSlotContent(slot)
            }

            if viewModel.assignedImageCount > 0 {
                Divider()
                Text(viewModel.imageAssignmentSummary)
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
        }
        .alert("Rename Tab", isPresented: renameAlertPresented) {
            TextField("Tab name", text: $renameText)
            Button("Save") {
                if let renameSlotID {
                    viewModel.setSlotTabLabel(renameSlotID, label: renameText)
                }
                self.renameSlotID = nil
            }
            Button("Cancel", role: .cancel) {
                renameSlotID = nil
            }
        } message: {
            Text("Enter a custom name for this image tab.")
        }
    }

    private var renameAlertPresented: Binding<Bool> {
        Binding(
            get: { renameSlotID != nil },
            set: { if !$0 { renameSlotID = nil } }
        )
    }

    private var imageTabBar: some View {
        HStack(spacing: 8) {
            ScrollView(.horizontal, showsIndicators: false) {
                HStack(spacing: 4) {
                    ForEach(Array(viewModel.imageSlots.enumerated()), id: \.element.id) { index, slot in
                        ImagePillTab(
                            title: slot.displayTabTitle(index: index),
                            isSelected: viewModel.activeImageSlot?.id == slot.id,
                            onSelect: { viewModel.selectImageSlot(slot.id) },
                            onClose: viewModel.imageSlots.count > 1 ? { viewModel.removeImageSlot(slot.id) } : nil,
                            onRename: {
                                renameText = slot.customTabLabel ?? slot.displayTabTitle(index: index)
                                renameSlotID = slot.id
                            },
                            onRemove: viewModel.imageSlots.count > 1 ? { viewModel.removeImageSlot(slot.id) } : nil
                        )
                    }
                }
                .padding(4)
            }
            .background(
                Capsule()
                    .fill(Color(nsColor: .quaternaryLabelColor).opacity(0.45))
            )

            if viewModel.canAddImageSlot {
                Button(action: { viewModel.addImageSlot() }) {
                    Image(systemName: "plus")
                        .font(.system(size: 13, weight: .bold))
                        .foregroundStyle(.white)
                        .frame(width: 28, height: 28)
                        .background(Circle().fill(Color(nsColor: .secondaryLabelColor)))
                }
                .buttonStyle(.plain)
                .help("Add image tab (max \(FluxGenerationProject.maxImageSlots))")
            }
        }
    }

    @ViewBuilder
    private func activeSlotContent(_ slot: GenerationImageSlot) -> some View {
        if let cgImage = slot.image {
            ImagesPaletteImagePreview(cgImage: cgImage) {
                viewModel.clearImageSlot(slot.id)
            }
        } else {
            AddImageSlot(edge: Self.imagePreviewMaxHeight, onAdd: { selectImage(for: slot.id) })
                .frame(maxWidth: .infinity)
                .onDrop(of: [.image], isTargeted: $isTargetedForDrop) { providers in
                    handleImageDrop(providers, slotID: slot.id)
                    return true
                }
        }

        HStack(alignment: .center, spacing: 8) {
            Picker("Role", selection: roleBinding(for: slot.id)) {
                ForEach(GenerationImageRole.allCases) { role in
                    Text(role.displayName).tag(role)
                }
            }
            .pickerStyle(.menu)
            .fixedSize()

            Toggle(
                "Primary",
                isOn: primaryBinding(for: slot.id)
            )
            .disabled(slot.role != .reference)
            .fixedSize()

            Spacer(minLength: 8)

            Button(action: { viewModel.saveInputImage() }) {
                Label("Save", systemImage: "square.and.arrow.down.on.square")
            }
            .buttonStyle(.bordered)
            .controlSize(.small)
            .disabled(!canSaveInput(for: slot))
            .help("Save the selected input variant (raw, formatted, or prepared)")

            Picker("Stage", selection: $viewModel.inputSaveSource) {
                ForEach(ImageInputSaveSource.allCases) { source in
                    Text(source.menuLabel).tag(source)
                }
            }
            .pickerStyle(.menu)
            .controlSize(.small)
            .fixedSize()
            .disabled(!canSaveInput(for: slot))
            .help("Input variant for Save: raw reference, formatted (crop/pad), or prepared (model input)")
        }

        Divider()

        if slot.hasImage {
            ImageSlotAspectRatioFormattingView(
                sizingFavor: favorBinding(for: slot.id),
                sizingMethod: methodBinding(for: slot.id),
                preparationScale: scaleBinding(for: slot.id),
                showScalingControl: false,
                onChanged: { viewModel.applySizingControlsForPreview() }
            )
        } else {
            Text("Add an image to set aspect ratio formatting for this tab.")
                .font(.caption)
                .foregroundStyle(.secondary)
        }

        if slot.isPrimary, slot.role == .reference, slot.hasImage {
            UpsizeControlView(viewModel: viewModel)
        }

        if !viewModel.isSpatialEditingActive, slot.hasImage {
            Text("Select the Primary reference tab to use Live Area and canvas tools on the preview.")
                .font(.caption2)
                .foregroundStyle(.orange)
        }
    }

    private func canSaveInput(for slot: GenerationImageSlot) -> Bool {
        slot.isPrimary && slot.role == .reference && slot.hasImage
    }

    private func roleBinding(for slotID: UUID) -> Binding<GenerationImageRole> {
        Binding(
            get: { viewModel.imageSlots.first(where: { $0.id == slotID })?.role ?? .unassigned },
            set: { viewModel.setSlotRole(slotID, role: $0) }
        )
    }

    private func primaryBinding(for slotID: UUID) -> Binding<Bool> {
        Binding(
            get: { viewModel.imageSlots.first(where: { $0.id == slotID })?.isPrimary ?? false },
            set: { viewModel.setSlotPrimary(slotID, isPrimary: $0) }
        )
    }

    private func favorBinding(for slotID: UUID) -> Binding<ImageSizingFavor> {
        Binding(
            get: { viewModel.imageSlots.first(where: { $0.id == slotID })?.sizingFavor ?? .original },
            set: { viewModel.setSlotSizingFavor(slotID, favor: $0) }
        )
    }

    private func methodBinding(for slotID: UUID) -> Binding<ImageSizingMethod> {
        Binding(
            get: { viewModel.imageSlots.first(where: { $0.id == slotID })?.sizingMethod ?? .crop },
            set: { viewModel.setSlotSizingMethod(slotID, method: $0) }
        )
    }

    private func scaleBinding(for slotID: UUID) -> Binding<Double> {
        Binding(
            get: { viewModel.imageSlots.first(where: { $0.id == slotID })?.preparationScale ?? 1.0 },
            set: { viewModel.setSlotPreparationScale(slotID, scale: $0) }
        )
    }

    private func selectImage(for slotID: UUID) {
        let panel = NSOpenPanel()
        panel.allowedContentTypes = [.image]
        panel.allowsMultipleSelection = false
        panel.canChooseFiles = true
        panel.canChooseDirectories = false

        if panel.runModal() == .OK, let url = panel.url {
            viewModel.loadImageIntoSlot(slotID, from: url)
        }
    }

    private func handleImageDrop(_ providers: [NSItemProvider], slotID: UUID) {
        guard let provider = providers.first, provider.canLoadObject(ofClass: NSImage.self) else { return }
        _ = provider.loadObject(ofClass: NSImage.self) { image, _ in
            if let nsImage = image as? NSImage {
                DispatchQueue.main.async {
                    viewModel.loadImageIntoSlot(slotID, from: nsImage)
                }
            }
        }
    }
}

private struct ImagePillTab: View {
    let title: String
    let isSelected: Bool
    let onSelect: () -> Void
    let onClose: (() -> Void)?
    var onRename: (() -> Void)?
    var onRemove: (() -> Void)?

    var body: some View {
        HStack(spacing: 6) {
            Text(title)
                .font(.system(size: 12, weight: isSelected ? .semibold : .regular))
                .foregroundStyle(isSelected ? Color.primary : Color.secondary)
                .lineLimit(1)
                .onTapGesture(perform: onSelect)

            if let onClose, isSelected {
                Button(action: onClose) {
                    Image(systemName: "xmark")
                        .font(.system(size: 9, weight: .bold))
                        .foregroundStyle(.secondary)
                }
                .buttonStyle(.plain)
            }
        }
        .padding(.horizontal, 14)
        .padding(.vertical, 7)
        .background(
            Capsule()
                .fill(isSelected ? Color(nsColor: .tertiaryLabelColor) : Color.clear)
        )
        .contentShape(Capsule())
        .onTapGesture(perform: onSelect)
        .contextMenu {
            if let onRename {
                Button("Rename Tab…") {
                    onRename()
                }
            }
            if let onRemove {
                Button("Remove Tab", role: .destructive) {
                    onRemove()
                }
            }
        }
    }
}

private struct ImagesPaletteImagePreview: View {
    let cgImage: CGImage
    let onRemove: () -> Void

    private static let maxHeight: CGFloat = 256

    var body: some View {
        Image(decorative: cgImage, scale: 1)
            .resizable()
            .aspectRatio(contentMode: .fit)
            .frame(maxWidth: .infinity)
            .frame(maxHeight: Self.maxHeight)
            .clipShape(RoundedRectangle(cornerRadius: 8))
            .overlay(
                RoundedRectangle(cornerRadius: 8)
                    .stroke(Color.accentColor.opacity(0.6), lineWidth: 1)
            )
            .overlay(alignment: .topTrailing) {
                Button(action: onRemove) {
                    Image(systemName: "xmark.circle.fill")
                        .font(.system(size: 20))
                        .symbolRenderingMode(.palette)
                        .foregroundStyle(.white, .red)
                }
                .buttonStyle(.plain)
                .padding(6)
            }
    }
}

struct ImageSlotAspectRatioFormattingView: View {
    @Binding var sizingFavor: ImageSizingFavor
    @Binding var sizingMethod: ImageSizingMethod
    @Binding var preparationScale: Double
    var showScalingControl: Bool
    var onChanged: () -> Void

    var body: some View {
        GroupBox("Aspect Ratio") {
            VStack(alignment: .leading, spacing: 10) {
                HStack(alignment: .top, spacing: 14) {
                    VStack(alignment: .leading, spacing: 6) {
                        Text("Favour")
                            .font(.caption.bold())

                        HStack(spacing: 8) {
                            ForEach(ImageSizingFavor.allCases) { favor in
                                ImagePreparationOptionButton(
                                    title: favor.rawValue,
                                    isSelected: sizingFavor == favor
                                ) {
                                    sizingFavor = favor
                                    onChanged()
                                }
                            }
                        }
                    }

                    Divider()
                        .frame(height: 34)

                    VStack(alignment: .leading, spacing: 6) {
                        Text("Method")
                            .font(.caption.bold())

                        HStack(spacing: 8) {
                            ForEach(ImageSizingMethod.allCases) { method in
                                ImagePreparationOptionButton(
                                    title: method.rawValue,
                                    isSelected: sizingMethod == method
                                ) {
                                    sizingMethod = method
                                    onChanged()
                                }
                            }
                        }
                    }
                }
                .frame(maxWidth: .infinity, alignment: .leading)

                if showScalingControl {
                    HStack(spacing: 10) {
                        Slider(
                            value: Binding(
                                get: { preparationScale },
                                set: {
                                    preparationScale = min(max($0, 0.1), 1.0)
                                    onChanged()
                                }
                            ),
                            in: 0.1...1.0,
                            step: 0.05
                        )

                        Text("\(Int((preparationScale * 100).rounded()))%")
                            .font(.caption.monospacedDigit())
                            .foregroundStyle(.secondary)
                            .frame(width: 44, alignment: .trailing)
                    }
                }
            }
        }
    }
}

/// Scale or generatively enlarge the working (primary) reference to the megapixel
/// budget. The dropdown picks a resampler or a FLUX model; Apply runs it. FLUX
/// rows grey out when the source already fills the budget (only resampling
/// applies when downscaling). The faithful enlarge prompt lives in Settings.
struct UpsizeControlView: View {
    @ObservedObject var viewModel: ImageGenerationViewModel

    var body: some View {
        GroupBox("Upsize") {
            VStack(alignment: .leading, spacing: 8) {
                HStack(spacing: 8) {
                    Menu {
                        ForEach(UpsizeMethod.allCases) { method in
                            Button {
                                viewModel.upsizeMethod = method
                            } label: {
                                if method == viewModel.upsizeMethod {
                                    Label(method.displayName, systemImage: "checkmark")
                                } else {
                                    Text(method.displayName)
                                }
                            }
                            .disabled(method.isGenerative && viewModel.isUpsizeDownscaling)
                        }
                    } label: {
                        Text(viewModel.upsizeMethod.displayName)
                            .frame(maxWidth: .infinity, alignment: .leading)
                    }
                    .frame(width: 150)

                    Button(action: { viewModel.performUpsize() }) {
                        Text("Apply")
                    }
                    .buttonStyle(.borderedProminent)
                    .controlSize(.small)
                    .disabled(!viewModel.canApplyUpsize)

                    if viewModel.isGenerating {
                        ProgressView()
                            .controlSize(.small)
                    }
                }

                Text(hint)
                    .font(.caption2)
                    .foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
            .frame(maxWidth: .infinity, alignment: .leading)
        }
    }

    private var hint: String {
        if viewModel.upsizeMethod.isGenerative {
            if viewModel.isUpsizeDownscaling {
                return "Source already fills the budget — generative enlarge isn’t available; choose Bicubic or Lanczos."
            }
            return "Replaces the working image with a FLUX-enlarged version at the megapixel budget (a full generation)."
        }
        return "Resamples the working image to the megapixel budget."
    }
}
