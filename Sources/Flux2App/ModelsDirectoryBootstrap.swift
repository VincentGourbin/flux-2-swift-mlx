/**
 * ModelsDirectoryBootstrap.swift
 * Applies F2SM_MODELS_DIR and Tart VM shared-model mounts before ModelManager runs.
 */

import Flux2Core
import FluxTextEncoders

enum ModelsDirectoryBootstrap {
    static func apply() {
        ModelRegistry.applyLaunchModelsDirectoryOverride()
        guard let custom = ModelRegistry.customModelsDirectory else { return }
        TextEncoderModelDownloader.customModelsDirectory = custom
        TextEncoderModelDownloader.reconfigureHubApi()
    }
}
