# Plan d'Implémentation : Training Flux2.dev LoRA

## Status: Phase 1-3 COMPLÉTÉES ✅

**Date:** 2025-02-05

---

## Vue d'ensemble

Flux2.dev est un modèle 32B paramètres qui utilise :
- **Mistral Small 3.2 (24B)** comme text encoder (au lieu de Qwen3)
- **Guidance embeddings** (le modèle est "guidance-distilled")
- Architecture transformer plus grande (8 double + 48 single blocks)

## Différences Clés avec Klein

| Aspect | Klein 4B/9B | Dev 32B |
|--------|-------------|---------|
| Text Encoder | Qwen3 (4B/8B) | Mistral 24B |
| Guidance | Désactivé (`guidanceEmbeds: false`) | Activé (`guidanceEmbeds: true`) |
| jointAttentionDim | 7680 / 12288 | 15360 |
| innerDim | 3072 / 4096 | 6144 |
| Blocks | 5+20 / 8+24 | 8+48 |
| VRAM Training | ~13GB / ~29GB | ~60GB |

## Phases d'Implémentation

### Phase 1 : Extraction d'Embeddings Mistral pour Training ✅ COMPLÉTÉE

**Fichiers créés :**

1. **`Sources/Flux2Core/Loading/DevTextEncoder.swift`** ✅
   - Wrapper pour Mistral text encoding (similaire à KleinTextEncoder)
   - Utilise `FluxTextEncoders.shared.extractMfluxEmbeddings()` (layers 10, 20, 30)
   - Output: [1, 512, 15360] (3 × 5120)

2. **`Sources/Flux2Core/Loading/TrainingTextEncoder.swift`** ✅
   - Protocole commun pour KleinTextEncoder et DevTextEncoder
   - Méthode `encodeForTraining(_ prompt: String) throws -> MLXArray`

**Note:** L'infrastructure Mistral existait déjà dans FluxTextEncoders - juste besoin d'un wrapper.

### Phase 2 : Gestion des Guidance Embeddings dans le Training Loop ✅ COMPLÉTÉE

**Modification dans `SimpleLoRATrainer.swift` :**

```swift
// Guidance embedding (Dev model uses guidance, Klein models do NOT)
// IMPORTANT: For training, use guidance_scale=1.0 (not 4.0 used in inference)
let guidance: MLXArray? = modelType.usesGuidanceEmbeds ?
    MLXArray.full([batchSize], values: MLXArray(Float(1.0))) : nil
```

Le code gérait déjà les guidance embeddings via `modelType.usesGuidanceEmbeds`.
Changement: guidance_scale 4.0 → 1.0 pour le training (ref: Ostris).

### Phase 3 : Intégration CLI ✅ COMPLÉTÉE

**Modification dans `TrainLoRACommand.swift` :**

```swift
private func loadTextEncoder(
    for model: Flux2Model,
    quantization: TrainingQuantization
) async throws -> any TrainingTextEncoder {
    switch model {
    case .klein4B:
        let encoder = KleinTextEncoder(variant: .klein4B, quantization: mistralQuant)
        // ...
    case .klein9B:
        let encoder = KleinTextEncoder(variant: .klein9B, quantization: mistralQuant)
        // ...
    case .dev:
        // Dev uses Mistral Small 3.2 (24B)
        let encoder = DevTextEncoder(quantization: mistralQuant)
        // ...
    }
}
```

### Phase 4 : Tests et Validation ⏳ À FAIRE

**Tests requis :**

1. **Test embeddings Mistral** : Vérifier shape [1, 512, 15360]
   ```bash
   flux2 train-lora --model dev --config test_dev.yaml --max-steps 1
   ```

2. **Test loss initial** : Devrait être ~1.1-1.2
   ```bash
   flux2 train-lora --model dev --config test_dev.yaml --max-steps 10
   ```

3. **Test stabilité** : Pas de collapse à 500 steps
   - Comparer avec Klein training sur même dataset
   - Vérifier que guidance=1.0 fonctionne correctement

**Travail estimé :** 2 jours de tests

## Prérequis Matériels

| Configuration | VRAM Requise | Faisable sur |
|---------------|--------------|--------------|
| Dev bf16 + Mistral 8bit | ~60GB | Mac Studio M2 Ultra 128GB |
| Dev bf16 + Mistral 4bit | ~50GB | Mac Studio M2 Max 96GB |
| Dev qint8 + Mistral 4bit | ~45GB | Mac Studio M2 Max 64GB (serré) |

**Recommandation :** Utiliser Mistral 8-bit pour un bon compromis qualité/VRAM.

**Note :** Le training Dev n'est PAS recommandé sur machines < 64GB RAM.

## Timeline

| Phase | Status | Durée réelle |
|-------|--------|--------------|
| Phase 1 : DevTextEncoder wrapper | ✅ Complété | 0.5 jour |
| Phase 2 : Guidance training (1.0) | ✅ Complété | 0.25 jour |
| Phase 3 : CLI integration | ✅ Complété | 0.25 jour |
| Phase 4 : Tests et validation | ⏳ À faire | ~2 jours |

**Total réel : ~1 jour de dev + tests à venir**

L'infrastructure Mistral existait déjà, ce qui a considérablement réduit le travail.

## Risques et Mitigations

1. **VRAM insuffisante**
   - Mitigation : Tester d'abord avec Mistral 4bit
   - Mitigation : Utiliser gradient checkpointing

2. **Loss divergent avec guidance**
   - Mitigation : Commencer avec guidance_scale=1.0 strictement
   - Référence : Vérifier implémentation Ostris

3. **Embeddings Mistral incorrects**
   - Mitigation : Comparer avec mflux ou diffusers
   - Vérifier les layers extraits

## Références

- [Ostris ai-toolkit](https://github.com/ostris/ai-toolkit) - Implémentation de référence
- [RunComfy FLUX.2 Dev Training](https://www.runcomfy.com/trainer/ai-toolkit/flux-2-dev-lora-training)
- [HuggingFace FLUX.1-dev](https://huggingface.co/black-forest-labs/FLUX.1-dev)

## Fichiers Impactés (Résumé)

| Fichier | Action | Status |
|---------|--------|--------|
| `Sources/Flux2Core/Loading/DevTextEncoder.swift` | Créé | ✅ |
| `Sources/Flux2Core/Loading/TrainingTextEncoder.swift` | Créé (protocole) | ✅ |
| `Sources/Flux2Core/Training/Loop/SimpleLoRATrainer.swift` | Modifié (guidance=1.0) | ✅ |
| `Sources/Flux2CLI/TrainLoRACommand.swift` | Modifié (Dev support) | ✅ |
| `Sources/Flux2Core/Configuration/ModelRegistry.swift` | Modifié (Klein 9B base) | ✅ |
