# Plan: Intégration des Best Practices Ostris pour LoRA Training

## Contexte
- LoRA entraîné avec Ostris (ai-toolkit) sur Klein 4B fonctionne correctement
- Learning rate 0.0001 (1e-4) validée
- Analyse comparative effectuée le 2026-02-01

## Différences Clés Identifiées

| Aspect | Ostris (ai-toolkit) | Notre implémentation MLX | Impact |
|--------|---------------------|--------------------------|--------|
| **EMA** | ✅ `use_ema: true, decay: 0.99` | ❌ Pas d'EMA | **CRITIQUE** |
| **Caption Dropout** | ✅ 5% (`caption_dropout_rate: 0.05`) | ❌ 0% | **MOYEN** |
| **Multi-Resolution** | ✅ `[512, 768, 1024]` | ❌ Single resolution | **MOYEN** |
| **Optimizer** | AdamW8bit | AdamW standard | Faible |
| **content_or_style** | ✅ "balanced" option | ❌ Absent | À investiguer |
| **linear_timesteps** | Option "vell curved weighting" | ❌ Absent | Expérimental |

## Phase 1 : EMA (Exponential Moving Average) - PRIORITÉ HAUTE

**Pourquoi c'est critique :**
- EMA lisse les poids pendant le training
- Évite les oscillations et les poids "bruiteux"
- Produit des LoRAs plus stables et de meilleure qualité

**À implémenter :**
```swift
// Ajouter à LoRATrainingConfig
public var useEMA: Bool = true
public var emaDecay: Float = 0.99

// Créer une classe EMAManager
class EMAManager {
    var shadowWeights: [String: MLXArray]
    let decay: Float
    
    func update(parameters: ModuleParameters)
    func copyToModel(model: Module)
    func saveToCheckpoint(path: URL)
}
```

## Phase 2 : Caption Dropout - PRIORITÉ MOYENNE

**À implémenter :**
```swift
// Ajouter à LoRATrainingConfig
public var captionDropoutRate: Float = 0.05

// Dans TrainingDataset ou LoRATrainer
func processCaption(_ caption: String) -> String {
    if Float.random(in: 0...1) < captionDropoutRate {
        return ""
    }
    return caption
}
```

## Phase 3 : Multi-Resolution Training - PRIORITÉ MOYENNE

**À implémenter :**
```swift
// Modifier LoRATrainingConfig
public var resolutions: [Int] = [512, 768, 1024]

// Bucketing par aspect ratio et résolution
```

## Phase 4 : Optimizer AdamW8bit (Optionnel)

## Phase 5 : Investiguer content_or_style

## Fichiers à Modifier

| Fichier | Modifications |
|---------|---------------|
| `LoRATrainingConfig.swift` | Ajouter `useEMA`, `emaDecay`, `captionDropoutRate`, `resolutions` |
| `LoRATrainer.swift` | Intégrer EMA update à chaque step, caption dropout |
| Nouveau: `EMAManager.swift` | Gestion des poids EMA |
| `TrainingDataset.swift` | Multi-resolution bucketing, caption dropout |
