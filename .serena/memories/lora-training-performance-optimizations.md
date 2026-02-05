# LoRA Training Performance Optimizations

## Date: 2026-02-03
## Commit: eb144ae

## Problèmes résolus

### 1. CPU à 118% au lieu de GPU
**Cause**: Boucle séquentielle dans `clipGradNorm()`
```swift
// PROBLÈME: Crée un graph de calcul séquentiel énorme
for grad in allGrads {
    totalNormSq = totalNormSq + (grad * grad).sum()
}
```

**Solution**: Opérations batch
```swift
// SOLUTION: Calculs parallèles puis réduction
let squaredNorms = allGrads.map { ($0 * $0).sum() }
let totalNormSq = MLX.stacked(squaredNorms).sum()
eval(clipCoef)  // Un seul eval avant d'appliquer
```

### 2. Text encoder pendant le training (~4-6GB gaspillés)
**Solution**: Pré-cache tous les embeddings AVANT le training
```swift
try await preCacheTextEmbeddings(textEncoder: textEncoder)
self.validationTextEncoder = nil  // Libère la mémoire
```

### 3. eval() fragmentés
**Problème**: Multiples eval() intermédiaires
**Solution**: Suivre la doc MLX
```swift
// UN SEUL eval à la fin du step
eval(transformer.trainableParameters())
eval(loss)
```

### 4. OOM sur grandes résolutions
**Solution**: Limiter les bucket resolutions à 512, 768 (pas 1024)

## Résultats

| Métrique | Avant | Après |
|----------|-------|-------|
| CPU | 118% | ~15% |
| ETA | 48h | 10-15h |
| Crash | Step 16 | Stable |

## Fichiers clés

- `Sources/Flux2Core/Training/Loop/LoRATrainer.swift`: Optimisations principales
- `groot_training.yaml`: Config de référence avec résolutions limitées

## Pattern MLX recommandé

```swift
// Dans trainStep():
let (losses, grads) = lossAndGrad(transformer, inputArrays)
let filteredGrads = filterLoRAGradients(grads)
let clippedGrads = clipGradNorm(filteredGrads, maxNorm: config.maxGradNorm)
optimizer.update(model: transformer, gradients: clippedGrads)

// SINGLE eval following MLX docs
eval(transformer.trainableParameters())
eval(loss)
let lossValue = loss.item(Float.self)

MLX.Memory.clearCache()
```

## Améliorations futures possibles

1. **Gradient checkpointing** (Swift wrapper pour `mlx_checkpoint`)
   - Permettrait 1024+ résolutions
   - Voir: `.serena/memories/gradient-checkpointing-mlx.md`

2. **mx.compile** pour le training step
   - Fusion des opérations
   - Potentiellement 2-3x plus rapide

3. **Flash attention** si disponible en MLX Swift
