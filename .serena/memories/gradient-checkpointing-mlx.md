# Gradient Checkpointing in MLX

## Disponibilité

| Platform | Disponible | API |
|----------|-----------|-----|
| Python | ✅ Oui | `mx.checkpoint(fun)` |
| C | ✅ Oui | `mlx_checkpoint(res, fun)` |
| Swift | ❌ Non (MLX Swift 0.30.2) | À implémenter |

## Comment ça marche

Le gradient checkpointing permet d'économiser la mémoire pendant le training en:
1. **Forward pass**: Ne pas stocker les activations intermédiaires
2. **Backward pass**: Recalculer les activations à partir des inputs sauvegardés

**Trade-off**: Plus de calcul ↔ Moins de mémoire

## Usage Python

```python
import mlx.core as mx
from mlx.nn.utils import checkpoint

# Wrapper pour un module
layer = checkpoint(layer)

# Ou décorateur
@mx.checkpoint
def my_function(x):
    ...
```

## Référence mlx-lm

PR de référence: https://github.com/ml-explore/mlx-lm/pull/167

```python
def checkpointed_fn(model, *args, **kwargs):
    def inner_fn(params, *args, **kwargs):
        model.update(params)
        return fn(model, *args, **kwargs)
    
    return mx.checkpoint(inner_fn)(
        model.trainable_parameters(), *args, **kwargs
    )
```

## Implémentation pour Swift

Pour ajouter le support Swift, il faudrait:
1. Créer un wrapper dans `Transforms.swift` autour de `mlx_checkpoint`
2. Signature suggérée:
   ```swift
   public func checkpoint<T>(_ fn: @escaping ([MLXArray]) -> [T]) -> ([MLXArray]) -> [T]
   ```

## Cas d'usage pour LoRA Training Flux.2

- Sans checkpoint: Max ~768x768 pixels sur 96GB M3 Max
- Avec checkpoint: Permettrait ~1024x1024+ (estimation)

## Économies mémoire typiques

- ~30-50% de réduction de la mémoire peak pendant le backward pass
- Coût: ~20-30% de temps de calcul supplémentaire
