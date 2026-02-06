# Plan d'Optimisation Mémoire pour l'Inférence

## Résumé de l'Analyse

### Références Étudiées
| Projet | Techniques Trouvées |
|--------|---------------------|
| **Issue #31** | `Memory.cacheLimit` + `eval()` + `clearCache()` pattern complet |
| **mzbac/flux2.swift** | Simple `MLX.eval()` à la fin de génération |
| **LTX-2-MLX** | `mx.eval(latent)` dans la boucle de denoising "to prevent memory buildup" |
| **mflux** | `prompt_cache` pour mise en cache des embeddings |

### État Actuel de Notre Codebase
| Composant | Memory.cacheLimit | clearCache | eval() |
|-----------|-------------------|------------|--------|
| Klein Text Encoder | ✅ 512 MB | ✅ | ✅ |
| Mistral/Dev Text Encoder | ❌ | ❌ | ✅ |
| Transformer (denoising) | ❌ | ❌ | ✅ |
| VAE Tiled | ❌ | ✅ par tile | ✅ |
| Pipeline global | ❌ | ✅ fin phase | ✅ |

**Constat : Seul le Klein text encoder a une limite de cache.**

---

## Plan d'Implémentation

### Phase 1: Memory Configuration Centralisée

**Fichier: `Sources/Flux2Core/Configuration/MemoryConfig.swift`**

Configuration centralisée avec profils auto-détectés selon la RAM système.

### Phase 2: Intégration dans le Pipeline (CRITIQUE)

Ajouter `Memory.cacheLimit` et `clearCache()` périodique dans la boucle de denoising.

### Phase 3: Text Encoders

Configurer les limites pour Dev/Mistral text encoder.

### Phase 4: CLI Flag

Option `--memory-profile` pour les utilisateurs avancés.

---

## Métriques Cibles

| Scénario | Avant | Après (cible) |
|----------|-------|---------------|
| Klein 4B 512x512 | ~12 GB | <10 GB |
| Klein 4B 1024x1024 | ~18 GB | <14 GB |
| Klein 9B 512x512 | ~18 GB | <14 GB |
| Dev 512x512 | ~45 GB | <35 GB |
