# LoRA Training Hyperparameters Research

## Objectif
Trouver des patterns de paramètres qui fonctionnent bien pour simplifier l'expérience utilisateur.
L'utilisateur ne devrait pas avoir à saisir 20 paramètres - on doit avoir des presets intelligents.

## Expériences

### Expérience 1 : Training initial (session précédente)
- **Dataset** : blue_cat (15 images)
- **Paramètres** : LR 1e-4, 3000 steps, rank 16
- **Résultat** : Loss converge (5.28 → 3.96 en quelques steps)
- **Problème** : 3000 steps probablement trop long, risque d'overfitting

### Expérience 2 : Terminée
- **Dataset** : blue_cat (6 images dans /private/tmp/lora-test-dataset)
- **Paramètres** : LR 5e-5, 600 steps, rank 16
- **Durée** : ~36 minutes (3.6s/step)
- **Checkpoints** : 200, 400, 600 + final
- **Résultat** : Training OK, checkpoints créés
- **Bug identifié** : Pas de preview images générées (VAE libéré après caching)

### Expérience 3 : Terminée - Test inférence
- **checkpoint_200** : ✅ Bon résultat, image cohérente
- **checkpoint_600** : ❌ Overfitté, pattern répétitif bizarre
- **Sans LoRA** : ✅ Image de référence parfaite

### Conclusions importantes
1. **Overfitting rapide** : 600 steps pour 6 images = catastrophe
2. **Règle steps/image** : ~30-40 steps max par image
   - 6 images → 200 steps max
   - 15 images → 500 steps max
   - 30 images → 1000 steps max
3. **Early stopping** : Devrait détecter l'overfitting avant qu'il ne soit trop tard
4. **Validation images** : CRITIQUES pour voir l'évolution et stopper à temps

### Presets recommandés (à valider)
| Taille dataset | Steps max | LR | Notes |
|----------------|-----------|-----|-------|
| 5-10 images | 200-300 | 5e-5 | Très peu de données |
| 10-20 images | 400-600 | 5e-5 | Dataset standard |
| 20-50 images | 800-1500 | 1e-4 | Dataset confortable |
| 50+ images | 2000+ | 1e-4 | Peut aller plus loin |

## Patterns à investiguer

### Learning Rate
- 1e-4 : Standard pour LoRA, peut être trop agressif
- 5e-5 : Plus conservateur, moins de risque d'overfitting
- À tester : adaptive LR basé sur taille dataset ?

### Nombre de steps
- Règle empirique possible : `steps = images * 40-80` ?
- 15 images → 600-1200 steps semble raisonnable
- À valider avec les previews de validation

### Early Stopping
- **Problème actuel** : patience en epochs, pas adapté aux courts trainings
- **Idée** : patience en % du total de steps plutôt qu'en epochs
  - Ex: patience = 20% du training, min_delta = 0.01
  - Pour 600 steps : stop si pas d'amélioration pendant 120 steps
- **Alternative** : Pas d'early stopping pour trainings courts, juste des checkpoints
  - L'utilisateur choisit le meilleur checkpoint via les previews

### Presets intelligents par type de sujet
| Type | LR | Steps/image | Rank | Notes |
|------|-----|-------------|------|-------|
| Person/Face | 5e-5 | 50-80 | 16-32 | Plus de détails, plus de steps |
| Style | 1e-4 | 30-50 | 8-16 | Style global, moins de steps |
| Object | 5e-5 | 40-60 | 16 | Équilibré |
| Character | 5e-5 | 60-100 | 32 | Détails + style |

## TODO
- [ ] Analyser les previews de l'expérience 2
- [ ] Tester avec différentes tailles de dataset
- [ ] Mesurer la qualité finale (génération avec prompt varié)
- [ ] Définir les presets par défaut pour train-lora-easy

## Expérience 4 : Tests de scale à l'inférence (checkpoint_600 overfitté)

| Scale | Résultat | Notes |
|-------|----------|-------|
| 0.3 | ❌ Trop faible | Effet quasi-invisible, résultat cartoon comme sans LoRA |
| 0.5 | ✅ Bon compromis | Compense l'overfitting, chat bleu réaliste |
| 0.7 | ⚠️ Acceptable | Artefacts commencent à apparaître |
| 1.0 | ❌ Trop fort | Overfitting visible, patterns répétitifs |

### Insight clé
On peut "sauver" un LoRA overfitté en réduisant le scale à l'inférence (0.3-0.5).
Cela suggère que :
1. Le scale est un paramètre important à exposer à l'utilisateur
2. Un bon preset pourrait inclure un scale recommandé basé sur les steps/image
3. Pour les trainings longs (potentiellement overfittés), suggérer scale 0.5-0.7

## Recherche externe (Reddit, Civitai, FluxGym)

### Paramètres recommandés pour Flux.2 (sources multiples)

| Paramètre | Style LoRA | Character LoRA | Notes |
|-----------|------------|----------------|-------|
| Learning Rate | 1.5e-4 | 2.5e-5 | Style = plus agressif, Character = plus conservateur |
| Steps | 3000 | 8000 | Character nécessite plus de détails |
| Resolution | 512px ou 1024px | 1024px | 512px réduit le temps à ~2.5h |
| Dataset size | 20-30 images | 20-30 images | Min 10-15 si consistant |
| Checkpoint interval | 250 steps | 250 steps | Samples toutes les 500 steps |

### Formule Reddit (discussion Flux.2 Klein)
- 50 images → 5000 steps (100 epochs, 100 steps/image)
- Checkpoint toutes les 10 epochs (~500 steps)
- LR plus élevé que le défaut donne de meilleurs résultats
- Training sur toutes les résolutions améliore la généralisation

### Validation loss (nouveauté implémentée)
- Split train/val recommandé : 80%/20%
- Validation loss permet de détecter l'overfitting objectivement
- Gap train_loss vs val_loss : si val > train + 0.1, signe d'overfitting
- Implémenté dans LoRATrainer avec `--validation-dataset` option
