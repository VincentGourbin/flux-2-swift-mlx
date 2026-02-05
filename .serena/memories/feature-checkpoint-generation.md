# ✅ IMPLEMENTED: Checkpoint Generation Preview

## Description
Générer automatiquement une image de validation à chaque checkpoint pendant le training pour visualiser l'apprentissage du LoRA.

## Comportement souhaité
- À chaque sauvegarde de checkpoint (ex: tous les 500 steps), générer une image avec le LoRA actuel
- Utiliser un **prompt constant** configuré par l'utilisateur (ex: "blue_cat on a couch")
- Utiliser un **seed constant** pour la reproductibilité
- Résolution **512x512** (suffisant pour évaluer visuellement, plus rapide)
- Nommer les images par numéro de step: `preview_500.png`, `preview_1000.png`, etc.
- `preview_latest.png` pour le dernier checkpoint

## Exemple d'usage
```bash
flux2 train-lora /dataset \
  --output ./my-lora.safetensors \
  --save-every-n-steps 500 \
  --max-steps 1500 \
  --validation-prompt "blue_cat on a couch" \
  --validation-seed 42 \
  --validation-size 512
```

## Fichiers de sortie
```
output/
├── my-lora.safetensors          # Final LoRA
├── checkpoint_500.safetensors   # Checkpoint step 500
├── checkpoint_1000.safetensors  # Checkpoint step 1000
├── preview_500.png              # Image générée avec LoRA step 500
├── preview_1000.png             # Image générée avec LoRA step 1000
└── preview_latest.png           # Dernière image (= preview_1500.png)
```

## Implémentation
- Modifier `LoRATrainer.swift` pour appeler la génération à chaque checkpoint
- Réutiliser le pipeline d'inférence existant avec le LoRA courant
- Paramètres à ajouter dans `LoRATrainingConfig`:
  - `validationPrompt: String?` (déjà existant mais pas utilisé)
  - `validationSeed: UInt64 = 42`
  - `validationSize: Int = 512`
  - `generateAtCheckpoint: Bool = true`

## Notes
- Le `validationPrompt` existe déjà dans la config mais n'est pas encore implémenté
- Voir `LoRATrainer.swift` lignes 325-330 (TODO existant)

---

# TODO: Benchmark CPU sans verbose

## Contexte
Pendant le training 3000 steps avec `--verbose`, on a observé ~50% CPU.
L'analyse a montré que 96% des logs (91k lignes sur 94k) sont du verbose debug.

## Test à faire
Lancer un training court (100 steps) **sans** `--verbose` pour vérifier si la consommation CPU baisse significativement.

```bash
flux2 train-lora /private/tmp/lora-test-dataset \
  --output /tmp/lora-test-cpu.safetensors \
  --model klein-4b \
  --rank 16 \
  --learning-rate 1e-4 \
  --max-steps 100 \
  --batch-size 1 \
  --cache-latents \
  --cache-text-embeddings
  # PAS de --verbose
```

Monitorer avec `ps aux | grep flux2` et comparer le % CPU.
