#!/usr/bin/env python3
"""Génère une mire de test couleur 1024x1024 pour le test de dérive de quantisation.

Grille 6x6 de patchs de 160px (avec marges), sur fond gris neutre 128 :
- Rangée 0 : primaires/secondaires saturées (R, G, B, C, M, Y)
- Rangée 1 : mêmes teintes à 50% de saturation
- Rangée 2 : tons chair (clair -> foncé)
- Rangée 3 : rampe de gris (0 -> 255)
- Rangée 4 : couleurs mémoire (ciel, feuillage, terre, orange, violet, rose)
- Rangée 5 : rampe de bleus (la dérive qint8 touchait surtout le bleu)
"""
from PIL import Image, ImageDraw
import json, sys, colorsys

SIZE = 1024
GRID = 6
CELL = SIZE // GRID          # 170
PATCH = 128                  # patch centré dans la cellule, marge confortable
BG = (128, 128, 128)

def hsv(h, s, v):
    r, g, b = colorsys.hsv_to_rgb(h, s, v)
    return (round(r * 255), round(g * 255), round(b * 255))

rows = [
    # saturées pures
    [hsv(h, 1.0, 1.0) for h in (0.0, 1/3, 2/3, 0.5, 5/6, 1/6)],
    # mi-saturées
    [hsv(h, 0.5, 0.9) for h in (0.0, 1/3, 2/3, 0.5, 5/6, 1/6)],
    # tons chair
    [(255, 224, 196), (240, 195, 160), (219, 168, 126), (180, 128, 90), (140, 95, 65), (96, 62, 40)],
    # gris
    [(0, 0, 0), (51, 51, 51), (102, 102, 102), (153, 153, 153), (204, 204, 204), (255, 255, 255)],
    # couleurs mémoire
    [(100, 160, 220), (70, 130, 60), (150, 110, 70), (235, 140, 50), (120, 70, 160), (230, 120, 150)],
    # rampe de bleus
    [(0, 0, 255), (30, 60, 220), (60, 100, 200), (40, 80, 160), (20, 40, 120), (10, 20, 80)],
]

img = Image.new("RGB", (SIZE, SIZE), BG)
d = ImageDraw.Draw(img)
patches = []
for r, row in enumerate(rows):
    for c, color in enumerate(row):
        x0 = c * CELL + (CELL - PATCH) // 2
        y0 = r * CELL + (CELL - PATCH) // 2
        d.rectangle([x0, y0, x0 + PATCH - 1, y0 + PATCH - 1], fill=color)
        patches.append({"row": r, "col": c, "rgb": list(color),
                        "box": [x0, y0, x0 + PATCH, y0 + PATCH]})

out = sys.argv[1] if len(sys.argv) > 1 else "mire.png"
img.save(out)
with open(out.replace(".png", "_patches.json"), "w") as f:
    json.dump(patches, f, indent=1)
print(f"mire: {out} ({len(patches)} patchs)")
