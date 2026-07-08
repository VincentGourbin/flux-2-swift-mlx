#!/usr/bin/env python3
"""Grille comparative : référence + 6 sorties, avec libellés."""
from PIL import Image, ImageDraw
import os

DIR = os.path.dirname(os.path.abspath(__file__))
ITEMS = [("mire.png", "référence"), ("mire_bf16.png", "bf16"),
         ("mire_qint8.png", "qint8"), ("mire_int4.png", "int4"),
         ("mire_mxfp8.png", "mxfp8"), ("mire_mxfp4.png", "mxfp4"),
         ("mire_nvfp4.png", "nvfp4")]
THUMB = 512
LABEL = 36
COLS = 4
rows = (len(ITEMS) + COLS - 1) // COLS
canvas = Image.new("RGB", (COLS * THUMB, rows * (THUMB + LABEL)), (24, 24, 24))
d = ImageDraw.Draw(canvas)
for i, (fname, label) in enumerate(ITEMS):
    path = os.path.join(DIR, fname)
    if not os.path.exists(path):
        continue
    im = Image.open(path).convert("RGB").resize((THUMB, THUMB), Image.LANCZOS)
    x, y = (i % COLS) * THUMB, (i // COLS) * (THUMB + LABEL)
    canvas.paste(im, (x, y))
    d.text((x + 10, y + THUMB + 8), label, fill=(240, 240, 240))
canvas.save(os.path.join(DIR, "montage.png"))
print("montage.png")
