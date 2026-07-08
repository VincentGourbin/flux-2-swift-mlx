#!/usr/bin/env python3
"""Compare les sorties i2i à la mire de référence, par patch.

Usage: analyze.py mire.png mire_patches.json out_bf16.png out_qint8.png ...
Pour chaque patch : moyenne RGB sur les 60% centraux (évite les bords adoucis
par le VAE), ΔRGB, ΔSaturation (HSV), ΔLuminosité. Sortie JSON + tableau texte.
"""
import json, sys
import numpy as np
from PIL import Image

ROW_NAMES = ["saturées", "mi-saturées", "chair", "gris", "mémoire", "bleus"]

def patch_mean(arr, box):
    x0, y0, x1, y1 = box
    # 60% centraux
    w, h = x1 - x0, y1 - y0
    mx, my = int(w * 0.2), int(h * 0.2)
    region = arr[y0 + my : y1 - my, x0 + mx : x1 - mx].reshape(-1, 3).astype(np.float64)
    return region.mean(axis=0)

def sat_val(rgb):
    r, g, b = rgb / 255.0
    mx, mn = max(r, g, b), min(r, g, b)
    s = 0.0 if mx == 0 else (mx - mn) / mx
    return s * 100, mx * 100  # saturation %, value %

def main():
    ref_path, patches_path, *outs = sys.argv[1:]
    ref = np.asarray(Image.open(ref_path).convert("RGB"))
    patches = json.load(open(patches_path))

    results = {}
    for out_path in outs:
        name = out_path.split("/")[-1].replace("mire_", "").replace(".png", "")
        img = np.asarray(Image.open(out_path).convert("RGB"))
        if img.shape != ref.shape:
            print(f"!! {name}: shape {img.shape} != ref {ref.shape}", file=sys.stderr)
            continue
        per_patch = []
        for p in patches:
            m_ref = patch_mean(ref, p["box"])
            m_out = patch_mean(img, p["box"])
            s_ref, v_ref = sat_val(m_ref)
            s_out, v_out = sat_val(m_out)
            per_patch.append({
                "row": p["row"], "col": p["col"], "target": p["rgb"],
                "ref": m_ref.round(1).tolist(), "out": m_out.round(1).tolist(),
                "d_rgb": (m_out - m_ref).round(1).tolist(),
                "d_sat": round(s_out - s_ref, 1), "d_val": round(v_out - v_ref, 1),
                "d_e_simple": round(float(np.linalg.norm(m_out - m_ref)), 1),
            })
        # agrégats par rangée
        rows = {}
        for r in range(6):
            pp = [q for q in per_patch if q["row"] == r]
            rows[ROW_NAMES[r]] = {
                "d_sat_mean": round(float(np.mean([q["d_sat"] for q in pp])), 1),
                "d_val_mean": round(float(np.mean([q["d_val"] for q in pp])), 1),
                "d_e_mean": round(float(np.mean([q["d_e_simple"] for q in pp])), 1),
                "d_e_max": round(float(np.max([q["d_e_simple"] for q in pp])), 1),
            }
        glob = {
            "d_sat_mean": round(float(np.mean([q["d_sat"] for q in per_patch])), 1),
            "d_e_mean": round(float(np.mean([q["d_e_simple"] for q in per_patch])), 1),
            "d_e_max": round(float(np.max([q["d_e_simple"] for q in per_patch])), 1),
        }
        results[name] = {"global": glob, "rows": rows, "patches": per_patch}

    json.dump(results, open("analysis.json", "w"), indent=1)

    # tableau récapitulatif
    names = list(results)
    print(f"{'':14}" + "".join(f"{n:>10}" for n in names))
    print("ΔSat moyen (points de %)")
    for r in ROW_NAMES:
        print(f"  {r:12}" + "".join(f"{results[n]['rows'][r]['d_sat_mean']:>10}" for n in names))
    print(f"  {'GLOBAL':12}" + "".join(f"{results[n]['global']['d_sat_mean']:>10}" for n in names))
    print("ΔE moyen (norme RGB)")
    for r in ROW_NAMES:
        print(f"  {r:12}" + "".join(f"{results[n]['rows'][r]['d_e_mean']:>10}" for n in names))
    print(f"  {'GLOBAL':12}" + "".join(f"{results[n]['global']['d_e_mean']:>10}" for n in names))
    print("ΔE max")
    print(f"  {'GLOBAL':12}" + "".join(f"{results[n]['global']['d_e_max']:>10}" for n in names))

if __name__ == "__main__":
    main()
