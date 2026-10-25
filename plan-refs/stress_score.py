#!/usr/bin/env python3
"""Score OCR outputs on the stress set: word- and character-multiset F1 against
the ground truth, macro-averaged per variant.
Usage: stress_score.py <stress_dir> <name=out_dir>...  -> markdown table + JSON"""
import json
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path


def norm(s):
    return unicodedata.normalize("NFC", s)


def f1(pred: Counter, gt: Counter) -> float:
    tp = sum((pred & gt).values())
    if not tp:
        return 0.0
    p, r = tp / sum(pred.values()), tp / sum(gt.values())
    return 2 * p * r / (p + r)


root = Path(sys.argv[1])
engines = dict(a.split("=", 1) for a in sys.argv[2:])
meta = json.loads((root / "variants.json").read_text())
scores = defaultdict(lambda: defaultdict(list))  # variant -> engine -> [(wf1, cf1)]
for key, info in meta.items():
    variant = key.split("__")[0]
    gt = norm((root / "gt" / f"{key}.txt").read_text())
    for name, d in engines.items():
        f = Path(d) / f"{key}.txt"
        pred = norm(f.read_text()) if f.exists() else ""
        w = f1(Counter(pred.split()), Counter(gt.split()))
        c = f1(Counter(pred.replace(" ", "").replace("\n", "")), Counter(gt.replace(" ", "").replace("\n", "")))
        scores[variant][name].append((w, c))

order = list(dict.fromkeys(k.split("__")[0] for k in meta))
names = list(engines)
print("| variant | " + " | ".join(f"{n} word / char" for n in names) + " |")
print("|---|" + "---:|" * len(names))
out = {}
for v in order:
    cells = []
    for n in names:
        xs = scores[v][n]
        w = sum(x[0] for x in xs) / len(xs)
        c = sum(x[1] for x in xs) / len(xs)
        out.setdefault(v, {})[n] = {"word_f1": round(w, 4), "char_f1": round(c, 4), "pages": len(xs)}
        cells.append(f"{w:.3f} / {c:.3f}")
    print(f"| {v} | " + " | ".join(cells) + " |")
(root / "scores.json").write_text(json.dumps(out, indent=1))
