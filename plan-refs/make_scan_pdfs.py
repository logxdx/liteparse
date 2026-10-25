#!/usr/bin/env python3
"""Image-only PDFs from stress-set images (one page each, 200 DPI), plus one
40-page scan for the memory check. Usage: make_scan_pdfs.py <stress_dir>"""
import sys
from pathlib import Path

from PIL import Image

root = Path(sys.argv[1])
out = root / "pdf"
out.mkdir(exist_ok=True)
imgs = sorted((root / "img").glob("*.png"))
for p in imgs:
    Image.open(p).convert("L").save(out / f"{p.stem}.pdf", resolution=200.0)
big = [Image.open(p).convert("L") for p in imgs if p.stem.startswith(("upright__", "noisy__", "photo__", "rot90__", "tilt3__", "half_res__", "inverted__"))][:40]
big[0].save(root / "scan40.pdf", save_all=True, append_images=big[1:], resolution=200.0)
print(len(imgs), "pdfs;", len(big), "pages in scan40.pdf")
