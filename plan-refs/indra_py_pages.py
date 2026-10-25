#!/usr/bin/env python3
"""Full Python Indra pipeline (orientation, deskew, script routing, LM, reading
order) over page images; one text file per page.
Usage: indra_py_pages.py <out_dir> <png>...  (run with the indra-ocr venv)"""
import sys
import time
from pathlib import Path

import indra_ocr

out = Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)
conv = indra_ocr.load(threads=8, mode="ocr")
total = 0.0
for p in sys.argv[2:]:
    t = time.perf_counter()
    text = "\n".join(page.text for _, page in conv.convert(p))
    total += time.perf_counter() - t
    (out / f"{Path(p).stem}.txt").write_text(text + "\n")
print(f"{len(sys.argv) - 2} pages, {total / max(1, len(sys.argv) - 2):.2f}s/page", file=sys.stderr)
