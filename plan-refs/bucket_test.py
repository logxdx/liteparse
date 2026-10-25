import os, time, math, numpy as np, sys
import indra_ocr
from indra_ocr import pipeline, rec_preproc
from pathlib import Path
P = Path(os.environ.get("OCR_BENCH", "ocr-bench")) / "corpus" / "img"
names = ["p011","p021","p033","p045","p054","p061","p070","p078","p090","p100","p108","p118"]
print("WIDTH_MULTIPLE", rec_preproc.WIDTH_MULTIPLE)
orig = rec_preproc.pad_batch
shapes = set()

def make(wmul, rows_pow2):
    def pb(lines, multiple=None):
        if rows_pow2:
            n = len(lines); r = 1 << (n - 1).bit_length(); r = max(r, 8)
            lines = list(lines) + [lines[-1]] * (r - n)
        arr, w = orig(lines, wmul)
        shapes.add(arr.shape)
        return arr, w
    return pb

def run(label, wmul, rows_pow2, algo="EXHAUSTIVE"):
    shapes.clear()
    pipeline.pad_batch = make(wmul, rows_pow2)
    c = indra_ocr.load(threads=8, mode="ocr", providers=[("CUDAExecutionProvider", {"cudnn_conv_algo_search": algo}), "CPUExecutionProvider"])
    list(c.convert(str(P/"p002.png")))
    res = []
    for p in range(2):
        t = time.perf_counter()
        txt = ""
        for n in names:
            txt += "\n".join(pg.text for _, pg in c.convert(str(P/f"{n}.png")))
        res.append((time.perf_counter() - t) / len(names) * 1000)
    print(f"{label:28s} pass1 {res[0]:6.0f} ms/page  pass2 {res[1]:6.0f} ms/page  distinct shapes {len(shapes)}", flush=True)
    return txt

base = run("baseline (mult default)", rec_preproc.WIDTH_MULTIPLE, False)
t2 = run("width mult 128", 128, False)
t3 = run("width mult 128 + rows pow2", 128, True)
t4 = run("width mult 256 + rows pow2", 256, True)
print("text identical to baseline:", t2 == base, t3 == base, t4 == base)
