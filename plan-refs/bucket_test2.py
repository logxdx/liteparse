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

def run(label, wmul, rows_pow2, algo="EXHAUSTIVE", cpu=False):
    shapes.clear()
    pipeline.pad_batch = make(wmul, rows_pow2)
    c = indra_ocr.load(threads=8, mode="ocr", providers=(["CPUExecutionProvider"] if cpu else [("CUDAExecutionProvider", {"cudnn_conv_algo_search": algo}), "CPUExecutionProvider"]))
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


base = run("GPU baseline", 8, False)
for label, wm, rp, al in [("GPU 256 pow2 HEURISTIC", 256, True, "HEURISTIC"), ("GPU 512 pow2", 512, True, "EXHAUSTIVE"), ("GPU 1024 pow2", 1024, True, "EXHAUSTIVE")]:
    t = run(label, wm, rp, al); print("   identical:", t == base)
cb = run("CPU baseline", 8, False, cpu=True)
for label, wm, rp in [("CPU width 128", 128, False), ("CPU 256 pow2", 256, True)]:
    t = run(label, wm, rp, cpu=True); print("   identical:", t == cb)
