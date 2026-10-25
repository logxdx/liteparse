#!/usr/bin/env python3
"""Indra OCR adapter for ocr-bench: ONNX Runtime CPU, image in -> text out.

Run with the indra venv; bench_common is imported from the ocr-bench checkout.
"""
import argparse
import os
import sys
import time
from pathlib import Path

BENCH = Path(os.environ.get("OCR_BENCH", "ocr-bench"))
sys.path.insert(0, str(BENCH))
from bench_common import Recorder, pages  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--dpi", type=float, default=200.0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--suffix", default="")
    ap.add_argument("--providers", default="CPUExecutionProvider")
    a = ap.parse_args()

    import indra_ocr

    rec = Recorder(f"indra-cpu{a.suffix}", "cpu", f"Indra OCR ({a.providers}), threads={a.threads}, mode=ocr")
    try:
        t = time.perf_counter()
        conv = indra_ocr.load(threads=a.threads, mode="ocr", providers=a.providers.split(","))
        rec.load_done((time.perf_counter() - t) * 1000.0)

        ps = pages()
        if a.limit:
            ps = ps[: a.limit]
        for m in ps:
            t = time.perf_counter()
            out = list(conv.convert(m["image"]))
            ms = (time.perf_counter() - t) * 1000.0
            text = "\n".join(p.text for _, p in out)
            rec.page(m["name"], ms, text, {"lines": sum(len(p.lines) for _, p in out)})
    except Exception as e:  # noqa: BLE001
        rec.finish(error=f"{type(e).__name__}: {e}")
        raise
    rec.finish()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
