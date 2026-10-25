# Plan: Indra OCR as an in-process liteparse engine

Branch `feat/indra-inprocess`, based on upstream `origin/main` (`a47f56d`, 2026-10-25).
All phases implemented (2026-10-25). The sections below the status are the original plan.

## Status (2026-10-25)

Open questions answered with the plan's defaults, except scope: all phases were done on request. Rust API only (no `lit` flag), private branch, the 20-page corpus plus a stress set.

- **Phase 0.** `ort =2.0.0-rc.12`, default features (same as `oar-ocr`): with `indra-ocr` and `oar-ocr` both on there is one `ort`/`ort-sys`. Signatures as in the table below (rec emits T = W/4 steps). Grayscale contract: `render_pages_for_ocr` gives tightly packed 1 byte/px luma; the engine prefers it and also accepts RGB.
- **Phase 1.** Detection, crops (dark lines, 3x window), width batches, greedy CTC, `MIN_CONF`.
- **Phase 3.** `orient.rs`: preview, skew search (same angle grid as Python, scored on all cores, equal to warp + row sums), `orient.onnx` (p > 0.95), quarter turns, deskew, and the `Frame` that maps points back. Turned/straightened pages return 4-point polygons in the reading frame (LiteParse snaps them to 0/90/180/270); vertical CJK pages return axis-aligned boxes.
- **Phase 4.** `script.onnx` routing (unsure lines read by base too, best confidence wins), `rec_cjk` / `rec_rtl` / `rec_world` on the line band, RTL to logical order (`rtl.rs`, port of Indra's `flip`), LaTeX/jamo decoding, n-gram LM beam search (`lm.rs`, `npz.rs`) for lines between `MIN_CONF` and the gate, line splitting at wide gaps and gutters, leader-rule removal and list-marker joining (`assemble.rs`).
- **Phase 2.** End to end through `LiteParse::with_ocr_engine` (`examples/ocr_parse.rs`, scanned PDFs from the stress set). README section and feature list. DPI: 150 stays the default; 200 helps only very small print (0.35x page: 0.922 -> 0.958 word F1) for ~10% more time. Word boxes (`IndraOptions::word_boxes`) cost word F1 in LiteParse's output (0.949 vs 0.974 line boxes on upright pages, same character F1: words are glued or split at box gaps), so line boxes are the default.
- **Phase 5.** `indra-ocr-cuda` (`Execution::Cuda`), provider-dependent batching (`Batching`; CUDA default: widths padded to 1024, power-of-two rows, min 8), `indra-ocr-auto-download` (`IndraOcrEngine::download(ModelSet::{Base, Full})`, pinned commit `bd02167` = tag `v1.1.0`, size + SHA-256 checked, OpenRAIL-M notice once).

### Parity with Python Indra 1.1.2 (full pipeline)

- Corpus (20 pages): 743/743 lines identical, word F1 1.0000 (`tests/indra_parity.rs`).
- Stress set (77 images): 3101/3103 boxes identical (99.94%); the two differences are on the 0.35x-resolution pages (resampling).

### Stress set, word / character F1 against ground truth

Raw engine output (`ocr_pages`), rust-v1 = Phase 1 only, rust = all phases:

| variant | rust-v1 | rust | py-indra | tesseract |
|---|---:|---:|---:|---:|
| upright | 0.976 / 0.996 | 0.975 / 0.996 | 0.975 / 0.996 | 0.972 / 0.988 |
| rot90 | 0.002 / 0.002 | 0.975 / 0.996 | 0.975 / 0.996 | 0.040 / 0.420 |
| rot180 | 0.029 / 0.506 | 0.975 / 0.996 | 0.975 / 0.996 | 0.040 / 0.425 |
| rot270 | 0.000 / 0.001 | 0.975 / 0.996 | 0.975 / 0.996 | 0.971 / 0.989 |
| tilt +3 | 0.192 / 0.357 | 0.974 / 0.995 | 0.974 / 0.995 | 0.942 / 0.967 |
| tilt -10 | 0.005 / 0.009 | 0.975 / 0.995 | 0.975 / 0.995 | 0.078 / 0.253 |
| tilt +25 | 0.002 / 0.004 | 0.978 / 0.995 | 0.978 / 0.995 | 0.001 / 0.000 |
| half resolution | 0.969 / 0.994 | 0.969 / 0.994 | 0.969 / 0.994 | 0.922 / 0.956 |
| 0.35 resolution | 0.962 / 0.993 | 0.962 / 0.993 | 0.962 / 0.993 | 0.694 / 0.877 |
| inverted | 0.965 / 0.994 | 0.965 / 0.994 | 0.965 / 0.994 | 0.972 / 0.988 |
| blur + noise | 0.965 / 0.993 | 0.965 / 0.993 | 0.965 / 0.993 | 0.963 / 0.986 |
| photo | 0.462 / 0.834 | 0.971 / 0.995 | 0.971 / 0.995 | 0.957 / 0.971 |
| Cyrillic, Greek, Devanagari | 1.000 | 1.000 | 1.000 | 1.000 / 0.899 / 1.000 |
| Chinese | 0.000 / 0.000 | 1.000 / 1.000 | 1.000 / 1.000 | 0.000 / 1.000 (no spaces) |
| Arabic | 0.000 / 0.008 | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 / 1.000 |

Through LiteParse at 150 dpi (scanned PDFs, Tesseract with `eng` as LiteParse defaults): Indra line boxes 0.971-0.974 word F1 on every rotated and tilted variant, Tesseract 0.000-0.103 on rot90/rot180/tilt -10/tilt +25; on 0.35x resolution Indra 0.922, Tesseract 0.040.

### Speed and memory (i5-13420H, 12 threads; RTX 4050)

- Raw engine, CPU: 0.24-0.28 s/page (orient 55 ms, det 30 ms, rec ~140 ms incl. crops 20 ms, script 5 ms, models ~100 ms). Python Indra: 1.11 s/page. Tesseract: 1.6-2.0 s/page.
- Through LiteParse: 0.27 s/page (150 dpi), 0.30 (200 dpi); Tesseract 1.61 / 1.76.
- CUDA: detection 30 -> 10 ms; recognizer models ~105-113 ms with the 1024 / pow2 buckets (242 ms without bucketing), i.e. no faster than this CPU: the 5 MB recognizer is launch/copy bound. Same text as CPU (742/743 lines). Worth it only to free CPU cores.
- Memory: engine load 202 MiB (full; LM tables ~160 MiB), 41 MiB without LMs and script groups. A 40-page scan through LiteParse peaks at ~1.2 GiB (Tesseract ~0.7 GiB); 120 pages in one process peak at 1.25 GiB, so it plateaus (scales with LiteParse's OCR workers, not with pages). The CPU arena is off (`DisableCpuMemArena`, as in Python).

### Not covered / left open

- Not measured on real scans, photos, handwriting, forms or mixed-script pages (the stress set is synthetic from one born-digital document).
- CLI: `lit parse` / `batch-parse` take `--ocr-engine indra` and `--indra-model-dir` (build with `--features indra-ocr` or `indra-ocr-auto-download`). No Node/Python/WASM exposure.
- Math (`--math` models), picture regions and Indra's reading order/layout are not ported (LiteParse has its own layout).
- `tests/indra_parity.rs` thresholds are loose (95% lines, F1 0.998); the measured parity is exact on the corpus.

## Goal

Run [Indra OCR](https://github.com/datalab-to/indra) (1M-parameter CPU-first ONNX OCR, Apache-2.0 code, modified OpenRAIL-M weights) **inside the liteparse process**, as an `OcrEngine` next to `oar.rs`, instead of the HTTP server that upstream already ships (`ocr/indraocr`, PR #508, port 8831).

Why in-process: no second process or port to manage, no per-page HTTP and PNG/multipart round trip, one model load per process, and a single binary story for the Rust API.

Non-goals (at least for v1): layout blocks, reading order (`order.onnx`), math (`--math` models), PDF text-layer reading (liteparse has its own), picture detection, WASM, Node/Python wheels.

## What Indra actually is (read from `indra-ocr` 1.1.2, ~5k lines of Python)

Python reference: `pip install indra-ocr==1.1.2` (numpy 2.5, opencv-python-headless 5.0, onnxruntime 1.31). Models: HF `datalab-to/indra-ocr`, revision `v1.1.0` (`HF_REVISION` in `models.py`); snapshot I tested was `bd02167fdf2005a19962cc15188f8496a2c87d3b`.

Pipeline per page (`pipeline.py::OrtOCR.__call__ / _read / recognize`):

1. **Orientation**: `preview` thumbnail, `estimate_skew` (row-sharpness search, +-45 degrees, step 0.1), `orient.onnx` (5 classes, only applied above p=0.95), page turned upright and deskewed. Boxes are mapped back afterwards.
2. **Detect** (`det.onnx`, 0.4 MB): resize to long side 1024 (`meta.det_long_side`), pad to multiple of 32, ink map `1 - gray/255`, run, `det_post.decode` (sigmoid, threshold 0.5, 4-connected components with stats, per-component mean top/bottom distance, x grown by `0.15*line height`). If median line height < 9 px and >= 8 lines, a second pass at a larger long side (max 2048).
3. **Crop** (`rec_preproc.crop_line`): expand box by 0.3 h horizontally and 0.2 h vertically, white (or black if `is_dark_line`) fill outside the page, resize to height 32 (INTER_AREA when shrinking, else INTER_LINEAR), ink map. **`rec.onnx` reads a 3x tall window** (`meta.rec_window = 3`, input `[B,1,96,W]`: the line plus the page above and below); group recognizers read the 32 px line band only.
4. **Recognize** (`rec.onnx`, 5 MB): batches grouped by width (`width_batches`: max 64 lines, ratio 1.3, `max_area = 64*640` to bound ORT memory), padded to multiple of 8, inputs `x` and `widths`, **greedy CTC runs inside the graph** (outputs `ids`, `conf`, `lengths`, plus `blank`/`topv`/`topi` for the LM). Confidence = mean CTC peak probability. A line under `MIN_CONF = 0.5` gets no text. `vocab.decode` is a plain character join (1,697 chars; blank = 0; `meta.vocab` is a JSON string inside JSON).
5. **Optional extras**: script routing (`script.onnx` -> `rec_cjk` / `rec_rtl` / `rec_world`, read twice when unsure), n-gram LM beam search (`lm*.npz`, 12-19 MB each, only for lines under confidence 0.9), RTL flip, line splitting at wide CTC gaps (`split_line`), reading order and layout (`order.onnx`).

ONNX signatures (verified with onnxruntime): det `x [1,1,H,W] -> out [1,3,h,w]` (3 channels: core logit, top distance, bottom distance; no picture channel in this model); rec `x [B,1,96,W], widths [B] -> ids, conf, lengths, blank, topv [B,T,8], topi [B,T,8]`; orient `x [B,1,768,768] -> p [B,5]`.

## What liteparse needs

`OcrEngine::recognize(image, w, h, &OcrOptions) -> Vec<OcrResult { text, bbox [x1,y1,x2,y2], confidence, polygon: Option<[[f32;2];4]> }>`, called per rendered page on a blocking worker (see `ocr_merge.rs::ocr_and_merge_rendered`). liteparse then drops OCR boxes that overlap native text and drops results with confidence <= 0.1. So Indra only has to produce **text boxes with text and confidence**; layout and ordering stay liteparse's job (the existing `oar.rs` returns line-level regions the same way).

`prefers_grayscale()` exists, so Indra can ask for a single-channel buffer. Verify the exact buffer layout in `render_pages_for_ocr` before relying on it. Default render DPI is 150 (`DEFAULT_DPI`); Indra's own CLI renders at 200 (see experiments).

`oar.rs` is the template: feature-gated module, `Mutex` around the runtime, `OcrOptions::language` ignored with a one-time warning, an `rgb_image` validator, `from_models`/`from_builder`/`from_runtime` constructors, auto-download behind a second feature, README section. It is **Rust-API only**: not wired into `lit`, napi, python or wasm.

## Design

New module `crates/liteparse/src/ocr/indra/` behind feature `indra-ocr` (non-wasm), mirroring `oar-ocr` / `oar-ocr-cuda` / `oar-ocr-auto-download`:

```
ocr/indra/mod.rs       IndraOcrEngine, OcrEngine impl, constructors, feature docs
ocr/indra/models.rs    model dir layout, meta.json (serde), optional download + cache
ocr/indra/imgproc.rs   resize (area/linear), connected components, percentile, rotate
ocr/indra/detect.rs    page_to_input, decode, adaptive second pass
ocr/indra/crop.rs      is_dark_line, crop_line, width_batches, pad_batch
ocr/indra/recognize.rs session run + id/conf decode, vocab, MIN_CONF
ocr/indra/orient.rs    (phase 3) skew estimate, orient.onnx, rotate/unrotate
```

Decisions:

- **Runtime**: depend on `ort` directly (`oar-ocr` already pulls `ort 2.0.0-rc.12`; use the same version and feature set so a build with both features links one ONNX Runtime). Phase 0 must confirm how `oar-ocr` configures `ort` (download-binaries vs load-dynamic) and pick the same.
- **Numerics**: no OpenCV. The image ops are small, and exact `cv2` semantics matter for accuracy, so port them: `INTER_AREA` (fractional box average) and `INTER_LINEAR` (half-pixel centers) on `u8`/`f32` gray, 4-connected component labeling with area/left/width stats, histogram percentiles (the Python is already histogram-based), and later `warpAffine` bilinear for deskew. `imageproc` and `ndarray` are already in the lockfile but `image::imageops::resize` filters are not `INTER_AREA`, so don't use them for the model inputs.
- **Concurrency and memory**: `ort::Session::run` needs exclusive access in rc.12, so one `Mutex<Session>` per model first (as `oar.rs` does), configurable intra-op threads, **disable the CPU memory arena** and keep `max_area` batching (Python notes ORT held 5 GB on a 40-page scan otherwise). Revisit a session pool after measuring.
- **Execution providers**: CPU default. `indra-ocr-cuda` (and later tensorrt/coreml) features forward to `ort`. CUDA build here is CUDA 13 and needs **cuDNN 9** on the loader path (for example the `nvidia/cudnn/lib` folder of a pip `nvidia-cudnn-cu13` install, via `LD_LIBRARY_PATH`).
- **Batching is provider-dependent** (measured, see below): on CUDA, round pad width up to a large multiple (512-1024) and batch rows up to a power of two; on CPU keep Indra's own `width_batches` and multiple of 8, because extra padding costs real compute there.
- **Models and licensing**: never embed or redistribute the weights. Load from a model dir (`IndraOcrEngine::from_dir`, env `INDRA_OCR_MODEL_DIR`), minimum set `det.onnx`, `rec.onnx`, `meta.json` (about 5.6 MB). Optional `indra-ocr-auto-download` downloads revision `v1.1.0` from Hugging Face into a cache dir with checksum verification, and logs the OpenRAIL-M terms once (free for research, personal use, startups under $20M funding/revenue; not usable competitively with Datalab's API; broader use needs a Datalab license).
- **Language**: `OcrOptions::language` ignored with a one-time warning (one model reads all supported scripts), like `oar.rs` and the upstream Indra server.

## Phases (each ends with something runnable and a measurement)

### Phase 0: spike (about 1 day)
- Add `ort` dependency under the new feature; confirm `cargo build --features indra-ocr` works with and without `oar-ocr` enabled, and note link/runtime-download behaviour in CI.
- Load `det.onnx` and `rec.onnx` from a model dir, print the I/O signatures, run one dummy tensor each.
- Decide the grayscale buffer contract from `render_pages_for_ocr`.
- Exit: the signatures match the table above; build story documented.

### Phase 1: single-script MVP, parity with Python on upright pages (the core of the work)
- `imgproc`, `detect` (incl. adaptive second pass), `crop` (dark-line inversion, 3x tall window, 0.3/0.2 padding), `recognize` (width batches, `widths` input, ids/conf/lengths, `MIN_CONF`), vocab/meta parsing, `OcrEngine` impl returning line-level `OcrResult` with `polygon = None`.
- Unit tests for each numeric helper against fixtures dumped from the Python reference (small arrays, committed).
- Parity test (gated on `INDRA_OCR_MODEL_DIR`, skipped otherwise): on the 20-page ocr-bench corpus, match Rust and Python line boxes (IoU >= 0.9) and compare text; targets: word-multiset F1 within 0.002 of Python (0.9941 on that corpus) and identical text on at least 95% of lines. Expect small differences from float resampling; investigate anything larger than that.
- Exit: `cargo test --features indra-ocr` green; parity numbers recorded here.

### Phase 2: integration quality in liteparse
- End-to-end on scanned PDFs through `Liteparse::with_ocr_engine`, compared with the Tesseract and HTTP-Indra paths on the same pages (F1 against ground truth, time per page, RSS).
- Decide DPI (150 default vs 200) and whether to render grayscale; set a recommended OCR DPI for the engine.
- Word-level boxes: split lines at space tokens or wide CTC gaps (`split_line`, gaps 3.0 / 1.8 / 1.6 line heights; needs the per-token x positions `ix0 + (t+0.5)*stride*f`), so liteparse's overlap filter and projection get word-sized boxes like Tesseract's. Keep line-level as a switch and measure which gives better liteparse output.
- README section modeled on the `oar-ocr` one; `lit` CLI flag (`--indra-model-dir`) if upstream wants it, otherwise Rust API only like oar.
- Exit: documented numbers; clean `cargo fmt` and `clippy`.

### Phase 3: rotated and tilted scans
- Port `estimate_skew`, `orient.onnx` (thumbnail 768x768, threshold 0.95), page rotate/deskew, and map boxes back as 4-point polygons (liteparse's `OcrResult.polygon` already supports rotated text).
- Exit: a rotated/tilted test set (rotate the corpus 90/180/270 and +-10 degrees) reads within a small margin of upright.

### Phase 4: other scripts and LM (only if wanted)
- `script.onnx` routing plus `rec_cjk` / `rec_rtl` / `rec_world`, RTL reordering (e.g. `unicode-bidi`), and the n-gram LM beam search (`.npz` reader, `lm.py` port). Each is an extra model file (5-19 MB) and Python measured these mostly for handwriting; skip unless a language need shows up.

### Phase 5: GPU
- `indra-ocr-cuda` feature, provider-dependent batching as above. Re-measure on the corpus; budget about 1.1 GiB VRAM for the CUDA path.

## Measured baselines to beat or match (this machine: i5-13420H, RTX 4050 6 GB, 20 dense pages of `spider-cloud-blog-archive.pdf` @ 200 DPI, scored against `pdftotext`)

| engine | pages/s | F1 |
|---|---:|---:|
| Python Indra, CPU, 8 threads | 0.84 (about 1.2 s/page, RSS 671 MB) | 0.9941 |
| Python Indra, ORT CUDA | 1.9 (0.52 s/page) | 0.9941 |
| Python Indra, CUDA + width 1024 bucket + pow2 rows | about 4.8 (0.21 s/page) | identical text |
| faster-paddle-tiny CPU / TurboOCR CPU | 1.99 / 1.59 | 0.9894 / 0.9872 |

Other measured facts that shaped the design:
- CPU time per page split (p045): orient 105 ms, det 85 ms, **rec 955 ms**, order 31 ms. Recognition dominates, so batching and thread use in `recognize.rs` matter most.
- On GPU the run is host-bound (GPU SM utilisation 10-40%); detection drops 85 -> 10 ms, recognition 955 -> 365 ms.
- ORT CUDA with cuDNN algorithm-search options changed nothing; **shape bucketing** is what helped. TurboOCR does the same on its CPU/ORT path (`REC_BUCKET_STEP`, width snap, groups per bucket) and uses fixed buckets plus CUDA graphs on its TensorRT path. On CPU, coarser padding made Indra slower.
- This snapshot ships no `rec.int8.onnx` and no `mask.onnx`; the int8 and CenterMask code paths are not needed.
- The corpus is one born-digital document rendered to images. It says nothing about handwriting, rotation or non-Latin text, which are Indra's selling points. Add a scanned/rotated/multilingual set before judging those.

Reference scripts (Python, not part of the build): `plan-refs/indra_run.py` (ocr-bench adapter for Indra; `--providers` selects CPU/CUDA), `plan-refs/bucket_test.py` and `bucket_test2.py` (the bucketing experiments). They expect an `indra-ocr` venv and an ocr-bench checkout (its path in `OCR_BENCH`).

## Risks

- **Exactness of image ops**: wrong `INTER_AREA`/half-pixel handling shifts crops by a fraction of a pixel and costs accuracy quietly. Mitigation: fixture tests per helper before the end-to-end parity test.
- **`ort` rc.12 API and linking**: pre-release crate, and the way ONNX Runtime is fetched or linked affects every CI target. Resolved in Phase 0.
- **Concurrency**: liteparse calls the engine from several blocking workers; a single `Mutex<Session>` serializes them. Fine for CPU (intra-op threads), possibly a bottleneck on GPU.
- **Memory on big scans**: the ORT arena issue above; keep the arena off and the `max_area` cap, and test a 40-page scan.
- **Licensing**: weights are not Apache-2.0. Keep download opt-in and don't bundle weights in release artifacts.
- **Upstream fit**: upstream chose a server for Indra (#508) and `oar-ocr` for in-process. An in-process port is a bigger surface (about 1,500-2,000 lines Rust for Phases 1-3); check appetite before sending it upstream.

## Open questions (need an answer before Phase 0)

1. **Scope**: is Phase 1 + 2 (upright pages, line or word boxes) enough for v1, or are rotated scans (Phase 3) required from the start? Do you need non-Latin scripts (Phase 4)?
2. **Destination**: private fork/branch for your own use, or something to propose upstream? (Your notes say nothing is pushed and no fork exists yet.)
3. **GPU**: do you want the CUDA path at all, given the 6 GB card is shared with the agent models? CPU-only keeps Phases 0-3 simple.
4. **Surface**: Rust API only (like `oar.rs`), or also a `lit` CLI flag?
5. **Test data**: is the 20-page ocr-bench corpus enough for parity, or can you point me at scanned/rotated/handwritten pages to add?
