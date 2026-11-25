# LiteParse

[![Crates.io version](https://img.shields.io/crates/v/liteparse.svg)](https://crates.io/crates/liteparse)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

Rust library and CLI for fast, lightweight PDF and document parsing with spatial text extraction. Runs entirely locally with zero cloud dependencies.

> LiteParse is also available for [Node.js/TypeScript](https://www.npmjs.com/package/@llamaindex/liteparse), [Python](https://pypi.org/project/liteparse/), and the [browser (WASM)](https://www.npmjs.com/package/@llamaindex/liteparse-wasm). See the [project README](https://github.com/run-llama/liteparse) for all options.

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
liteparse = "2"
```

Or install the CLI:

```bash
cargo install liteparse
```

## Quick Start

```rust
use liteparse::{LiteParse, LiteParseConfig};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let parser = LiteParse::new(LiteParseConfig::default());
    let result = parser.parse("document.pdf").await?;

    println!("{}", result.text);

    for page in &result.pages {
        println!("Page {}: {} text items", page.page_num, page.text_items.len());
    }

    Ok(())
}
```

## Configuration

```rust
use liteparse::{LiteParse, LiteParseConfig, OutputFormat};

let config = LiteParseConfig {
    ocr_enabled: true,                    // Enable OCR (default: true)
    ocr_language: "eng".to_string(),      // Tesseract language code
    ocr_server_url: None,                 // HTTP OCR server URL (optional)
    tessdata_path: None,                  // Path to tessdata directory (optional)
    max_pages: 1000,                      // Max pages to parse
    target_pages: Some("1-5,10".into()),  // Specific pages (optional)
    dpi: 150.0,                           // Rendering DPI
    output_format: OutputFormat::Json,    // Json | Text | Markdown
    extract_annotations: false,           // Include page annotations in output
    extract_structure_tree: false,        // Include tagged-PDF logical structure
    preserve_very_small_text: false,      // Keep tiny text
    extract_text_metadata: false,         // Opt in to rich PDF text metadata
    password: None,                       // Password for protected documents
    quiet: false,                         // Suppress progress output
    ..Default::default()
};

let parser = LiteParse::new(config);
```

Set `extract_annotations: true` to populate `ParsedPage::annotations` with
annotation subtype, contents, author/title, PDF date strings, viewport-space
rectangle and quadpoint rectangles, and external link URI. It is independent
of `extract_links`, which controls Markdown link rendering. The field is
`None` when extraction is disabled.

Set `extract_structure_tree: true` to populate `ParsedPage::structure_tree` with
the complete tagged-PDF hierarchy: all roots, element type/ID, actual and alternate
text, title, typed scalar attributes, MCIDs, recursive children, and referenced link
annotations. Disabled pages use `None`; enabled untagged pages have no roots.

## Markdown Output

LiteParse can render documents directly to Markdown, including headings, tables, lists,
images, and links reconstructed from the spatial layout. Set
`output_format: OutputFormat::Markdown`; the rendered Markdown is returned on
`result.text`. Two related knobs control Markdown rendering:

- `image_mode` (`ImageMode::Placeholder` default | `Off` | `Embed`) — how raster
  images are surfaced in the output.
- `extract_images` (default `false`) — return embedded image bytes and metadata
  without changing Markdown image handling. This is the only option that enables
  extraction.
- `image_output_dir` — write extracted image files and return their names/paths;
  requires `extract_images: true`. Duplicate image resources reuse the same file.
- `extract_links` (default `true`) — render hyperlink annotations as
  `[text](url)`; set `false` for plain anchor text.

```rust
use liteparse::config::{ImageMode, LiteParseConfig, OutputFormat};

let config = LiteParseConfig {
    output_format: OutputFormat::Markdown,
    image_mode: ImageMode::Placeholder,
    extract_images: true,
    image_output_dir: Some("./images".into()),
    extract_links: true,
    ..Default::default()
};
let result = LiteParse::new(config).parse("document.pdf").await?;
println!("{}", result.text); // rendered Markdown
```

> Reconstruction quality varies with document complexity.

## Parsing from Bytes

```rust
use liteparse::types::PdfInput;

let pdf_bytes: Vec<u8> = std::fs::read("document.pdf")?;
let result = parser.parse_input(PdfInput::Bytes(pdf_bytes)).await?;
println!("{}", result.text);
```

## Document Complexity

Before committing to a full parse, check whether a document needs OCR or heavier
processing. `is_complex` is a cheap, text-layer-only pass that returns a
`PageComplexityStats` per page with a `needs_ocr` verdict and the signals behind it —
useful for routing documents to different pipelines, rejecting ones you can't handle, or
estimating cost.

```rust
use liteparse::types::PdfInput;

let parser = LiteParse::new(LiteParseConfig::default());
let pages = parser.is_complex(PdfInput::Path("document.pdf".into())).await?;

if pages.iter().any(|p| p.needs_ocr) {
    // Route to the OCR-enabled pipeline, inspect `p.reasons`, etc.
    for page in pages.iter().filter(|p| p.needs_ocr) {
        println!("Page {} needs OCR: {:?}", page.page_number, page.reasons);
    }
}
```

`reasons` is a `Vec<ComplexityReason>` (`Scanned`, `NoText`, `SparseText`,
`EmbeddedImages`, `Garbled`, `VectorText`, `AnnotationText`); new variants may be added over time, so match
leniently.

## Custom OCR Engine

Implement the `OcrEngine` trait to plug in your own OCR backend:

```rust
use liteparse::ocr::OcrEngine;
use std::sync::Arc;

let parser = LiteParse::new(LiteParseConfig::default())
    .with_ocr_engine(Arc::new(my_engine));
```

### Indra OCR (in-process)

The `indra-ocr` feature runs [Indra OCR](https://github.com/datalab-to/indra)
inside the process, on ONNX Runtime, as an alternative to the Indra HTTP server
in `ocr/indraocr`. It ports Indra's whole page pipeline: skew search and
orientation (pages turned by a quarter turn or tilted up to 45 degrees are read
upright), line detection, script routing to the base, CJK, right-to-left and
world recognizers, a language-model re-read of unsure lines, and line splitting
at column gutters. Reading order and layout stay LiteParse's own. Results are
line boxes (or word boxes with `IndraOptions::word_boxes`); on a turned or
straightened page each carries a 4-point polygon, so LiteParse sees the text's
orientation.

In builds with `indra-ocr`, Indra is the default OCR backend
(`OcrBackend::Indra`, CLI `--ocr-backend indra`): it loads on the first page
that needs OCR. It runs on the ONNX Runtime that
`liteparse models pull ocria-nano` installs (or `ORT_DYLIB_PATH`); that package
also brings LayoutNet, which then reads tables, headings and page
headers/footers with the Indra (and ocria-nano) backend.

To build an engine yourself instead:

```rust,no_run
use liteparse::ocr::indra::IndraOcrEngine;
use liteparse::{LiteParse, LiteParseConfig};
use std::sync::Arc;

// A folder with det.onnx, rec.onnx and meta.json (plus, optionally,
// orient.onnx, script.onnx, rec_<group>.onnx and lm*.npz), e.g. a download
// of the Hugging Face repo datalab-to/indra-ocr at tag v1.1.0.
let engine = IndraOcrEngine::from_dir("models/indra-ocr")?;
let parser = LiteParse::new(LiteParseConfig {
    ocr_enabled: true,
    ..LiteParseConfig::default()
})
.with_ocr_engine(Arc::new(engine));
# Ok::<(), Box<dyn std::error::Error>>(())
```

`IndraOcrEngine::from_env` reads the folder from `INDRA_OCR_MODEL_DIR`;
`IndraOcrEngine::new` takes `IndraOptions` (orientation, script routing,
language model, word boxes, ONNX Runtime threads, execution provider). Optional
model files that are missing switch their step off.

With `indra-ocr-auto-download`, `IndraOcrEngine::download(ModelSet::Full, options)`
fetches the models from the pinned revision, verifies their SHA-256 digests and
caches them under `$INDRA_OCR_HOME` (default
`~/.cache/liteparse/indra-ocr/<commit>`, or
`%LOCALAPPDATA%\liteparse\indra-ocr\<commit>` on Windows). `ModelSet::Base` (about 6 MB) covers
Latin, Greek, Cyrillic, Devanagari and Bengali; `ModelSet::Full` (about 73 MB)
adds Chinese, Japanese, Korean, Arabic, Hebrew and other scripts, and the
language models.

The weights are not Apache-2.0: they are under a modified OpenRAIL-M license
(free for research, personal use and startups under $20M in funding or
revenue; not for use competing with Datalab's API). They are never bundled, and
a download prints the terms once. `OcrOptions::language` is not interpreted: the
script classifier picks the recognizer per line.

## Features

- **`tesseract`** (default) — Built-in Tesseract OCR via `tesseract-rs`. Disable with `default-features = false` if you don't need OCR or want to use an HTTP OCR server instead.
- **`ocria-nano`** (default) — `liteparse models pull ocria-nano`: the ocria-nano OCR backend, LayoutNet and the ONNX Runtime library (loaded at run time, never linked).
- **`indra-ocr`** — In-process Indra OCR (`liteparse::ocr::indra`), the default OCR backend in builds with it; shares ocria-nano's ONNX Runtime; non-WASM only.
- **`indra-ocr-auto-download`** — SHA-256-verified download and caching of the Indra models at a pinned revision, on first use.
- **`indra-ocr-cuda`** — CUDA execution provider for Indra (`Execution::Cuda`); needs CUDA and cuDNN 9 at run time. Recognizer batches then use fixed width buckets and power-of-two rows.

## Supported Formats

- PDF (`.pdf`)
- Microsoft Office (`.docx`, `.xlsx`, `.pptx`, etc.) — requires LibreOffice
- OpenDocument (`.odt`, `.ods`, `.odp`) — requires LibreOffice
- Images (`.png`, `.jpg`, `.tiff`, etc.)

## CLI

The crate also builds the `liteparse` CLI binary:

```bash
liteparse parse document.pdf
liteparse parse document.pdf --format json -o output.json
liteparse parse document.pdf --format markdown -o output.md
liteparse screenshot document.pdf -o ./screenshots
liteparse batch-parse ./input ./output
liteparse is-complex document.pdf
```

See `liteparse --help` for all options.

## License

Apache-2.0
