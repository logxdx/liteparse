//! Parse scanned PDFs through LiteParse with a chosen OCR engine and write the
//! document text, for end-to-end engine comparisons.
//!
//! `OCR_ENGINE`: `indra` (line boxes, default), `indra-words` (word boxes;
//! both need the `indra-ocr` feature and `INDRA_OCR_MODEL_DIR`) or `tesseract`.
//! `OCR_DPI`: render resolution (default LiteParse's). Prints seconds per page
//! and the peak resident set size.
//!
//! `cargo run --release --example ocr_parse --features indra-ocr -- <out_dir> <file.pdf>...`

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use liteparse::LiteParse;
use liteparse::config::LiteParseConfig;
use liteparse::ocr::OcrEngine;

fn engine(name: &str) -> Result<Arc<dyn OcrEngine>, Box<dyn std::error::Error>> {
    match name {
        #[cfg(feature = "indra-ocr")]
        "indra" | "indra-words" => {
            use liteparse::ocr::indra::models::ModelFiles;
            use liteparse::ocr::indra::{IndraOcrEngine, IndraOptions};
            // OCR_INDRA_LITE=1: no language models and no script routing.
            let lite = std::env::var_os("OCR_INDRA_LITE").is_some();
            let options = IndraOptions {
                word_boxes: name == "indra-words",
                language_model: !lite,
                scripts: !lite,
                ..IndraOptions::default()
            };
            Ok(Arc::new(IndraOcrEngine::new(
                &ModelFiles::from_env()?,
                options,
            )?))
        }
        #[cfg(feature = "tesseract")]
        "tesseract" => Ok(Arc::new(
            liteparse::ocr::tesseract::TesseractOcrEngine::new(None),
        )),
        other => Err(format!("unknown or disabled OCR_ENGINE {other:?}").into()),
    }
}

/// Peak resident set size in MiB (Linux), if readable.
fn peak_rss_mib() -> Option<f64> {
    let status = std::fs::read_to_string("/proc/self/status").ok()?;
    let line = status.lines().find(|l| l.starts_with("VmHWM:"))?;
    let kb: f64 = line.split_whitespace().nth(1)?.parse().ok()?;
    Some(kb / 1024.0)
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let out_dir = PathBuf::from(args.next().ok_or("usage: ocr_parse <out_dir> <pdf>...")?);
    std::fs::create_dir_all(&out_dir)?;
    let name = std::env::var("OCR_ENGINE").unwrap_or_else(|_| "indra".into());
    let mut config = LiteParseConfig {
        ocr_enabled: true,
        quiet: true,
        ..LiteParseConfig::default()
    };
    if let Ok(dpi) = std::env::var("OCR_DPI") {
        config.dpi = dpi.parse()?;
    }
    let dpi = config.dpi;
    let parser = LiteParse::new(config).with_ocr_engine(engine(&name)?);
    eprintln!(
        "{name}: engine loaded, RSS {:.0} MiB",
        peak_rss_mib().unwrap_or(f64::NAN)
    );

    let (mut secs, mut pages) = (0.0, 0usize);
    for path in args {
        let t = Instant::now();
        let result = parser.parse(&path).await?;
        secs += t.elapsed().as_secs_f64();
        pages += result.pages.len();
        let stem = PathBuf::from(&path);
        let stem = stem
            .file_stem()
            .ok_or("bad path")?
            .to_string_lossy()
            .into_owned();
        std::fs::write(out_dir.join(format!("{stem}.txt")), &result.text)?;
    }
    eprintln!(
        "{name} @ {dpi} dpi: {pages} pages, {:.2}s/page, peak RSS {:.0} MiB",
        secs / pages.max(1) as f64,
        peak_rss_mib().unwrap_or(f64::NAN)
    );
    Ok(())
}
