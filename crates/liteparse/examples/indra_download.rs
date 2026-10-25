//! Download the Indra models (if not cached), load the engine and OCR one image.
//!
//! `INDRA_OCR_HOME=<cache> cargo run --release --example indra_download \
//!    --features indra-ocr-auto-download -- [base|full] [page.png]`

use liteparse::ocr::indra::download::{ModelSet, cache_dir};
use liteparse::ocr::indra::{IndraOcrEngine, IndraOptions};
use liteparse::ocr::{OcrEngine, OcrOptions};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let set = match args.next().as_deref() {
        Some("full") => ModelSet::Full,
        _ => ModelSet::Base,
    };
    let engine = IndraOcrEngine::download(set, IndraOptions::default()).await?;
    eprintln!("{} (cache {})", engine.describe(), cache_dir()?.display());
    if let Some(page) = args.next() {
        let img = image::open(&page)?.to_luma8();
        let (w, h) = img.dimensions();
        let options = OcrOptions {
            language: String::new(),
            dpi: 200.0,
        };
        let lines = engine
            .recognize(img.as_raw(), w, h, &options)
            .await
            .map_err(|e| e.to_string())?;
        for l in lines.iter().take(5) {
            println!("{:.2} {:?}", l.confidence, l.text);
        }
        eprintln!("{} lines", lines.len());
    }
    Ok(())
}
