//! Run an OCR engine over page images and write one text file per page (lines
//! in reading order), for parity checks and engine comparisons.
//!
//! `OCR_ENGINE` selects the engine: `indra` (default, needs the `indra-ocr`
//! feature and `INDRA_OCR_MODEL_DIR`), `indra-cuda` (`indra-ocr-cuda`) or
//! `tesseract`. `OCR_LANG` is passed as
//! `OcrOptions::language` (default `eng`; Indra ignores it). Boxes on one row
//! are joined into a text line, unless `OCR_BOX_LINES` is set (one box per line).
//!
//! `INDRA_OCR_MODEL_DIR=<dir> cargo run --release --example ocr_pages \
//!    --features indra-ocr -- <out_dir> <page.png>...`

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use liteparse::ocr::{OcrEngine, OcrOptions};

fn engine(name: &str) -> Result<Arc<dyn OcrEngine>, Box<dyn std::error::Error>> {
    match name {
        #[cfg(feature = "indra-ocr")]
        "indra" => Ok(Arc::new(liteparse::ocr::indra::IndraOcrEngine::from_env()?)),
        #[cfg(feature = "indra-ocr-cuda")]
        "indra-cuda" => {
            use liteparse::ocr::indra::models::ModelFiles;
            use liteparse::ocr::indra::recognize::Batching;
            use liteparse::ocr::indra::{Execution, IndraOcrEngine, IndraOptions};
            // OCR_BATCH="<width multiple>,<pow2 rows 0|1>" overrides the CUDA default.
            let batching = std::env::var("OCR_BATCH").ok().and_then(|s| {
                let (w, p) = s.split_once(',')?;
                Some(Batching {
                    width_multiple: w.parse().ok()?,
                    pow2_rows: p == "1",
                })
            });
            let options = IndraOptions {
                execution: Execution::Cuda { device_id: 0 },
                batching,
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

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let out_dir = PathBuf::from(args.next().ok_or("usage: ocr_pages <out_dir> <png>...")?);
    std::fs::create_dir_all(&out_dir)?;
    let engine = engine(&std::env::var("OCR_ENGINE").unwrap_or_else(|_| "indra".into()))?;
    // IO and timers: Tesseract downloads missing traineddata on first use.
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()?;
    let options = OcrOptions {
        language: std::env::var("OCR_LANG").unwrap_or_else(|_| "eng".into()),
        dpi: 200.0,
    };

    let mut total = 0.0;
    let mut pages = 0;
    for path in args {
        let img = image::open(&path)?.to_luma8();
        let (w, h) = img.dimensions();
        let t = Instant::now();
        let mut boxes = rt
            .block_on(engine.recognize(img.as_raw(), w, h, &options))
            .map_err(|e| e.to_string())?;
        let secs = t.elapsed().as_secs_f64();
        total += secs;
        pages += 1;
        // Reading order: boxes sorted by centre y join a row while their centre
        // is within half a box height of the row's first box; rows read left to
        // right. Word-level engines (Tesseract) get one text line per row.
        let cy = |b: &[f32; 4]| (b[1] + b[3]) / 2.0;
        boxes.sort_by(|a, b| cy(&a.bbox).total_cmp(&cy(&b.bbox)));
        let mut rows: Vec<Vec<&liteparse::ocr::OcrResult>> = Vec::new();
        for b in &boxes {
            let joins = rows.last().is_some_and(|r| {
                let first = r[0].bbox;
                (cy(&b.bbox) - cy(&first)).abs() <= (first[3] - first[1]) / 2.0
            });
            if joins {
                rows.last_mut().unwrap().push(b);
            } else {
                rows.push(vec![b]);
            }
        }
        let mut text = String::new();
        if std::env::var("OCR_BOX_LINES").is_ok() {
            // One result per text line, as the engine returned them.
            for b in &boxes {
                text.push_str(&b.text);
                text.push('\n');
            }
        } else {
            for row in &mut rows {
                row.sort_by(|a, b| a.bbox[0].total_cmp(&b.bbox[0]));
                let words: Vec<&str> = row.iter().map(|b| b.text.as_str()).collect();
                text.push_str(&words.join(" "));
                text.push('\n');
            }
        }
        let lines = boxes;
        let stem = PathBuf::from(&path);
        let stem = stem
            .file_stem()
            .ok_or("bad path")?
            .to_string_lossy()
            .into_owned();
        std::fs::write(out_dir.join(format!("{stem}.txt")), text)?;
        eprintln!("{stem}: {} boxes, {secs:.2}s", lines.len());
    }
    eprintln!("{pages} pages, {:.2}s/page", total / pages.max(1) as f64);
    Ok(())
}
