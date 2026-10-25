//! Parity of the in-process Indra engine with the Python reference.
//!
//! Skipped unless all three variables are set:
//! - `INDRA_OCR_MODEL_DIR`: model directory (`det.onnx`, `rec.onnx`, `meta.json`)
//! - `INDRA_PARITY_IMAGES`: directory of page images (`pNNN.png`)
//! - `INDRA_PARITY_REF`: directory of Python Indra text for the same pages (`pNNN.txt`)
//!
//! Targets (plan, Phase 1): at least 95% of lines identical to a reference line,
//! and word-multiset F1 against the reference within 0.002 of perfect agreement
//! on the corpus.
#![cfg(feature = "indra-ocr")]

use std::collections::HashMap;
use std::path::PathBuf;

use liteparse::ocr::indra::IndraOcrEngine;
use liteparse::ocr::{OcrEngine, OcrOptions};

fn env_dir(name: &str) -> Option<PathBuf> {
    std::env::var_os(name).map(PathBuf::from)
}

fn counts<'a>(items: impl Iterator<Item = &'a str>) -> HashMap<&'a str, usize> {
    let mut m = HashMap::new();
    for i in items {
        *m.entry(i).or_insert(0) += 1;
    }
    m
}

fn overlap(a: &HashMap<&str, usize>, b: &HashMap<&str, usize>) -> usize {
    a.iter()
        .map(|(k, n)| (*n).min(*b.get(k).unwrap_or(&0)))
        .sum()
}

#[test]
fn matches_python_reference() {
    let (Some(_), Some(images), Some(reference)) = (
        env_dir("INDRA_OCR_MODEL_DIR"),
        env_dir("INDRA_PARITY_IMAGES"),
        env_dir("INDRA_PARITY_REF"),
    ) else {
        eprintln!("skipped: set INDRA_OCR_MODEL_DIR, INDRA_PARITY_IMAGES, INDRA_PARITY_REF");
        return;
    };
    let engine = IndraOcrEngine::from_env().expect("load indra models");
    let rt = tokio::runtime::Builder::new_current_thread()
        .build()
        .unwrap();
    let options = OcrOptions {
        language: String::new(),
        dpi: 200.0,
    };

    let mut pages: Vec<PathBuf> = std::fs::read_dir(&images)
        .unwrap()
        .map(|e| e.unwrap().path())
        .filter(|p| p.extension().is_some_and(|e| e == "png"))
        .collect();
    pages.sort();
    assert!(!pages.is_empty(), "no pages in {}", images.display());

    let (mut same_lines, mut lines, mut shared_words, mut rust_words, mut ref_words) =
        (0, 0, 0, 0, 0);
    for page in &pages {
        let img = image::open(page).unwrap().to_luma8();
        let (w, h) = img.dimensions();
        let results = rt
            .block_on(engine.recognize(img.as_raw(), w, h, &options))
            .unwrap();
        let ours: Vec<String> = results.into_iter().map(|r| r.text).collect();
        let theirs = std::fs::read_to_string(
            reference
                .join(page.file_stem().unwrap())
                .with_extension("txt"),
        )
        .unwrap();

        let mut b = counts(theirs.lines().map(str::trim).filter(|l| !l.is_empty()));
        lines += ours.len();
        for l in ours.iter().map(|s| s.trim()) {
            if let Some(n) = b.get_mut(l).filter(|n| **n > 0) {
                *n -= 1;
                same_lines += 1;
            }
        }
        let ow = counts(ours.iter().flat_map(|s| s.split_whitespace()));
        let tw = counts(theirs.split_whitespace());
        shared_words += overlap(&ow, &tw);
        rust_words += ow.values().sum::<usize>();
        ref_words += tw.values().sum::<usize>();
    }
    let line_ratio = same_lines as f64 / lines as f64;
    let (p, r) = (
        shared_words as f64 / rust_words as f64,
        shared_words as f64 / ref_words as f64,
    );
    let f1 = 2.0 * p * r / (p + r);
    eprintln!(
        "lines identical {same_lines}/{lines} = {line_ratio:.4}; word F1 vs reference {f1:.4}"
    );
    assert!(
        line_ratio >= 0.95,
        "only {line_ratio:.3} of lines match the reference"
    );
    assert!(f1 >= 0.998, "word F1 vs reference {f1:.4}");
}
