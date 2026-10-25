//! Phase 0 probe: load Indra's det/rec models, print their I/O signatures and
//! run one dummy tensor through each.
//!
//! `INDRA_OCR_MODEL_DIR=<dir> cargo run --example indra_probe --features indra-ocr`

use liteparse::ocr::indra::models::{Meta, ModelFiles};
use ort::session::Session;
use ort::value::Tensor;

fn describe(name: &str, s: &Session) {
    println!("{name}:");
    for i in s.inputs() {
        println!("  in  {} {:?}", i.name(), i.dtype());
    }
    for o in s.outputs() {
        println!("  out {} {:?}", o.name(), o.dtype());
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let files = ModelFiles::from_env()?;
    let meta = Meta::parse(&std::fs::read(&files.meta)?)?;
    println!("meta: {meta:?}");

    let mut det = Session::builder()?.commit_from_file(&files.det)?;
    let mut rec = Session::builder()?.commit_from_file(&files.rec)?;
    describe("det", &det);
    describe("rec", &rec);

    let x = Tensor::from_array(([1usize, 1, 64, 64], vec![0f32; 64 * 64]))?;
    let out = det.run(ort::inputs!["x" => x])?;
    println!("det out shape: {:?}", out[0].try_extract_tensor::<f32>()?.0);

    let h = (meta.rec_height * meta.rec_window) as usize;
    let x = Tensor::from_array(([1usize, 1, h, 64], vec![0f32; h * 64]))?;
    let w = Tensor::from_array(([1usize], vec![64i64]))?;
    let out = rec.run(ort::inputs!["x" => x, "widths" => w])?;
    for (name, v) in out.iter() {
        println!("rec out {name}: {:?}", v.dtype());
    }
    Ok(())
}
