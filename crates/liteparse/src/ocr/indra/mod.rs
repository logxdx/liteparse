//! In-process [Indra OCR](https://github.com/datalab-to/indra) engine.
//!
//! Runs Indra's ONNX models through `ort` inside the LiteParse process, as an
//! alternative to the HTTP server in `ocr/indraocr`. Model weights are not
//! bundled (they are under a modified OpenRAIL-M license): point the engine at a
//! model directory with [`IndraOcrEngine::from_dir`] or the
//! `INDRA_OCR_MODEL_DIR` environment variable ([`IndraOcrEngine::from_env`]),
//! or download them with the `indra-ocr-auto-download` feature.
//!
//! Per page: skew search and `orient.onnx` (turn and straighten the page),
//! line detection (`det.onnx`, with a second pass for small text), script
//! routing (`script.onnx`) to the base, CJK, right-to-left and world
//! recognizers, greedy CTC with an n-gram language-model re-read of unsure
//! lines, then lines split at wide gaps and column gutters. Boxes are mapped
//! back to the page as given; on a turned or straightened page each result
//! carries a 4-point polygon in its upright reading frame. Layout and reading
//! order are LiteParse's own (`projection`, `markdown_layout`).

use std::io;
use std::path::Path;
use std::pin::Pin;
use std::sync::{Mutex, Once};

use ort::ep;
use ort::session::Session;
use ort::value::Tensor;

use super::{OcrEngine, OcrOptions, OcrResult};
use crate::error::LiteParseError;

pub mod assemble;
pub mod crop;
pub mod detect;
#[cfg(feature = "indra-ocr-auto-download")]
pub mod download;
pub mod imgproc;
pub mod lm;
pub mod models;
mod npz;
pub mod orient;
pub mod recognize;
pub mod rtl;

use assemble::{Assembler, TextBox};
use detect::{LineBox, adaptive_long_side, decode, page_to_input, to_page};
use imgproc::{Gray, GrayBuf};
use lm::NgramLm;
use models::{LmParams, Meta, ModelFiles};
use orient::{
    DESKEW_MIN, Frame, ORIENT_MIN, ORIENT_SIDE, estimate_skew, orient_input, preview, rotate_page,
    turn,
};
use recognize::{Batching, Readers, Recognizer, ScriptRouter};

type BoxError = Box<dyn std::error::Error + Send + Sync>;

impl<R> From<ort::Error<R>> for LiteParseError {
    fn from(err: ort::Error<R>) -> Self {
        LiteParseError::Ocr(err.to_string())
    }
}

/// Where ONNX Runtime runs the models.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub enum Execution {
    #[default]
    Cpu,
    /// CUDA on this device (needs the `indra-ocr-cuda` feature, CUDA and cuDNN 9
    /// on the loader path). Fails at construction when the provider cannot load.
    Cuda { device_id: i32 },
}

/// Engine settings. The defaults run the whole pipeline the model directory
/// supports on the CPU.
#[derive(Debug, Clone)]
pub struct IndraOptions {
    /// ONNX Runtime intra-op threads per session (default: ONNX Runtime's choice).
    pub intra_threads: Option<usize>,
    /// Skew search and `orient.onnx`: turn and straighten pages before reading.
    pub orientation: bool,
    /// Route lines to the group recognizers (`script.onnx`, `rec_<group>.onnx`).
    pub scripts: bool,
    /// Re-read unsure lines with the n-gram language models (`lm*.npz`).
    pub language_model: bool,
    /// One result per word (lines cut at spaces) instead of per line.
    pub word_boxes: bool,
    pub execution: Execution,
    /// Recognizer batch shapes; `None` picks the provider's default (CUDA:
    /// widths padded to 1024 and power-of-two rows, which keeps the number of
    /// distinct shapes small; CPU: Indra's own, since padding costs CPU time).
    pub batching: Option<Batching>,
}

impl Default for IndraOptions {
    fn default() -> Self {
        Self {
            intra_threads: None,
            orientation: true,
            scripts: true,
            language_model: true,
            word_boxes: false,
            execution: Execution::Cpu,
            batching: None,
        }
    }
}

/// Native Indra OCR adapter for LiteParse.
///
/// Each model sits behind its own mutex (`Session::run` needs exclusive access),
/// so concurrent page calls serialize per model. The CPU memory arena is off:
/// ONNX Runtime otherwise keeps its largest allocation, which held gigabytes on
/// large scans.
pub struct IndraOcrEngine {
    det: Mutex<Session>,
    orient: Option<Mutex<Session>>,
    readers: Readers,
    det_long_side: usize,
    /// Turn and straighten pages (the skew search needs no model).
    orientation: bool,
    word_boxes: bool,
}

fn session(path: &Path, options: &IndraOptions) -> Result<Session, LiteParseError> {
    let cpu = ep::CPU::default().with_arena_allocator(false).build();
    let mut builder = Session::builder()?;
    builder = match options.execution {
        Execution::Cpu => builder.with_execution_providers([cpu])?,
        #[cfg(feature = "indra-ocr-cuda")]
        Execution::Cuda { device_id } => builder.with_execution_providers([
            ep::CUDA::default()
                .with_device_id(device_id)
                .build()
                .error_on_failure(),
            cpu,
        ])?,
        #[cfg(not(feature = "indra-ocr-cuda"))]
        Execution::Cuda { .. } => {
            return Err(LiteParseError::Ocr(
                "indra: Execution::Cuda needs the `indra-ocr-cuda` feature".into(),
            ));
        }
    };
    if let Some(n) = options.intra_threads {
        builder = builder.with_intra_threads(n)?;
    }
    Ok(builder.commit_from_file(path)?)
}

fn load_lm(
    path: Option<std::path::PathBuf>,
    params: LmParams,
) -> Result<Option<(NgramLm, LmParams)>, LiteParseError> {
    match path {
        Some(p) => Ok(Some((NgramLm::from_npz(&std::fs::read(p)?)?, params))),
        None => Ok(None),
    }
}

impl IndraOcrEngine {
    /// Load from a model directory (at least `det.onnx`, `rec.onnx` and
    /// `meta.json`) with default options.
    pub fn from_dir(dir: impl AsRef<Path>) -> Result<Self, LiteParseError> {
        Self::new(&ModelFiles::in_dir(dir.as_ref())?, IndraOptions::default())
    }

    /// Load from the directory named by `INDRA_OCR_MODEL_DIR`.
    pub fn from_env() -> Result<Self, LiteParseError> {
        Self::new(&ModelFiles::from_env()?, IndraOptions::default())
    }

    /// Load from resolved model files with explicit options. Optional models
    /// missing from the directory switch their step off.
    pub fn new(files: &ModelFiles, options: IndraOptions) -> Result<Self, LiteParseError> {
        let meta = Meta::parse(&std::fs::read(&files.meta)?)?;
        let Meta {
            det_long_side,
            rec_height,
            rec_window,
            vocab,
            script_groups,
            mut group_vocabs,
            lm_params,
        } = meta;
        let lm_for = |group: &str| -> Result<Option<(NgramLm, LmParams)>, LiteParseError> {
            if !options.language_model {
                return Ok(None);
            }
            let file = if group == "base" {
                "lm.npz".to_owned()
            } else {
                format!("lm_{group}.npz")
            };
            load_lm(
                files.optional(&file),
                lm_params.get(group).copied().unwrap_or_default(),
            )
        };

        let base = Recognizer::new(
            session(&files.rec, &options)?,
            vocab,
            lm_for("base")?,
            true,
            false,
        );
        let mut groups = Vec::new();
        let mut script = None;
        if options.scripts {
            let mut names: Vec<String> = group_vocabs.keys().cloned().collect();
            names.sort();
            for name in names {
                if let Some(path) = files.optional(&format!("rec_{name}.onnx")) {
                    let vocab = group_vocabs.remove(&name).expect("listed above");
                    let rec = Recognizer::new(
                        session(&path, &options)?,
                        vocab,
                        lm_for(&name)?,
                        false,
                        true,
                    );
                    groups.push((name, rec));
                }
            }
            if !groups.is_empty()
                && let Some(path) = files.optional("script.onnx")
            {
                script = Some(ScriptRouter::new(session(&path, &options)?, script_groups));
            }
        }
        let orient = match (options.orientation, files.optional("orient.onnx")) {
            (true, Some(p)) => Some(Mutex::new(session(&p, &options)?)),
            _ => None,
        };
        let batching = options.batching.unwrap_or(match options.execution {
            Execution::Cpu => Batching::default(),
            Execution::Cuda { .. } => Batching {
                width_multiple: 1024,
                pow2_rows: true,
            },
        });
        Ok(Self {
            det: Mutex::new(session(&files.det, &options)?),
            orient,
            readers: Readers {
                base,
                groups,
                script,
                rec_height: rec_height as usize,
                window: rec_window.max(1) as usize,
                batching,
            },
            det_long_side: det_long_side as usize,
            orientation: options.orientation,
            word_boxes: options.word_boxes,
        })
    }

    /// Loaded optional steps, for logs: orientation, script groups, LMs.
    pub fn describe(&self) -> String {
        let groups: Vec<&str> = self
            .readers
            .groups
            .iter()
            .map(|(n, _)| n.as_str())
            .collect();
        format!(
            "indra-ocr: orientation={} scripts={} groups={:?} word_boxes={}",
            self.orient.is_some(),
            self.readers.script.is_some(),
            groups,
            self.word_boxes
        )
    }

    /// `orient.onnx` class probabilities of a (straightened) preview.
    fn orientation(&self, small: &Gray) -> Result<usize, BoxError> {
        let Some(orient) = &self.orient else {
            return Ok(0);
        };
        let x = Tensor::from_array(([1usize, 1, ORIENT_SIDE, ORIENT_SIDE], orient_input(small)))?;
        let mut s = orient.lock().map_err(|_| {
            io::Error::other("indra orientation mutex was poisoned by a previous panic")
        })?;
        let out = s.run(ort::inputs!["x" => x])?;
        let (_, p) = out[0].try_extract_tensor::<f32>()?;
        let mut k = 0;
        for (i, &v) in p.iter().enumerate() {
            if v > p[k] {
                k = i;
            }
        }
        Ok(if p[k] > ORIENT_MIN { k } else { 0 })
    }

    /// The page turned upright and straightened, with its frame.
    fn upright(&self, gray: &Gray) -> Result<(GrayBuf, Frame), BoxError> {
        let small = preview(gray);
        let a = estimate_skew(&small.view());
        let k = if self.orient.is_some() {
            if a.abs() >= DESKEW_MIN {
                self.orientation(&rotate_page(&small.view(), a).view())?
            } else {
                self.orientation(&small.view())?
            }
        } else {
            0
        };
        Ok(turn(gray, k, a, &small.view()))
    }

    fn detect(&self, gray: &Gray) -> Result<Vec<LineBox>, BoxError> {
        let mut det = self.det.lock().map_err(|_| {
            io::Error::other("indra detector mutex was poisoned by a previous panic")
        })?;
        let mut run = |long_side: usize| -> Result<(Vec<LineBox>, f32), ort::Error> {
            let inp = page_to_input(gray, long_side);
            let x = Tensor::from_array(([1usize, 1, inp.height, inp.width], inp.data))?;
            let outputs = det.run(ort::inputs!["x" => x])?;
            let (shape, out) = outputs[0].try_extract_tensor::<f32>()?;
            Ok((decode(out, shape[2] as usize, shape[3] as usize), inp.scale))
        };
        let (mut boxes, mut scale) = run(self.det_long_side)?;
        if let Some(ls) = adaptive_long_side(&boxes, self.det_long_side) {
            (boxes, scale) = run(ls)?;
        }
        to_page(&mut boxes, scale, gray.width, gray.height);
        Ok(boxes)
    }

    /// OCR of one page: text boxes in the page's own pixels.
    pub fn read_page(&self, gray: &Gray) -> Result<Vec<OcrResult>, BoxError> {
        let timings = std::env::var_os("INDRA_OCR_TIMINGS").is_some();
        let t0 = std::time::Instant::now();
        let (page, frame) = if self.orientation {
            self.upright(gray)?
        } else {
            (gray.to_buf(), identity(gray))
        };
        let t_orient = t0.elapsed();
        let page = page.view();
        let boxes = self.detect(&page)?;
        let t_det = t0.elapsed();
        if boxes.is_empty() {
            return Ok(Vec::new());
        }
        let lines = self.readers.read(&page, &boxes, frame.vertical())?;
        if timings {
            let ms = |d: std::time::Duration| d.as_secs_f64() * 1000.0;
            eprintln!(
                "[indra-ocr] orient {:.0} ms (class {}, skew {:.1}), det {:.0} ms, rec {:.0} ms, {} lines",
                ms(t_orient),
                frame.class,
                frame.skew,
                ms(t_det - t_orient),
                ms(t0.elapsed() - t_det),
                boxes.len()
            );
        }
        let assembler = Assembler::new(&self.readers.base.vocab, &lines);
        let line_boxes: Vec<[f32; 4]> = boxes.iter().map(|b| [b.x0, b.y0, b.x1, b.y1]).collect();
        let mut out = Vec::new();
        for (line, toks) in assembler.assemble(&line_boxes, &lines) {
            if self.word_boxes {
                for w in assembler.words(&line, &toks) {
                    out.push(to_result(w, &frame));
                }
            } else {
                out.push(to_result(line, &frame));
            }
        }
        Ok(out)
    }
}

fn identity(gray: &Gray) -> Frame {
    Frame {
        class: 0,
        skew: 0.0,
        orig_w: gray.width,
        orig_h: gray.height,
        w: gray.width,
        h: gray.height,
    }
}

/// A box of the page that was read, on the page as given. A turned or
/// straightened page gets the 4-point polygon in the text's reading frame; a
/// vertical (CJK column) page keeps axis-aligned boxes.
fn to_result(tb: TextBox, frame: &Frame) -> OcrResult {
    let [x0, y0, x1, y1] = tb.bbox;
    if frame.is_identity() {
        return OcrResult {
            text: tb.text,
            bbox: tb.bbox,
            confidence: tb.conf.clamp(0.0, 1.0),
            polygon: None,
        };
    }
    let (w, h) = (frame.orig_w as f32, frame.orig_h as f32);
    let corner = |x: f32, y: f32| {
        let (px, py) = frame.to_page(x as f64, y as f64);
        [(px as f32).clamp(0.0, w), (py as f32).clamp(0.0, h)]
    };
    let poly = [
        corner(x0, y0),
        corner(x1, y0),
        corner(x1, y1),
        corner(x0, y1),
    ];
    let xs = poly.map(|p| p[0]);
    let ys = poly.map(|p| p[1]);
    let bbox = [
        xs.iter().copied().fold(f32::INFINITY, f32::min),
        ys.iter().copied().fold(f32::INFINITY, f32::min),
        xs.iter().copied().fold(f32::NEG_INFINITY, f32::max),
        ys.iter().copied().fold(f32::NEG_INFINITY, f32::max),
    ];
    OcrResult {
        text: tb.text,
        bbox,
        confidence: tb.conf.clamp(0.0, 1.0),
        polygon: (!frame.vertical()).then_some(poly),
    }
}

impl OcrEngine for IndraOcrEngine {
    fn name(&self) -> &str {
        "indra-ocr"
    }

    /// Indra reads a single-channel ink map, so a luma buffer saves the
    /// conversion (RGB buffers are still accepted).
    fn prefers_grayscale(&self) -> bool {
        true
    }

    fn recognize<'a, 'b: 'a, 'c: 'a>(
        &'a self,
        image_data: &'c [u8],
        width: u32,
        height: u32,
        options: &'b OcrOptions,
    ) -> Pin<
        Box<
            dyn Future<Output = Result<Vec<OcrResult>, Box<dyn std::error::Error + Send + Sync>>>
                + Send
                + '_,
        >,
    > {
        warn_language_ignored_once(&options.language);
        // `ocr_and_merge_rendered` polls this future on a blocking worker, so
        // synchronous ONNX inference does not occupy an async runtime worker.
        Box::pin(async move {
            let pixels = luma_pixels(image_data, width, height)?;
            self.read_page(&Gray {
                data: &pixels,
                width: width as usize,
                height: height as usize,
            })
        })
    }
}

static LANGUAGE_IGNORED_WARNING: Once = Once::new();

/// Warn once per process that this backend ignores `OcrOptions::language`.
fn warn_language_ignored_once(language: &str) {
    if language.is_empty() || language == "eng" || language == "en" {
        return;
    }
    LANGUAGE_IGNORED_WARNING.call_once(|| {
        eprintln!(
            "[indra-ocr] ignoring OcrOptions::language ({language:?}); the script classifier picks the recognizer per line"
        );
    });
}

/// Single-channel pixels from a 1- or 3-bytes-per-pixel buffer (3 channels use
/// the `cv2` BT.601 luma weights).
fn luma_pixels(image_data: &[u8], width: u32, height: u32) -> Result<Vec<u8>, io::Error> {
    let pixels = (width as usize)
        .checked_mul(height as usize)
        .filter(|&n| n > 0)
        .ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("invalid image size: {width}x{height}"),
            )
        })?;
    match image_data.len() {
        n if n == pixels => Ok(image_data.to_vec()),
        n if n == pixels * 3 => Ok(image_data
            .chunks(3)
            .map(|p| {
                (0.299 * p[0] as f32 + 0.587 * p[1] as f32 + 0.114 * p[2] as f32).round() as u8
            })
            .collect()),
        n => Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!(
                "invalid buffer length for {width}x{height}: expected {pixels} (gray) or {} (RGB) bytes, got {n}",
                pixels * 3
            ),
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn luma_accepts_gray_and_rgb() {
        assert_eq!(luma_pixels(&[7, 9], 2, 1).unwrap(), vec![7, 9]);
        assert_eq!(
            luma_pixels(&[255, 255, 255, 0, 0, 0], 2, 1).unwrap(),
            vec![255, 0]
        );
        assert!(luma_pixels(&[1, 2, 3, 4], 2, 1).is_err());
        assert!(luma_pixels(&[], 0, 1).is_err());
    }

    #[test]
    fn polygons_follow_the_turn() {
        // Content turned a quarter turn CCW (class 1) on a 100 x 200 page: the
        // read page is 200 x 100. A line along the read page's top edge reads
        // upwards on the page as given.
        let frame = Frame {
            class: 1,
            skew: 0.0,
            orig_w: 100,
            orig_h: 200,
            w: 200,
            h: 100,
        };
        let r = to_result(
            TextBox {
                bbox: [10.0, 0.0, 110.0, 20.0],
                text: "x".into(),
                conf: 0.9,
            },
            &frame,
        );
        let p = r.polygon.unwrap();
        // TL (10, 0) -> (0, 190); TR (110, 0) -> (0, 90): reading direction is up.
        assert_eq!(p[0], [0.0, 190.0]);
        assert_eq!(p[1], [0.0, 90.0]);
        assert_eq!(r.bbox, [0.0, 90.0, 20.0, 190.0]);
    }
}
