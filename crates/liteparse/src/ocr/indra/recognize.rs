//! Line recognition: script routing, the base and group recognizers, greedy
//! CTC with page positions per token, and the language-model re-read of
//! unsure lines (port of `OrtOCR.recognize` / `line_groups`).

use std::io;
use std::sync::Mutex;

use ort::session::Session;
use ort::value::Tensor;

use super::assemble::{LineRead, Tok, Token};
use super::crop::{PAD_X, crop_line, is_dark_line, line_crop_width, pad_batch, width_batches};
use super::detect::LineBox;
use super::imgproc::Gray;
use super::lm::{BeamParams, NgramLm, beam_search};
use super::models::{LmParams, Vocab};

type BoxError = Box<dyn std::error::Error + Send + Sync>;

/// A recognized line under this confidence (mean CTC peak probability) gets no
/// text: empty form fields and rules read as "-", "I" or "1" at low confidence.
pub const MIN_CONF: f32 = 0.5;
/// Lines per recognizer batch.
pub const REC_BATCH: usize = 64;
/// A line goes to one recognizer when the script classifier is surer than
/// this; otherwise the base recognizer reads it as well and the more
/// confident reading wins.
const ROUTE_SURE: f32 = 0.95;
const LM_BEAM: usize = 8;
/// A class under this probability at a step is not tried by the beam search.
const LM_CAND_FLOOR: f64 = 3e-3;

/// Shape policy for recognizer batches.
#[derive(Debug, Clone, Copy)]
pub struct Batching {
    /// Pad batch widths to a multiple of this (8 = the CTC step stride).
    pub width_multiple: usize,
    /// Pad the number of lines in a batch up to a power of two.
    pub pow2_rows: bool,
}

impl Default for Batching {
    fn default() -> Self {
        Self {
            width_multiple: 8,
            pow2_rows: false,
        }
    }
}

fn lock(m: &Mutex<Session>) -> Result<std::sync::MutexGuard<'_, Session>, io::Error> {
    m.lock()
        .map_err(|_| io::Error::other("indra session mutex was poisoned by a previous panic"))
}

/// Outputs of one recognizer batch, row-major.
struct RecBatch {
    ids: Vec<i64>,
    conf: Vec<f32>,
    lengths: Vec<i64>,
    t: usize,
    lm: Option<LmOutputs>,
}

/// The beam-search inputs of a batch: blank log-probabilities `[B, T]` and
/// the `k` best classes per step, `topv` / `topi` `[B, T, k]`.
struct LmOutputs {
    blank: Vec<f32>,
    topv: Vec<f32>,
    topi: Vec<i64>,
    k: usize,
}

/// One CTC line recognizer and its vocabulary and language model.
pub struct Recognizer {
    session: Mutex<Session>,
    pub vocab: Vocab,
    lm: Option<(NgramLm, LmParams)>,
    /// Reads the tall window (`rec.onnx`); group recognizers read the line band.
    pub tall: bool,
    /// Tokens are characters of its own vocabulary (group recognizers).
    pub as_chars: bool,
}

impl Recognizer {
    pub fn new(
        session: Session,
        vocab: Vocab,
        lm: Option<(NgramLm, LmParams)>,
        tall: bool,
        as_chars: bool,
    ) -> Self {
        Self {
            session: Mutex::new(session),
            vocab,
            lm,
            tall,
            as_chars,
        }
    }

    fn run(
        &self,
        data: Vec<f32>,
        shape: [usize; 4],
        widths: Vec<i64>,
    ) -> Result<RecBatch, BoxError> {
        let mut s = lock(&self.session)?;
        let x = Tensor::from_array((shape, data))?;
        let w = Tensor::from_array(([shape[0]], widths))?;
        let out = s.run(ort::inputs!["x" => x, "widths" => w])?;
        let (ids_shape, ids) = out["ids"].try_extract_tensor::<i64>()?;
        let t = ids_shape[1] as usize;
        let lm = match &self.lm {
            Some(_) => {
                let (_, blank) = out["blank"].try_extract_tensor::<f32>()?;
                let (k_shape, topv) = out["topv"].try_extract_tensor::<f32>()?;
                let (_, topi) = out["topi"].try_extract_tensor::<i64>()?;
                Some(LmOutputs {
                    blank: blank.to_vec(),
                    topv: topv.to_vec(),
                    topi: topi.to_vec(),
                    k: k_shape[2] as usize,
                })
            }
            None => None,
        };
        Ok(RecBatch {
            ids: ids.to_vec(),
            conf: out["conf"].try_extract_tensor::<f32>()?.1.to_vec(),
            lengths: out["lengths"].try_extract_tensor::<i64>()?.1.to_vec(),
            t,
            lm,
        })
    }
}

/// `script.onnx`: the recognizer group each line belongs to.
pub struct ScriptRouter {
    session: Mutex<Session>,
    /// Class order of the classifier.
    pub groups: Vec<String>,
}

impl ScriptRouter {
    pub fn new(session: Session, groups: Vec<String>) -> Self {
        Self {
            session: Mutex::new(session),
            groups,
        }
    }

    /// Per line crop: (group name, sure).
    fn classify(
        &self,
        crops: &[(Vec<f32>, usize)],
        rows: usize,
        batching: Batching,
    ) -> Result<Vec<(String, bool)>, BoxError> {
        let mut out = vec![("base".to_owned(), true); crops.len()];
        let widths: Vec<usize> = crops.iter().map(|c| c.1).collect();
        for idx in width_batches(&widths, REC_BATCH) {
            let lines: Vec<(&[f32], usize)> = idx
                .iter()
                .map(|&j| (crops[j].0.as_slice(), crops[j].1))
                .collect();
            let (data, w, wm, b) =
                pad_batch(&lines, rows, batching.width_multiple, batching.pow2_rows);
            let mut s = lock(&self.session)?;
            let x = Tensor::from_array(([b, 1, rows, wm], data))?;
            let wt = Tensor::from_array(([b], w))?;
            let res = s.run(ort::inputs!["x" => x, "widths" => wt])?;
            let (shape, p) = res[0].try_extract_tensor::<f32>()?;
            let classes = if shape.len() == 2 {
                shape[1] as usize
            } else {
                1
            };
            for (k, &j) in idx.iter().enumerate() {
                if classes == 1 {
                    // An older classifier: the chance the line holds Chinese.
                    let pk = p[k];
                    out[j] = (
                        if pk > 0.5 { "cjk" } else { "base" }.to_owned(),
                        pk.max(1.0 - pk) > ROUTE_SURE,
                    );
                } else {
                    let row = &p[k * classes..(k + 1) * classes];
                    let mut best = 0;
                    for (c, &v) in row.iter().enumerate() {
                        if v > row[best] {
                            best = c;
                        }
                    }
                    let name = self
                        .groups
                        .get(best)
                        .cloned()
                        .unwrap_or_else(|| "base".into());
                    out[j] = (name, row[best] > ROUTE_SURE);
                }
            }
        }
        Ok(out)
    }
}

/// The recognizers of an engine.
pub struct Readers {
    pub base: Recognizer,
    /// Group recognizers by name (`cjk`, `rtl`, `world`, ...).
    pub groups: Vec<(String, Recognizer)>,
    pub script: Option<ScriptRouter>,
    pub rec_height: usize,
    pub window: usize,
    pub batching: Batching,
}

impl Readers {
    fn group(&self, name: &str) -> Option<&Recognizer> {
        self.groups.iter().find(|(n, _)| n == name).map(|(_, r)| r)
    }

    /// Read line `boxes` of `page`. `vertical`: a vertical page turned a
    /// quarter turn, every line read by the CJK recognizer.
    pub fn read(
        &self,
        page: &Gray,
        boxes: &[LineBox],
        vertical: bool,
    ) -> Result<Vec<LineRead>, BoxError> {
        let n = boxes.len();
        let mut out = vec![LineRead::default(); n];
        if n == 0 {
            return Ok(out);
        }
        let timings = std::env::var_os("INDRA_OCR_TIMINGS").is_some();
        let t0 = std::time::Instant::now();
        // Light-on-dark is decided once per box, for both crop heights.
        let dark: Vec<bool> = boxes.iter().map(|b| is_dark_line(page, b)).collect();
        let widths: Vec<usize> = boxes.iter().map(line_crop_width).collect();
        let crops = |window: usize| -> Vec<(Vec<f32>, usize)> {
            boxes
                .iter()
                .zip(&dark)
                .map(|(b, &d)| crop_line(page, b, window, d))
                .collect()
        };
        // Line-band crops (script classifier, group recognizers), made on demand.
        let mut band: Option<Vec<(Vec<f32>, usize)>> = None;

        let mut t_script = None;
        // (recognizer, lines it reads)
        let mut plan: Vec<(&Recognizer, Vec<usize>)> = Vec::new();
        match (vertical, self.group("cjk")) {
            (true, Some(cjk)) => plan.push((cjk, (0..n).collect())),
            _ => match &self.script {
                Some(router) => {
                    let band = band.get_or_insert_with(|| crops(1));
                    let ts = std::time::Instant::now();
                    let names = router.classify(band, self.rec_height, self.batching)?;
                    t_script = Some(ts.elapsed());
                    let base: Vec<usize> = (0..n)
                        .filter(|&j| {
                            names[j].0 == "base" || !names[j].1 || self.group(&names[j].0).is_none()
                        })
                        .collect();
                    plan.push((&self.base, base));
                    for (name, rec) in &self.groups {
                        let sel: Vec<usize> = (0..n).filter(|&j| &names[j].0 == name).collect();
                        if !sel.is_empty() {
                            plan.push((rec, sel));
                        }
                    }
                }
                None => plan.push((&self.base, (0..n).collect())),
            },
        }

        let reads_tall = |r: &Recognizer| r.tall && self.window > 1;
        let tall = plan
            .iter()
            .any(|(r, s)| reads_tall(r) && !s.is_empty())
            .then(|| crops(self.window));
        if plan.iter().any(|(r, s)| !reads_tall(r) && !s.is_empty()) {
            band.get_or_insert_with(|| crops(1));
        }

        let t_crops = t0.elapsed() - t_script.unwrap_or_default();
        let (mut t_run, mut t_decode) = (std::time::Duration::ZERO, std::time::Duration::ZERO);
        let mut best_conf = vec![-1f32; n];
        for (rec, sel) in plan {
            let (src, rows) = if reads_tall(rec) {
                (tall.as_ref(), self.rec_height * self.window)
            } else {
                (band.as_ref(), self.rec_height)
            };
            let Some(src) = src else { continue };
            let sel_widths: Vec<usize> = sel.iter().map(|&j| widths[j]).collect();
            for part in width_batches(&sel_widths, REC_BATCH) {
                let idx: Vec<usize> = part.iter().map(|&i| sel[i]).collect();
                let lines: Vec<(&[f32], usize)> = idx
                    .iter()
                    .map(|&j| (src[j].0.as_slice(), src[j].1))
                    .collect();
                let (data, w, wm, b) = pad_batch(
                    &lines,
                    rows,
                    self.batching.width_multiple,
                    self.batching.pow2_rows,
                );
                let tr = std::time::Instant::now();
                let res = rec.run(data, [b, 1, rows, wm], w)?;
                t_run += tr.elapsed();
                let td = std::time::Instant::now();
                let stride = wm as f64 / res.t.max(1) as f64;
                for (k, &j) in idx.iter().enumerate() {
                    let read = read_one(rec, &res, k, stride, &boxes[j], widths[j]);
                    if read.conf <= best_conf[j] {
                        continue;
                    }
                    best_conf[j] = read.conf;
                    out[j] = read;
                }
                t_decode += td.elapsed();
            }
        }
        if timings {
            let ms = |d: std::time::Duration| d.as_secs_f64() * 1000.0;
            eprintln!(
                "[indra-ocr]   rec: crops {:.0} ms, script {:.0} ms, models {:.0} ms, decode+lm {:.0} ms",
                ms(t_crops),
                ms(t_script.unwrap_or_default()),
                ms(t_run),
                ms(t_decode)
            );
        }
        Ok(out)
    }
}

/// Decode row `k` of a batch: greedy CTC (or the LM beam search for an unsure
/// line), confidence, and tokens at page x positions.
fn read_one(
    rec: &Recognizer,
    res: &RecBatch,
    k: usize,
    stride: f64,
    b: &LineBox,
    band_width: usize,
) -> LineRead {
    let t = res.t;
    let n = (res.lengths[k].max(0) as usize).min(t);
    let seq = &res.ids[k * t..k * t + n];
    let pm = &res.conf[k * t..k * t + n];
    let mut steps: Vec<usize> = Vec::new();
    let mut ids: Vec<i64> = Vec::new();
    let mut sum = 0f32;
    for i in 0..n {
        if seq[i] != 0 && (i == 0 || seq[i] != seq[i - 1]) {
            steps.push(i);
            ids.push(seq[i]);
            sum += pm[i];
        }
    }
    let conf = if ids.is_empty() {
        0.0
    } else {
        sum / ids.len() as f32
    };
    if let (Some((lm, p)), Some(o)) = (&rec.lm, &res.lm)
        && (MIN_CONF..p.gate).contains(&conf)
    {
        let params = BeamParams {
            alpha: p.alpha,
            beta: p.beta,
            beam: LM_BEAM,
            cand_floor: LM_CAND_FLOOR,
        };
        let kk = o.k;
        (ids, steps) = beam_search(
            &o.blank[k * t..k * t + n],
            &o.topv[k * t * kk..(k * t + n) * kk],
            &o.topi[k * t * kk..(k * t + n) * kk],
            kk,
            lm,
            params,
        );
    }
    if conf < MIN_CONF {
        return LineRead {
            text: String::new(),
            conf,
            toks: Vec::new(),
        };
    }
    // CTC step -> page x, through the crop's padding and scale.
    let h = (b.y1 - b.y0).max(1.0) as f64;
    let ix0 = (b.x0 as f64 - PAD_X as f64 * h).floor();
    let cw = (b.x1 as f64 + PAD_X as f64 * h).ceil() - ix0;
    let f = cw / band_width as f64;
    let toks = ids
        .iter()
        .zip(&steps)
        .filter_map(|(&id, &s)| {
            let x = (ix0 + (s as f64 + 0.5) * stride * f) as f32;
            let tok = if rec.as_chars {
                Tok::Char(rec.vocab.token(id)?.to_owned())
            } else {
                Tok::Id(id)
            };
            Some(Token { tok, x })
        })
        .collect();
    LineRead {
        text: rec.vocab.decode(&ids),
        conf,
        toks,
    }
}
