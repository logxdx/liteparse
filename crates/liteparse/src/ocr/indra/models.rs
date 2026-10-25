//! Model directory layout and `meta.json` parsing for Indra OCR.
//!
//! The minimum model set is `det.onnx`, `rec.onnx` and `meta.json`. Optional
//! files switch on later pipeline steps when present: `orient.onnx`
//! (orientation), `script.onnx` with `rec_<group>.onnx` (script routing to the
//! CJK, right-to-left and world recognizers), and `lm.npz` / `lm_<group>.npz`
//! (language-model beam search). Weights are never bundled: callers point at a
//! directory (for example a Hugging Face snapshot of `datalab-to/indra-ocr`).

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use serde::Deserialize;
use unicode_normalization::UnicodeNormalization;

use crate::error::LiteParseError;

/// Environment variable naming the default model directory.
pub const MODEL_DIR_ENV: &str = "INDRA_OCR_MODEL_DIR";

/// Recognizer groups in `script.onnx`'s class order, unless `meta.json` names
/// them (`script_groups`). `base` is `rec.onnx`.
const DEFAULT_GROUPS: [&str; 4] = ["base", "cjk", "rtl", "other"];

/// Beam-search settings of one recognizer's language model.
#[derive(Debug, Clone, Copy, Deserialize)]
pub struct LmParams {
    #[serde(default = "LmParams::default_alpha")]
    pub alpha: f64,
    #[serde(default = "LmParams::default_beta")]
    pub beta: f64,
    /// Lines under this confidence are re-read with the LM.
    #[serde(default = "LmParams::default_gate")]
    pub gate: f32,
}

impl LmParams {
    fn default_alpha() -> f64 {
        0.5
    }
    fn default_beta() -> f64 {
        1.5
    }
    fn default_gate() -> f32 {
        0.9
    }
}

impl Default for LmParams {
    fn default() -> Self {
        Self {
            alpha: Self::default_alpha(),
            beta: Self::default_beta(),
            gate: Self::default_gate(),
        }
    }
}

#[derive(Deserialize)]
struct RawMeta {
    det_long_side: u32,
    rec_height: u32,
    #[serde(default = "one")]
    rec_window: u32,
    vocab: String,
    script_groups: Option<Vec<String>>,
    #[serde(default)]
    lm_params: HashMap<String, LmParams>,
    #[serde(flatten)]
    rest: HashMap<String, serde_json::Value>,
}

fn one() -> u32 {
    1
}

/// Fields of `meta.json` the in-process engine reads.
pub struct Meta {
    pub det_long_side: u32,
    pub rec_height: u32,
    /// `rec.onnx` reads a window this many line heights tall (the line and the
    /// page around it); group recognizers read the line band only.
    pub rec_window: u32,
    pub vocab: Vocab,
    /// `script.onnx` class order.
    pub script_groups: Vec<String>,
    /// Vocabularies of the group recognizers (`vocab_<group>`).
    pub group_vocabs: HashMap<String, Vocab>,
    pub lm_params: HashMap<String, LmParams>,
}

impl std::fmt::Debug for Meta {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Meta")
            .field("det_long_side", &self.det_long_side)
            .field("rec_height", &self.rec_height)
            .field("rec_window", &self.rec_window)
            .field("vocab", &self.vocab.len())
            .field("script_groups", &self.script_groups)
            .finish_non_exhaustive()
    }
}

impl Meta {
    pub fn parse(bytes: &[u8]) -> Result<Self, LiteParseError> {
        let raw: RawMeta = serde_json::from_slice(bytes)
            .map_err(|e| LiteParseError::Ocr(format!("indra meta.json: {e}")))?;
        let mut group_vocabs = HashMap::new();
        for (key, value) in &raw.rest {
            if let (Some(group), Some(doc)) = (key.strip_prefix("vocab_"), value.as_str()) {
                group_vocabs.insert(group.to_owned(), Vocab::from_json(doc)?);
            }
        }
        Ok(Self {
            det_long_side: raw.det_long_side,
            rec_height: raw.rec_height,
            rec_window: raw.rec_window,
            vocab: Vocab::from_json(&raw.vocab)?,
            script_groups: raw
                .script_groups
                .unwrap_or_else(|| DEFAULT_GROUPS.map(String::from).to_vec()),
            group_vocabs,
            lm_params: raw.lm_params,
        })
    }
}

#[derive(Deserialize)]
struct VocabDoc {
    #[serde(default)]
    blank: usize,
    chars: Vec<String>,
}

/// Token table for CTC ids: class 0 is the blank, class `i` is `chars[i - 1]`.
/// A token is one character, or one LaTeX command (`\frac`) inside math.
#[derive(Debug)]
pub struct Vocab {
    chars: Vec<String>,
    space: Option<i64>,
    has_commands: bool,
    /// Hangul read as jamo: decoded text is NFC-composed into syllables.
    jamo: bool,
}

impl Vocab {
    pub fn new(chars: Vec<String>) -> Self {
        let space = chars.iter().position(|c| c == " ").map(|i| i as i64 + 1);
        let has_commands = chars.iter().any(|c| c.chars().count() > 1);
        let jamo = chars.iter().any(|c| c == "\u{1100}");
        Self {
            chars,
            space,
            has_commands,
            jamo,
        }
    }

    pub fn from_json(doc: &str) -> Result<Self, LiteParseError> {
        let doc: VocabDoc = serde_json::from_str(doc)
            .map_err(|e| LiteParseError::Ocr(format!("indra vocab: {e}")))?;
        if doc.blank != 0 {
            return Err(LiteParseError::Ocr(
                "indra vocab: only blank=0 vocabularies are supported".into(),
            ));
        }
        Ok(Self::new(doc.chars))
    }

    /// The token of class `id` (None for the blank and out-of-range ids).
    pub fn token(&self, id: i64) -> Option<&str> {
        let i = usize::try_from(id).ok()?.checked_sub(1)?;
        self.chars.get(i).map(String::as_str)
    }

    /// Class id of the space token.
    pub fn space(&self) -> Option<i64> {
        self.space
    }

    /// Text of class ids. A LaTeX command followed by a letter gets a space
    /// (`\alpha x`, not `\alphax`); jamo compose into syllables.
    pub fn decode(&self, ids: &[i64]) -> String {
        let toks: Vec<&str> = ids.iter().filter_map(|&i| self.token(i)).collect();
        if self.jamo {
            return toks.concat().nfc().collect();
        }
        if !self.has_commands {
            return toks.concat();
        }
        let mut out = String::new();
        for (k, t) in toks.iter().enumerate() {
            out.push_str(t);
            let next = toks.get(k + 1).and_then(|n| n.chars().next());
            let is_command =
                t.len() > 2 && t.starts_with('\\') && t[1..].chars().all(char::is_alphabetic);
            if is_command && next.is_some_and(char::is_alphabetic) {
                out.push(' ');
            }
        }
        out
    }

    pub fn len(&self) -> usize {
        self.chars.len()
    }

    pub fn is_empty(&self) -> bool {
        self.chars.is_empty()
    }
}

/// Resolved files of a model directory.
#[derive(Debug, Clone)]
pub struct ModelFiles {
    pub dir: PathBuf,
    pub det: PathBuf,
    pub rec: PathBuf,
    pub meta: PathBuf,
}

impl ModelFiles {
    pub fn in_dir(dir: &Path) -> Result<Self, LiteParseError> {
        let files = Self {
            dir: dir.to_path_buf(),
            det: dir.join("det.onnx"),
            rec: dir.join("rec.onnx"),
            meta: dir.join("meta.json"),
        };
        for path in [&files.det, &files.rec, &files.meta] {
            if !path.is_file() {
                return Err(LiteParseError::Ocr(format!(
                    "indra model file missing: {}",
                    path.display()
                )));
            }
        }
        Ok(files)
    }

    /// Directory from `INDRA_OCR_MODEL_DIR`.
    pub fn from_env() -> Result<Self, LiteParseError> {
        let dir = std::env::var_os(MODEL_DIR_ENV)
            .ok_or_else(|| LiteParseError::Ocr(format!("{MODEL_DIR_ENV} is not set")))?;
        Self::in_dir(Path::new(&dir))
    }

    /// An optional model file, when the directory has it.
    pub fn optional(&self, name: &str) -> Option<PathBuf> {
        let p = self.dir.join(name);
        p.is_file().then_some(p)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_meta_and_decodes() {
        let json = serde_json::json!({
            "det_long_side": 1024,
            "rec_height": 32,
            "rec_window": 3,
            "vocab": r#"{"blank": 0, "chars": ["a", "b", "c", " "]}"#,
            "vocab_cjk": r#"{"blank": 0, "chars": ["中"]}"#,
            "script_groups": ["base", "cjk"],
            "lm_params": {"cjk": {"alpha": 0.2, "gate": 0.95}},
            "extra": 1,
        });
        let meta = Meta::parse(json.to_string().as_bytes()).unwrap();
        assert_eq!(
            (meta.det_long_side, meta.rec_height, meta.rec_window),
            (1024, 32, 3)
        );
        assert_eq!(meta.vocab.len(), 4);
        assert_eq!(meta.vocab.decode(&[1, 0, 3, 2]), "acb");
        assert_eq!(meta.vocab.space(), Some(4));
        assert_eq!(meta.group_vocabs["cjk"].decode(&[1]), "中");
        assert_eq!(meta.script_groups, vec!["base", "cjk"]);
        let p = meta.lm_params["cjk"];
        assert_eq!((p.alpha, p.beta, p.gate), (0.2, 1.5, 0.95));
    }

    #[test]
    fn latex_commands_keep_a_space_before_letters() {
        let v = Vocab::new(["\\alpha", "x", "+"].map(String::from).to_vec());
        assert_eq!(v.decode(&[1, 2, 3, 1, 3]), "\\alpha x+\\alpha+");
    }

    #[test]
    fn jamo_compose() {
        let v = Vocab::new(["\u{1100}", "\u{1161}"].map(String::from).to_vec());
        assert_eq!(v.decode(&[1, 2]), "가");
    }

    #[test]
    fn missing_dir_errors() {
        assert!(ModelFiles::in_dir(Path::new("/nonexistent-indra")).is_err());
    }
}
