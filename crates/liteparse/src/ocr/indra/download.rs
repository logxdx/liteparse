//! Optional download of the Indra OCR models (`indra-ocr-auto-download`).
//!
//! Files come from the Hugging Face repo `datalab-to/indra-ocr` at a pinned
//! commit (tag `v1.1.0`, the model set this port was validated against), are
//! checked against fixed SHA-256 digests, and are cached under
//! `$INDRA_OCR_HOME`, else `$XDG_CACHE_HOME/liteparse/indra-ocr/<commit>`, else
//! `~/.cache/liteparse/indra-ocr/<commit>`.
//!
//! The weights are licensed under a modified OpenRAIL-M license, not Apache-2.0:
//! free for research, personal use and startups under $20M in funding or
//! revenue, not for use competing with Datalab's API; broader use needs a
//! Datalab license. Downloading is an explicit call, never a side effect.

use std::path::{Path, PathBuf};
use std::sync::Once;

use sha2::{Digest, Sha256};

use super::models::ModelFiles;
use super::{IndraOcrEngine, IndraOptions};
use crate::error::LiteParseError;

const REPO: &str = "datalab-to/indra-ocr";
/// Commit of tag `v1.1.0`.
pub const REVISION: &str = "bd02167fdf2005a19962cc15188f8496a2c87d3b";

/// (file, size in bytes, SHA-256) at [`REVISION`].
const FILES: [(&str, u64, &str); 11] = [
    (
        "det.onnx",
        389_948,
        "f2bd4cba23e9c328cf7ec824634e423e191a2c6aacbbf35c62bd79dd70f62f59",
    ),
    (
        "rec.onnx",
        5_042_161,
        "475972a2710be4a3cbfeb2bc7f3624ffd563de07de42476fc7a290d4e1ece484",
    ),
    (
        "meta.json",
        211_869,
        "4ce4c94b46f5d644033c02e718322014ff4081270f385038c571912b56696339",
    ),
    (
        "orient.onnx",
        371_802,
        "20eaf670f55a7cf9facfe1bdd6c7b8e40be75bbf4d48a3df43f66aa3bbd874fd",
    ),
    (
        "script.onnx",
        191_630,
        "b6ce4e926e19c243faaf37bd46880398bf2cb4ae56a29e1177a3b017c1280e36",
    ),
    (
        "rec_cjk.onnx",
        10_176_018,
        "4a035516a17e62cdf33a4fe2d88077654cb8841cff024685cdef4e351694ef50",
    ),
    (
        "rec_rtl.onnx",
        4_812_710,
        "468796bd684486a8388c30c8b4ed2a26e3eeb1f8ff02831c24c2e12c79a2a834",
    ),
    (
        "rec_world.onnx",
        5_627_990,
        "c8d1e59f21b04966f179d2d2d8e22b230d22110b3f89855bbbf2624a58c40043",
    ),
    (
        "lm.npz",
        15_695_957,
        "6c0fb618a363a0acdd05d4f5b0edd9e8146d2d990541b87f6130bbfe17cea58d",
    ),
    (
        "lm_cjk.npz",
        18_955_160,
        "6d64f13f4f90f4e529bd63340c6df76238d007feaef2abb903d8cd28d92bc2f3",
    ),
    (
        "lm_world.npz",
        11_641_215,
        "34df5e74a2fcbb9cf5e7f360a559375ffa2e04d2d2943a42308c177bcb444d5d",
    ),
];

/// Which files to fetch.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ModelSet {
    /// Detector, base recognizer, metadata and orientation model (about 6 MB):
    /// Latin, Greek, Cyrillic, Devanagari and Bengali, any page orientation.
    Base,
    /// Everything the engine uses (about 73 MB): adds script routing, the CJK,
    /// right-to-left and world recognizers, and the language models.
    Full,
}

impl ModelSet {
    fn files(self) -> &'static [(&'static str, u64, &'static str)] {
        match self {
            ModelSet::Base => &FILES[..4],
            ModelSet::Full => &FILES,
        }
    }
}

static LICENSE_NOTICE: Once = Once::new();

/// The cache directory for [`REVISION`].
pub fn cache_dir() -> Result<PathBuf, LiteParseError> {
    if let Some(home) = std::env::var_os("INDRA_OCR_HOME") {
        return Ok(PathBuf::from(home));
    }
    let base = std::env::var_os("XDG_CACHE_HOME")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".cache")))
        .ok_or_else(|| {
            LiteParseError::Ocr("indra: no cache directory (set INDRA_OCR_HOME)".into())
        })?;
    Ok(base.join("liteparse").join("indra-ocr").join(REVISION))
}

fn sha256_hex(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

/// Whether `path` holds the expected file (size first, digest only then).
fn verified(path: &Path, size: u64, sha: &str) -> bool {
    match std::fs::metadata(path) {
        Ok(m) if m.len() == size => std::fs::read(path).is_ok_and(|b| sha256_hex(&b) == sha),
        _ => false,
    }
}

/// Make sure `set` is in `dir` (default [`cache_dir`]), downloading and
/// verifying what is missing, and return the model files.
pub async fn ensure_models(
    set: ModelSet,
    dir: Option<&Path>,
) -> Result<ModelFiles, LiteParseError> {
    let dir = match dir {
        Some(d) => d.to_path_buf(),
        None => cache_dir()?,
    };
    tokio::fs::create_dir_all(&dir).await?;
    for &(name, size, sha) in set.files() {
        let path = dir.join(name);
        if verified(&path, size, sha) {
            continue;
        }
        LICENSE_NOTICE.call_once(|| {
            eprintln!(
                "[indra-ocr] downloading {REPO} models ({REVISION}). The weights are under a modified \
                 OpenRAIL-M license: free for research, personal use and startups under $20M funding \
                 or revenue; not for use competing with Datalab's API. See \
                 https://huggingface.co/{REPO} for the terms."
            );
        });
        let url = format!("https://huggingface.co/{REPO}/resolve/{REVISION}/{name}");
        let response = reqwest::get(&url)
            .await
            .map_err(|e| LiteParseError::Ocr(format!("indra download {name}: {e}")))?;
        if !response.status().is_success() {
            return Err(LiteParseError::Ocr(format!(
                "indra download {name}: HTTP {} from {url}",
                response.status()
            )));
        }
        let bytes = response
            .bytes()
            .await
            .map_err(|e| LiteParseError::Ocr(format!("indra download {name}: {e}")))?;
        if bytes.len() as u64 != size || sha256_hex(&bytes) != sha {
            return Err(LiteParseError::Ocr(format!(
                "indra download {name}: size or SHA-256 does not match revision {REVISION}"
            )));
        }
        // Temp file + rename: concurrent first use never sees a partial file.
        let tmp = dir.join(format!("{name}.tmp.{}", std::process::id()));
        tokio::fs::write(&tmp, &bytes).await?;
        tokio::fs::rename(&tmp, &path).await?;
    }
    ModelFiles::in_dir(&dir)
}

impl IndraOcrEngine {
    /// Download `set` into the cache (when not there yet) and load the engine.
    pub async fn download(set: ModelSet, options: IndraOptions) -> Result<Self, LiteParseError> {
        let files = ensure_models(set, None).await?;
        tokio::task::spawn_blocking(move || Self::new(&files, options))
            .await
            .map_err(|e| LiteParseError::Ocr(format!("indra: loading models: {e}")))?
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn base_set_is_the_minimum_files() {
        let names: Vec<&str> = ModelSet::Base.files().iter().map(|f| f.0).collect();
        assert_eq!(names, ["det.onnx", "rec.onnx", "meta.json", "orient.onnx"]);
        assert_eq!(ModelSet::Full.files().len(), 11);
    }

    #[test]
    fn verifies_size_and_digest() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("f");
        std::fs::write(&p, b"abc").unwrap();
        let sha = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad";
        assert!(verified(&p, 3, sha));
        assert!(!verified(&p, 4, sha));
        assert!(!verified(&p, 3, &sha.replace('b', "c")));
    }
}
