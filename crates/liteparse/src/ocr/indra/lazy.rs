//! The process-wide Indra engine behind the `indra` OCR backend
//! ([`crate::config::OcrBackend::Indra`]). It loads on the first page that needs OCR, so
//! documents without any never load ONNX Runtime or the models.
//!
//! Models: the directory given to [`set_model_dir`] (the CLI's `--indra-model-dir`), else
//! `INDRA_OCR_MODEL_DIR`, else (with `indra-ocr-auto-download`) the SHA-256-checked download
//! cache. ONNX Runtime: shared with ocria-nano (see `environment` in the parent module).

use std::future::Future;
use std::path::PathBuf;
use std::pin::Pin;
use std::sync::{Arc, OnceLock};

use super::models::{MODEL_DIR_ENV, ModelFiles};
use super::{IndraOcrEngine, IndraOptions};
use crate::error::LiteParseError;
use crate::ocr::{OcrEngine, OcrOptions, OcrResult};

static MODEL_DIR: OnceLock<PathBuf> = OnceLock::new();

/// Read the models from `dir`. Takes effect when called before the engine first loads; the
/// first call wins.
pub fn set_model_dir(dir: PathBuf) {
    let _ = MODEL_DIR.set(dir);
}

fn model_files() -> Result<ModelFiles, LiteParseError> {
    let dir = MODEL_DIR
        .get()
        .cloned()
        .or_else(|| std::env::var_os(MODEL_DIR_ENV).map(PathBuf::from));
    if let Some(dir) = dir {
        return ModelFiles::in_dir(&dir);
    }
    #[cfg(feature = "indra-ocr-auto-download")]
    {
        // The caller may be on a tokio runtime, which cannot block on another one: download on
        // a thread of its own.
        std::thread::spawn(|| {
            tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .map_err(|e| LiteParseError::Ocr(format!("indra: {e}")))?
                .block_on(super::download::ensure_models(
                    super::download::ModelSet::Full,
                    None,
                ))
        })
        .join()
        .map_err(|_| LiteParseError::Ocr("indra: the model download panicked".into()))?
    }
    #[cfg(not(feature = "indra-ocr-auto-download"))]
    Err(LiteParseError::Ocr(format!(
        "indra: no models: set {MODEL_DIR_ENV} (or build with the `indra-ocr-auto-download` feature)"
    )))
}

/// The process-wide engine, loaded on first use with the default [`IndraOptions`].
pub fn shared_engine() -> Result<Arc<IndraOcrEngine>, LiteParseError> {
    static ENGINE: OnceLock<Result<Arc<IndraOcrEngine>, String>> = OnceLock::new();
    ENGINE
        .get_or_init(|| {
            let files = model_files().map_err(|e| e.to_string())?;
            IndraOcrEngine::new(&files, IndraOptions::default())
                .map(Arc::new)
                .map_err(|e| e.to_string())
        })
        .clone()
        .map_err(LiteParseError::Ocr)
}

/// [`shared_engine`] as an [`OcrEngine`] that loads on its first page.
pub struct LazyEngine;

impl OcrEngine for LazyEngine {
    fn name(&self) -> &str {
        "indra-ocr"
    }

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
                + 'a,
        >,
    > {
        Box::pin(async move {
            let engine = shared_engine()
                .map_err(|e| Box::new(e) as Box<dyn std::error::Error + Send + Sync>)?;
            engine.recognize(image_data, width, height, options).await
        })
    }
}
