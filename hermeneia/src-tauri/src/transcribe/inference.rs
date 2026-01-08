use crate::audio::decode_audio_file;
use crate::error::{AudioError, Result};
use crate::transcribe::{
    model::{get_device, ModelManager},
    preprocessing::preprocess_audio,
    types::{ModelFiles, TranscribeParams, TranscriptResult, TranscriptSegment},
};
use candle_core::Device;
use candle_nn::VarBuilder;
use candle_transformers::models::whisper::{self as m, Config};
use std::time::Instant;
use tokenizers::Tokenizer;

/// Main transcription function
pub fn transcribe_audio(file_path: &str, params: TranscribeParams) -> Result<TranscriptResult> {
    let start_time = Instant::now();

    // Load audio
    let audio_data = decode_audio_file(file_path)?;
    let duration = audio_data.duration_seconds();

    // Preprocess to mel-spectrogram
    let mel = preprocess_audio(&audio_data)?;

    // Download/load model
    let model_manager = ModelManager::new()?;
    let model_files = model_manager
        .ensure_model(params.model, params.use_quantized)?;

    let device = get_device(params.force_cpu)?;

    // Load model and tokenizer
    let (config, tokenizer, mut model) = load_model(&model_files, &device)?;

    // Run inference - create a minimal decoder
    let segments = run_inference_minimal(&mut model, &tokenizer, mel, &params, &device, &config)?;

    // Build result
    let text = segments
        .iter()
        .map(|s| s.text.as_str())
        .collect::<Vec<_>>()
        .join(" ");

    Ok(TranscriptResult {
        segments,
        text,
        language: params.language.clone(),
        duration,
        model: params.model,
        inference_time: start_time.elapsed().as_secs_f64(),
    })
}

/// Load config, tokenizer, and model
fn load_model(
    files: &ModelFiles,
    device: &Device,
) -> Result<(Config, Tokenizer, m::model::Whisper)> {
    if files.is_quantized {
        return Err(AudioError::ModelLoad {
            model: "quantized".to_string(),
            details: "Quantized models not yet supported".to_string(),
        });
    }

    // Load config
    let config_str = std::fs::read_to_string(&files.config).map_err(|e| {
        AudioError::ModelLoad {
            model: "config".to_string(),
            details: e.to_string(),
        }
    })?;
    let config: Config = serde_json::from_str(&config_str).map_err(|e| {
        AudioError::ModelLoad {
            model: "config".to_string(),
            details: e.to_string(),
        }
    })?;

    // Load tokenizer
    let tokenizer = Tokenizer::from_file(&files.tokenizer).map_err(|e| {
        AudioError::ModelLoad {
            model: "tokenizer".to_string(),
            details: e.to_string(),
        }
    })?;

    // Load model weights
    let vb = unsafe {
        VarBuilder::from_mmaped_safetensors(&[files.weights.clone()], m::DTYPE, device)
            .map_err(|e| AudioError::ModelLoad {
                model: "weights".to_string(),
                details: e.to_string(),
            })?
    };

    let model = m::model::Whisper::load(&vb, config.clone()).map_err(|e| {
        AudioError::ModelLoad {
            model: "model".to_string(),
            details: e.to_string(),
        }
    })?;

    Ok((config, tokenizer, model))
}

/// Minimal inference - just get the transcription text
fn run_inference_minimal(
    model: &mut m::model::Whisper,
    _tokenizer: &Tokenizer,
    mel: candle_core::Tensor,
    _params: &TranscribeParams,
    _device: &Device,
    _config: &Config,
) -> Result<Vec<TranscriptSegment>> {
    // Encode audio features
    let _audio_features = model
        .encoder
        .forward(&mel, true)
        .map_err(|e| AudioError::TranscriptionFailed(format!("Encoder failed: {}", e)))?;

    // For simplicity, just return a single segment with placeholder text
    // A full implementation would use the decoder to generate tokens and convert to text
    // This is a minimal stub to get compilation working

    Ok(vec![TranscriptSegment {
        id: 0,
        start: Some(0.0),
        end: Some(0.0),
        text: "[Transcription stub - decoder not yet implemented]".to_string(),
    }])
}
