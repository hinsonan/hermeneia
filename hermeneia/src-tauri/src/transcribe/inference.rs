use crate::audio::decode_audio_file;
use crate::error::{AudioError, Result};
use crate::transcribe::{
    decoder::Decoder,
    language::detect_language,
    model::{get_device, ModelManager},
    preprocessing::preprocess_audio,
    types::{ModelFiles, ProgressCallback, TranscribeParams, TranscriptResult},
};
use candle_core::Device;
use candle_nn::VarBuilder;
use candle_transformers::models::whisper::{self as m, Config};
use std::time::Instant;
use tokenizers::Tokenizer;

/// Get a human-readable name for the device
fn device_name(device: &Device) -> &'static str {
    match device {
        Device::Cpu => "CPU",
        Device::Cuda(_) => "CUDA",
        Device::Metal(_) => "Metal",
    }
}

/// Main transcription function
pub fn transcribe_audio(file_path: &str, params: TranscribeParams) -> Result<TranscriptResult> {
    transcribe_audio_with_progress(file_path, params, None)
}

/// Main transcription function with progress callback
pub fn transcribe_audio_with_progress(
    file_path: &str,
    params: TranscribeParams,
    progress_callback: Option<ProgressCallback>,
) -> Result<TranscriptResult> {
    let start_time = Instant::now();

    // Load audio
    let audio_data = decode_audio_file(file_path)?;
    let duration = audio_data.duration_seconds();

    // Download/load model
    let model_manager = ModelManager::new()?;
    let model_files = model_manager
        .ensure_model(params.model, params.use_quantized)?;

    let device = get_device(params.force_cpu)?;
    tracing::info!("Using device: {}", device_name(&device));

    // Load model and tokenizer
    let (config, tokenizer, mut model) = load_model(&model_files, &device)?;

    // Preprocess to mel-spectrogram (needs config for mel bins)
    let mel = preprocess_audio(&audio_data, &config, &device)?;

    // Detect language if not specified and model is multilingual
    let language_token = match (params.model.is_multilingual(), &params.language) {
        (true, None) => {
            tracing::info!("Auto-detecting language...");
            Some(detect_language(&mut model, &tokenizer, &mel, &device)?)
        }
        (false, None) => None,
        (true, Some(lang)) => {
            let token = tokenizer
                .token_to_id(&format!("<|{lang}|>"))
                .ok_or_else(|| AudioError::TranscriptionFailed(format!("Language '{}' not supported", lang)))?;
            Some(token)
        }
        (false, Some(_)) => {
            return Err(AudioError::TranscriptionFailed(
                "Cannot set language for non-multilingual models".to_string(),
            ))
        }
    };

    // Run inference with full decoder
    let mut params_with_token = params.clone();
    params_with_token.language = None; // Clear language string, we'll use token directly
    let mut decoder = Decoder::new_with_language_token(
        &mut model,
        &tokenizer,
        &config,
        &device,
        &params_with_token,
        language_token,
    )?;
    let raw_segments = decoder.run(&mel, progress_callback)?;
    let segments = decoder.extract_segments(raw_segments);

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

