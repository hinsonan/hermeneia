use crate::audio::{
    decode_audio_file, decode_audio_file_with_progress, prepare_speech_audio_owned,
    DecodeProgressCallback, SpeechAudio,
};
use crate::error::{AudioError, Result};
use crate::runtime_cache::{
    global_runtime_cache, load_whisper_runtime_by_key, RuntimeCacheManager, WhisperRuntime,
    WhisperRuntimeKey,
};
use crate::transcribe::{
    decoder::Decoder,
    language::detect_language,
    model::{get_device, ModelManager},
    preprocessing::preprocess_speech_audio,
    types::{ModelFiles, ProgressCallback, ProgressReporter, TranscribeParams, TranscriptResult},
};
use candle_core::Device;
use candle_transformers::models::whisper::{self as m, Config};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
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

fn check_cancelled(cancel_flag: &Option<Arc<AtomicBool>>) -> Result<()> {
    if let Some(flag) = cancel_flag {
        if flag.load(Ordering::SeqCst) {
            return Err(AudioError::Cancelled);
        }
    }
    Ok(())
}

fn should_auto_detect_language(
    model: crate::transcribe::WhisperModel,
    language: Option<&str>,
) -> bool {
    model.is_multilingual() && language.is_none()
}

fn build_whisper_runtime_key(params: &TranscribeParams) -> WhisperRuntimeKey {
    WhisperRuntimeKey {
        model: params.model,
        force_cpu: params.force_cpu,
        use_quantized: params.use_quantized,
    }
}

fn load_whisper_runtime(params: &TranscribeParams) -> Result<WhisperRuntime> {
    let key = build_whisper_runtime_key(params);
    let runtime =
        load_whisper_runtime_by_key(key).map_err(|e| enrich_oom_error(e, params.model))?;
    tracing::info!("Using device: {}", device_name(&runtime.device));
    Ok(runtime)
}

fn run_with_whisper_runtime<P: ProgressReporter>(
    runtime: &mut WhisperRuntime,
    speech_audio: &SpeechAudio,
    params: &TranscribeParams,
    reporter: &P,
    cancel_flag: Option<Arc<AtomicBool>>,
) -> Result<(Vec<crate::transcribe::TranscriptSegment>, String)> {
    check_cancelled(&cancel_flag)?;

    let mel = preprocess_speech_audio(speech_audio, &runtime.config, &runtime.device)?;
    check_cancelled(&cancel_flag)?;

    let language_token = if should_auto_detect_language(params.model, params.language.as_deref()) {
        tracing::info!("Auto-detecting language...");
        Some(detect_language(
            &mut runtime.model,
            &runtime.tokenizer,
            &mel,
            &runtime.device,
        )?)
    } else {
        match (params.model.is_multilingual(), &params.language) {
            (false, None) => None,
            (true, Some(lang)) => {
                let token = runtime
                    .tokenizer
                    .token_to_id(&format!("<|{lang}|>"))
                    .ok_or_else(|| {
                        AudioError::TranscriptionFailed(format!(
                            "Language '{}' not supported",
                            lang
                        ))
                    })?;
                Some(token)
            }
            (false, Some(lang)) => {
                tracing::warn!(
                "Ignoring language '{}' for English-only model; these models only support English",
                lang
            );
                None
            }
            (true, None) => None,
        }
    };

    let reporter_ptr = reporter as *const P as usize;
    let callback: ProgressCallback = Box::new(move |current, total| unsafe {
        let reporter_ref = &*(reporter_ptr as *const P);
        reporter_ref.report(current, total);
    });

    let mut decoder = Decoder::new_with_language_token(
        &mut runtime.model,
        &runtime.tokenizer,
        &runtime.config,
        &runtime.device,
        params.task,
        params.timestamps,
        language_token,
    )?;

    let raw_segments = decoder.run(&mel, Some(callback), cancel_flag)?;
    let segments = decoder.extract_segments(raw_segments);

    let text = join_segments_with_space(&segments);

    crate::gpu_cleanup::synchronize_device(&runtime.device);

    Ok((segments, text))
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
    let audio_data = decode_audio_file(file_path)?;
    let speech_audio = prepare_speech_audio_owned(audio_data)?;
    transcribe_prepared_audio_with_progress(&speech_audio, params, progress_callback)
}

/// Transcribe already-preprocessed mono 16kHz speech audio with progress callback.
pub fn transcribe_prepared_audio_with_progress(
    speech_audio: &SpeechAudio,
    params: TranscribeParams,
    progress_callback: Option<ProgressCallback>,
) -> Result<TranscriptResult> {
    let start_time = Instant::now();
    let duration = speech_audio.duration_seconds;

    // Download/load model
    let model_manager = ModelManager::new()?;
    let model_files = model_manager.ensure_model(params.model, params.use_quantized)?;

    let device = get_device(params.force_cpu)?;
    tracing::info!("Using device: {}", device_name(&device));

    // Scope model lifetime so GPU memory is freed before building result
    let (segments, text) = {
        // Load model and tokenizer
        let (config, tokenizer, mut model) =
            load_model(&model_files, &device).map_err(|e| enrich_oom_error(e, params.model))?;

        // Preprocess to mel-spectrogram (needs config for mel bins)
        let mel = preprocess_speech_audio(speech_audio, &config, &device)?;

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
                    .ok_or_else(|| {
                        AudioError::TranscriptionFailed(format!(
                            "Language '{}' not supported",
                            lang
                        ))
                    })?;
                Some(token)
            }
            (false, Some(lang)) => {
                // English-only models don't support language selection - ignore and continue
                tracing::warn!(
                    "Ignoring language '{}' for English-only model; these models only support English",
                    lang
                );
                None
            }
        };

        // Run inference with full decoder
        let mut decoder = Decoder::new_with_language_token(
            &mut model,
            &tokenizer,
            &config,
            &device,
            params.task,
            params.timestamps,
            language_token,
        )?;
        let raw_segments = decoder.run(&mel, progress_callback, None)?;

        // Debug logging
        tracing::info!("Raw segments count: {}", raw_segments.len());
        for (i, seg) in raw_segments.iter().enumerate() {
            tracing::info!(
                "Raw segment {}: start={:.2}s, text='{}', tokens={:?}",
                i,
                seg.start,
                seg.dr.text,
                seg.dr.tokens
            );
        }

        let segments = decoder.extract_segments(raw_segments);

        tracing::info!("Extracted segments count: {}", segments.len());
        for seg in &segments {
            tracing::info!("Extracted segment {}: text='{}'", seg.id, seg.text);
        }

        let text = join_segments_with_space(&segments);

        // Sync GPU before model/mel/decoder drop at end of scope
        crate::gpu_cleanup::synchronize_device(&device);
        tracing::info!("Model resources released from {}", device_name(&device));

        (segments, text)
    };

    Ok(TranscriptResult {
        segments,
        text,
        language: params.language.clone(),
        duration,
        model: params.model,
        inference_time: start_time.elapsed().as_secs_f64(),
    })
}

fn join_segments_with_space(segments: &[crate::transcribe::TranscriptSegment]) -> String {
    let mut text = String::new();
    for (idx, segment) in segments.iter().enumerate() {
        if idx > 0 {
            text.push(' ');
        }
        text.push_str(segment.text.as_str());
    }
    text
}

/// Main transcription function with progress reporter trait
pub fn transcribe_audio_with_reporter<P: ProgressReporter>(
    file_path: &str,
    params: TranscribeParams,
    reporter: &P,
    cancel_flag: Option<Arc<AtomicBool>>,
) -> Result<TranscriptResult> {
    check_cancelled(&cancel_flag)?;

    let reporter_ptr = reporter as *const P as usize;
    let cancel_for_decode = cancel_flag.clone();
    let decode_progress: DecodeProgressCallback = Box::new(move |current, total| unsafe {
        let reporter_ref = &*(reporter_ptr as *const P);
        reporter_ref.report(current, total);

        !cancel_for_decode
            .as_ref()
            .map(|flag| flag.load(Ordering::SeqCst))
            .unwrap_or(false)
    });

    let audio_data = decode_audio_file_with_progress(file_path, Some(decode_progress))?;
    check_cancelled(&cancel_flag)?;

    let speech_audio = prepare_speech_audio_owned(audio_data)?;
    check_cancelled(&cancel_flag)?;

    transcribe_prepared_audio_with_reporter(&speech_audio, params, reporter, cancel_flag)
}

/// Transcribe already-preprocessed mono 16kHz speech audio with progress reporter.
pub fn transcribe_prepared_audio_with_reporter<P: ProgressReporter>(
    speech_audio: &SpeechAudio,
    params: TranscribeParams,
    reporter: &P,
    cancel_flag: Option<Arc<AtomicBool>>,
) -> Result<TranscriptResult> {
    transcribe_prepared_audio_with_reporter_cached(
        speech_audio,
        params,
        reporter,
        cancel_flag,
        Some(global_runtime_cache()),
    )
}

pub fn transcribe_prepared_audio_with_reporter_cached<P: ProgressReporter>(
    speech_audio: &SpeechAudio,
    params: TranscribeParams,
    reporter: &P,
    cancel_flag: Option<Arc<AtomicBool>>,
    runtime_cache: Option<Arc<RuntimeCacheManager>>,
) -> Result<TranscriptResult> {
    let start_time = Instant::now();
    let duration = speech_audio.duration_seconds;

    check_cancelled(&cancel_flag)?;

    let (segments, text) = if let Some(cache) = runtime_cache {
        let key = build_whisper_runtime_key(&params);

        cache.with_whisper_runtime_cancellable(
            key,
            || load_whisper_runtime(&params),
            |runtime| {
                run_with_whisper_runtime(
                    runtime,
                    speech_audio,
                    &params,
                    reporter,
                    cancel_flag.clone(),
                )
            },
            cancel_flag.as_deref(),
        )?
    } else {
        let mut runtime = load_whisper_runtime(&params)?;
        run_with_whisper_runtime(
            &mut runtime,
            speech_audio,
            &params,
            reporter,
            cancel_flag.clone(),
        )?
    };

    reporter.finish();

    Ok(TranscriptResult {
        segments,
        text,
        language: params.language.clone(),
        duration,
        model: params.model,
        inference_time: start_time.elapsed().as_secs_f64(),
    })
}

/// Enrich OOM errors with model-specific information
fn enrich_oom_error(error: AudioError, model: crate::transcribe::WhisperModel) -> AudioError {
    match error {
        AudioError::OutOfMemory {
            message, device, ..
        } => {
            let reqs = model.requirements();
            let required_gb = if device == "VRAM" {
                reqs.min_vram_gb
            } else {
                reqs.min_ram_gb
            };

            AudioError::OutOfMemory {
                message: format!(
                    "{}. This model requires at least {:.1}GB of {}. Try using 'tiny' or 'base' model instead.",
                    message, required_gb, device
                ),
                device,
                required_gb,
                model_name: model.model_id().to_string(),
            }
        }
        other => other,
    }
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

    let config_str = std::fs::read_to_string(&files.config).map_err(|e| AudioError::ModelLoad {
        model: "config".to_string(),
        details: e.to_string(),
    })?;
    let config: Config = serde_json::from_str(&config_str).map_err(|e| AudioError::ModelLoad {
        model: "config".to_string(),
        details: e.to_string(),
    })?;

    let tokenizer = Tokenizer::from_file(&files.tokenizer).map_err(|e| AudioError::ModelLoad {
        model: "tokenizer".to_string(),
        details: e.to_string(),
    })?;

    let vb = crate::gpu_cleanup::load_safetensors_varbuilder(&files.weights, m::DTYPE, device)
        .map_err(|e| crate::gpu_cleanup::to_model_load_error(e, device, "weights"))?;

    let model = m::model::Whisper::load(&vb, config.clone())
        .map_err(|e| crate::gpu_cleanup::to_model_init_error(e, device, "whisper"))?;

    Ok((config, tokenizer, model))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transcribe::{TranscribeParams, TranscriptionTask, WhisperModel};

    #[test]
    fn test_transcribe_uses_cache_key_fields() {
        let params = TranscribeParams {
            model: WhisperModel::Small,
            task: TranscriptionTask::Transcribe,
            language: Some("en".to_string()),
            timestamps: true,
            force_cpu: true,
            use_quantized: false,
        };

        let key = build_whisper_runtime_key(&params);
        assert_eq!(key.model, WhisperModel::Small);
        assert!(key.force_cpu);
        assert!(!key.use_quantized);
    }

    #[test]
    fn test_language_auto_detect_branching_logic() {
        assert!(should_auto_detect_language(WhisperModel::Small, None));
        assert!(!should_auto_detect_language(
            WhisperModel::Small,
            Some("en")
        ));
        assert!(!should_auto_detect_language(WhisperModel::SmallEn, None));
    }
}
