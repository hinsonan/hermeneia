use crate::audio::{
    decode_audio_file_with_progress, prepare_speech_audio_owned, DecodeProgressCallback,
    SpeechAudio,
};
use crate::error::{AudioError, Result};
use crate::runtime_cache::{
    global_runtime_cache, load_whisper_runtime_by_key, RuntimeCacheManager, WhisperRuntime,
    WhisperRuntimeKey,
};
use crate::transcribe::{
    decoder::Decoder,
    language::detect_language,
    preprocessing::preprocess_speech_audio,
    types::{ProgressCallback, ProgressReporter, TranscribeParams, TranscriptResult},
};
use candle_core::Device;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Instant;

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
    }
}

fn load_whisper_runtime(params: &TranscribeParams) -> Result<WhisperRuntime> {
    let key = build_whisper_runtime_key(params);
    let runtime =
        load_whisper_runtime_by_key(key).map_err(|e| enrich_oom_error(e, params.model))?;
    tracing::info!("Using device: {}", device_name(&runtime.device));
    Ok(runtime)
}

fn run_with_whisper_runtime<P: ProgressReporter + 'static>(
    runtime: &mut WhisperRuntime,
    speech_audio: &SpeechAudio,
    params: &TranscribeParams,
    reporter: Arc<P>,
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

    let reporter_for_callback = Arc::clone(&reporter);
    let callback: ProgressCallback = Box::new(move |current, total| {
        reporter_for_callback.report(current, total);
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
pub fn transcribe_audio_with_reporter<P: ProgressReporter + 'static>(
    file_path: &str,
    params: TranscribeParams,
    reporter: Arc<P>,
    cancel_flag: Option<Arc<AtomicBool>>,
) -> Result<TranscriptResult> {
    check_cancelled(&cancel_flag)?;

    let reporter_for_decode = Arc::clone(&reporter);
    let cancel_for_decode = cancel_flag.clone();
    let decode_progress: DecodeProgressCallback = Box::new(move |current, total| {
        reporter_for_decode.report(current, total);

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
pub fn transcribe_prepared_audio_with_reporter<P: ProgressReporter + 'static>(
    speech_audio: &SpeechAudio,
    params: TranscribeParams,
    reporter: Arc<P>,
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

pub fn transcribe_prepared_audio_with_reporter_cached<P: ProgressReporter + 'static>(
    speech_audio: &SpeechAudio,
    params: TranscribeParams,
    reporter: Arc<P>,
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
                    Arc::clone(&reporter),
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
            Arc::clone(&reporter),
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
        };

        let key = build_whisper_runtime_key(&params);
        assert_eq!(key.model, WhisperModel::Small);
        assert!(key.force_cpu);
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
