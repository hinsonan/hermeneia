use crate::audio::{decode_audio_file, prepare_speech_audio, SpeechAudio};
use crate::error::{AudioError, Result};
use crate::runtime_cache::{
    global_runtime_cache, RuntimeCacheManager, SpeakerRuntime, SpeakerRuntimeKey,
};
use crate::speaker::{
    model::SpeakerModelManager,
    types::{validate_diarize_params, DiarizationResult, DiarizeParams, SpeakerSegment},
};
use sherpa_rs::diarize::{Diarize, DiarizeConfig};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Instant;

const CUDA_WARMUP_DURATION_SEC: f32 = 2.0;
const SPEECH_SAMPLE_RATE: usize = 16_000;

fn should_attempt_warmup(
    warmed_up: bool,
    device: &crate::speaker::SpeakerDevice,
    cancel: &Option<Arc<AtomicBool>>,
) -> bool {
    if warmed_up {
        return false;
    }
    if !matches!(device, crate::speaker::SpeakerDevice::Cuda) {
        return false;
    }
    !cancel
        .as_ref()
        .map(|c| c.load(Ordering::SeqCst))
        .unwrap_or(false)
}

fn warmup_sample_count(total_samples: usize) -> usize {
    let warmup_samples = (CUDA_WARMUP_DURATION_SEC * SPEECH_SAMPLE_RATE as f32) as usize;
    warmup_samples.min(total_samples)
}

/// Progress callback: (num_processed_chunks, num_total_chunks)
pub type DiarizeProgressCallback = Box<dyn Fn(i32, i32) + Send + Sync>;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DiarizeStage {
    EnsuringModels,
    InitializingRuntime,
    Diarizing,
    Finalizing,
}

#[derive(Debug, Clone)]
pub struct DiarizeStageProgress {
    pub stage: DiarizeStage,
    pub current: Option<usize>,
    pub total: Option<usize>,
    pub message: &'static str,
    pub indeterminate: bool,
}

pub type DiarizeStageProgressCallback = Arc<dyn Fn(DiarizeStageProgress) + Send + Sync>;

#[derive(Default)]
pub struct DiarizeCallbacks {
    pub chunk_progress: Option<DiarizeProgressCallback>,
    pub stage_progress: Option<DiarizeStageProgressCallback>,
}

fn check_cancelled(cancel: &Option<Arc<AtomicBool>>) -> Result<()> {
    if cancel
        .as_ref()
        .map(|c| c.load(Ordering::SeqCst))
        .unwrap_or(false)
    {
        return Err(AudioError::Cancelled);
    }
    Ok(())
}

/// Diarize an audio file. Downloads models on first use.
pub fn diarize_audio(audio_path: &str, params: DiarizeParams) -> Result<DiarizationResult> {
    diarize_audio_with_progress(audio_path, params, None, None)
}

/// Diarize with optional progress callback and cancellation flag.
pub fn diarize_audio_with_progress(
    audio_path: &str,
    params: DiarizeParams,
    progress: Option<DiarizeProgressCallback>,
    cancel: Option<Arc<AtomicBool>>,
) -> Result<DiarizationResult> {
    validate_diarize_params(&params)?;
    check_cancelled(&cancel)?;

    tracing::info!("Decoding audio: {}", audio_path);
    let audio = decode_audio_file(audio_path)?;
    check_cancelled(&cancel)?;

    let speech_audio = prepare_speech_audio(&audio)?;
    drop(audio);
    check_cancelled(&cancel)?;

    diarize_prepared_audio_with_progress(&speech_audio, params, progress, cancel)
}

/// Diarize already-preprocessed mono 16kHz speech audio.
pub fn diarize_prepared_audio(
    speech_audio: &SpeechAudio,
    params: DiarizeParams,
) -> Result<DiarizationResult> {
    diarize_prepared_audio_with_progress(speech_audio, params, None, None)
}

fn emit_stage(
    stage_progress: &Option<DiarizeStageProgressCallback>,
    stage: DiarizeStage,
    current: Option<usize>,
    total: Option<usize>,
    message: &'static str,
    indeterminate: bool,
) {
    if let Some(cb) = stage_progress {
        cb(DiarizeStageProgress {
            stage,
            current,
            total,
            message,
            indeterminate,
        });
    }
}

fn ensure_model_paths(
    params: &DiarizeParams,
    cancel: &Option<Arc<AtomicBool>>,
    stage_progress: &Option<DiarizeStageProgressCallback>,
) -> Result<(PathBuf, PathBuf)> {
    check_cancelled(cancel)?;
    emit_stage(
        stage_progress,
        DiarizeStage::EnsuringModels,
        None,
        None,
        "Ensuring speaker diarization models...",
        true,
    );

    tracing::info!("Ensuring speaker diarization models...");
    let paths = SpeakerModelManager::ensure_models(&params.model)?;

    check_cancelled(cancel)?;
    Ok(paths)
}

fn create_diarizer(
    seg_path: &PathBuf,
    emb_path: &PathBuf,
    params: &DiarizeParams,
    stage_progress: &Option<DiarizeStageProgressCallback>,
) -> Result<Diarize> {
    emit_stage(
        stage_progress,
        DiarizeStage::InitializingRuntime,
        None,
        None,
        "Initializing diarization runtime...",
        true,
    );

    let config = DiarizeConfig {
        num_clusters: params.num_speakers,
        threshold: Some(params.threshold),
        provider: Some(params.device.provider_string().to_string()),
        ..Default::default()
    };

    Diarize::new(seg_path, emb_path, config)
        .map_err(|e| AudioError::DiarizationFailed(e.to_string()))
}

fn load_speaker_runtime(
    params: &DiarizeParams,
    cancel: &Option<Arc<AtomicBool>>,
    stage_progress: &Option<DiarizeStageProgressCallback>,
) -> Result<crate::runtime_cache::SpeakerRuntime> {
    let (seg_path, emb_path) = ensure_model_paths(params, cancel, stage_progress)?;
    check_cancelled(cancel)?;

    let diarize = create_diarizer(&seg_path, &emb_path, params, stage_progress)?;

    Ok(crate::runtime_cache::SpeakerRuntime {
        diarize,
        provider: params.device.provider_string().to_string(),
        warmed_up: false,
    })
}

fn maybe_warmup_speaker_runtime(
    runtime: &mut SpeakerRuntime,
    device: &crate::speaker::SpeakerDevice,
    speech_audio: &SpeechAudio,
    cancel: &Option<Arc<AtomicBool>>,
) {
    if !should_attempt_warmup(runtime.warmed_up, device, cancel) {
        if !matches!(device, crate::speaker::SpeakerDevice::Cuda) {
            runtime.warmed_up = true;
        }
        return;
    }

    let samples_len = warmup_sample_count(speech_audio.samples_16k_mono.len());
    if samples_len < SPEECH_SAMPLE_RATE / 2 {
        runtime.warmed_up = true;
        return;
    }
    let samples = speech_audio.samples_16k_mono[..samples_len].to_vec();

    tracing::info!(
        provider = %runtime.provider,
        samples = samples.len(),
        "Warming up cached speaker runtime"
    );

    let warmup_start = Instant::now();
    let result = runtime.diarize.compute(samples, None);
    let warmup_ms = warmup_start.elapsed().as_millis();

    match result {
        Ok(_) => {
            tracing::info!(warmup_ms, "Speaker runtime warmup completed");
        }
        Err(e) => {
            tracing::warn!(
                warmup_ms,
                error = %e,
                "Speaker runtime warmup failed; continuing without warmup"
            );
        }
    }

    // Mark attempted regardless of outcome so we only pay this once.
    runtime.warmed_up = true;
}

fn compute_speaker_segments(
    sd: &mut Diarize,
    speech_audio: &SpeechAudio,
    chunk_progress: Option<DiarizeProgressCallback>,
    stage_progress: Option<DiarizeStageProgressCallback>,
    cancel: Option<Arc<AtomicBool>>,
) -> Result<Vec<SpeakerSegment>> {
    emit_stage(
        &stage_progress,
        DiarizeStage::Diarizing,
        None,
        None,
        "Diarizing...",
        true,
    );

    let chunk_progress = chunk_progress.map(Arc::new);
    let last_percent = Arc::new(AtomicUsize::new(usize::MAX));

    let needs_callback = chunk_progress.is_some() || stage_progress.is_some();
    let sherpa_callback = if needs_callback {
        let cancel_for_callback = cancel.clone();
        let chunk_progress_for_callback = chunk_progress.clone();
        let stage_progress_for_callback = stage_progress.clone();
        let last_percent_for_callback = last_percent.clone();

        Some(Box::new(move |processed: i32, total: i32| -> i32 {
            if cancel_for_callback
                .as_ref()
                .map(|c| c.load(Ordering::SeqCst))
                .unwrap_or(false)
            {
                return 1;
            }

            if let Some(cb) = &chunk_progress_for_callback {
                cb(processed, total);
            }

            if let Some(stage_cb) = &stage_progress_for_callback {
                if processed >= 0 && total > 0 {
                    let processed_u = processed as usize;
                    let total_u = total as usize;
                    let percent = processed_u.saturating_mul(100) / total_u.max(1);
                    if last_percent_for_callback.swap(percent, Ordering::Relaxed) != percent {
                        stage_cb(DiarizeStageProgress {
                            stage: DiarizeStage::Diarizing,
                            current: Some(processed_u),
                            total: Some(total_u),
                            message: "Diarizing...",
                            indeterminate: false,
                        });
                    }
                }
            }

            0
        }) as Box<dyn Fn(i32, i32) -> i32 + Send + 'static>)
    } else {
        None
    };

    // sherpa-rs consumes Vec<f32>, so clone when using shared prepared audio.
    let samples_16k = speech_audio.samples_16k_mono.clone();

    let segments = sd.compute(samples_16k, sherpa_callback).map_err(|e| {
        if cancel
            .as_ref()
            .map(|c| c.load(Ordering::SeqCst))
            .unwrap_or(false)
        {
            AudioError::Cancelled
        } else {
            AudioError::DiarizationFailed(e.to_string())
        }
    })?;

    let result_segments: Vec<SpeakerSegment> = segments
        .iter()
        .map(|s| SpeakerSegment {
            speaker: s.speaker,
            start: s.start,
            end: s.end,
        })
        .collect();

    Ok(result_segments)
}

fn count_unique_speakers(segments: &[SpeakerSegment]) -> usize {
    let mut ids: Vec<i32> = segments.iter().map(|s| s.speaker).collect();
    ids.sort_unstable();
    ids.dedup();
    ids.len()
}

/// Diarize already-preprocessed mono 16kHz speech audio with progress/cancel.
pub fn diarize_prepared_audio_with_progress(
    speech_audio: &SpeechAudio,
    params: DiarizeParams,
    progress: Option<DiarizeProgressCallback>,
    cancel: Option<Arc<AtomicBool>>,
) -> Result<DiarizationResult> {
    diarize_prepared_audio_with_callbacks(
        speech_audio,
        params,
        DiarizeCallbacks {
            chunk_progress: progress,
            stage_progress: None,
        },
        cancel,
    )
}

/// Diarize with both chunk-level and stage-level progress callbacks.
pub fn diarize_prepared_audio_with_callbacks(
    speech_audio: &SpeechAudio,
    params: DiarizeParams,
    callbacks: DiarizeCallbacks,
    cancel: Option<Arc<AtomicBool>>,
) -> Result<DiarizationResult> {
    diarize_prepared_audio_with_callbacks_cached(
        speech_audio,
        params,
        callbacks,
        cancel,
        Some(global_runtime_cache()),
    )
}

pub fn diarize_prepared_audio_with_callbacks_cached(
    speech_audio: &SpeechAudio,
    params: DiarizeParams,
    callbacks: DiarizeCallbacks,
    cancel: Option<Arc<AtomicBool>>,
    runtime_cache: Option<Arc<RuntimeCacheManager>>,
) -> Result<DiarizationResult> {
    let start = Instant::now();
    validate_diarize_params(&params)?;
    check_cancelled(&cancel)?;

    let device_name = params.device.provider_string().to_string();
    let model_name = params.model.display_name().to_string();

    tracing::info!(
        "Running diarization with {} model on {}...",
        model_name,
        device_name
    );

    let result_segments = if let Some(cache) = runtime_cache {
        let key = SpeakerRuntimeKey {
            model: params.model.clone(),
            device: params.device.clone(),
        };

        cache.with_speaker_runtime(
            key,
            || load_speaker_runtime(&params, &cancel, &callbacks.stage_progress),
            |runtime| {
                maybe_warmup_speaker_runtime(runtime, &params.device, speech_audio, &cancel);
                compute_speaker_segments(
                    &mut runtime.diarize,
                    speech_audio,
                    callbacks.chunk_progress,
                    callbacks.stage_progress.clone(),
                    cancel.clone(),
                )
            },
        )?
    } else {
        let mut runtime = load_speaker_runtime(&params, &cancel, &callbacks.stage_progress)?;
        maybe_warmup_speaker_runtime(&mut runtime, &params.device, speech_audio, &cancel);
        compute_speaker_segments(
            &mut runtime.diarize,
            speech_audio,
            callbacks.chunk_progress,
            callbacks.stage_progress.clone(),
            cancel.clone(),
        )?
    };

    check_cancelled(&cancel)?;

    emit_stage(
        &callbacks.stage_progress,
        DiarizeStage::Finalizing,
        None,
        None,
        "Finalizing speaker segments...",
        true,
    );

    let num_speakers = count_unique_speakers(&result_segments);

    let inference_time = start.elapsed().as_secs_f64();

    Ok(DiarizationResult {
        segments: result_segments,
        num_speakers,
        audio_duration: speech_audio.duration_seconds as f32,
        inference_time,
        model: model_name,
        device: device_name,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::audio::{convert_to_mono, resample_to_16khz};
    use std::sync::atomic::AtomicBool;
    use std::sync::Arc;

    #[test]
    fn test_convert_to_mono_passthrough() {
        let samples = vec![0.1, 0.2, 0.3];
        let mono = convert_to_mono(&samples, 1);
        assert_eq!(mono, samples);
    }

    #[test]
    fn test_convert_to_mono_stereo() {
        let samples = vec![0.0f32, 1.0, 0.0, 1.0];
        let mono = convert_to_mono(&samples, 2);
        assert_eq!(mono.len(), 2);
        assert!((mono[0] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_resample_passthrough() {
        let samples = vec![0.1f32, 0.2, 0.3];
        let out = resample_to_16khz(&samples, 16000).unwrap();
        assert_eq!(out, samples);
    }

    #[test]
    fn test_should_skip_warmup_when_already_warmed() {
        let cancel = None;
        let should = should_attempt_warmup(true, &crate::speaker::SpeakerDevice::Cuda, &cancel);
        assert!(!should);
    }

    #[test]
    fn test_should_skip_warmup_on_cpu() {
        let cancel = None;
        let should = should_attempt_warmup(false, &crate::speaker::SpeakerDevice::Cpu, &cancel);
        assert!(!should);
    }

    #[test]
    fn test_should_skip_warmup_when_cancelled() {
        let cancel = Some(Arc::new(AtomicBool::new(true)));
        let should = should_attempt_warmup(false, &crate::speaker::SpeakerDevice::Cuda, &cancel);
        assert!(!should);
    }

    #[test]
    fn test_warmup_sample_window_bounds() {
        let max_samples = (CUDA_WARMUP_DURATION_SEC * SPEECH_SAMPLE_RATE as f32) as usize;
        assert_eq!(warmup_sample_count(100), 100);
        assert_eq!(warmup_sample_count(max_samples + 10_000), max_samples);
    }
}
