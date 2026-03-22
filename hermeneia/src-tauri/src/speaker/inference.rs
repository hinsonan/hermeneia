use crate::audio::{decode_audio_file, prepare_speech_audio, SpeechAudio};
use crate::error::{AudioError, Result};
use crate::speaker::{
    model::SpeakerModelManager,
    types::{validate_diarize_params, DiarizationResult, DiarizeParams, SpeakerSegment},
};
use sherpa_rs::diarize::{Diarize, DiarizeConfig};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Instant;

/// Progress callback: (num_processed_chunks, num_total_chunks)
pub type DiarizeProgressCallback = Box<dyn Fn(i32, i32) + Send + Sync>;

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

/// Diarize already-preprocessed mono 16kHz speech audio with progress/cancel.
pub fn diarize_prepared_audio_with_progress(
    speech_audio: &SpeechAudio,
    params: DiarizeParams,
    progress: Option<DiarizeProgressCallback>,
    cancel: Option<Arc<AtomicBool>>,
) -> Result<DiarizationResult> {
    let start = Instant::now();
    validate_diarize_params(&params)?;

    tracing::info!("Ensuring speaker diarization models...");
    let (seg_path, emb_path) = SpeakerModelManager::ensure_models(&params.model)?;

    check_cancelled(&cancel)?;

    let device_name = params.device.provider_string().to_string();
    let model_name = params.model.display_name().to_string();

    let config = DiarizeConfig {
        num_clusters: params.num_speakers,
        threshold: Some(params.threshold),
        provider: Some(params.device.provider_string().to_string()),
        ..Default::default()
    };

    tracing::info!(
        "Running diarization with {} model on {}...",
        model_name,
        device_name
    );

    let mut sd = Diarize::new(&seg_path, &emb_path, config)
        .map_err(|e| AudioError::DiarizationFailed(e.to_string()))?;

    let cancel_for_callback = cancel.clone();
    let sherpa_callback = progress.map(|cb| {
        let cb = Arc::new(cb);
        Box::new(move |processed: i32, total: i32| -> i32 {
            if cancel_for_callback
                .as_ref()
                .map(|c| c.load(Ordering::SeqCst))
                .unwrap_or(false)
            {
                return 1;
            }
            cb(processed, total);
            0
        }) as Box<dyn Fn(i32, i32) -> i32 + Send + 'static>
    });

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

    let num_speakers = {
        let mut ids: Vec<i32> = result_segments.iter().map(|s| s.speaker).collect();
        ids.sort_unstable();
        ids.dedup();
        ids.len()
    };

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
    use crate::audio::{convert_to_mono, resample_to_16khz};

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
}
