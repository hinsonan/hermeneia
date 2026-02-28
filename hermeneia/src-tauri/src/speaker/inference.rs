use crate::audio::decode_audio_file;
use crate::error::{AudioError, Result};
use crate::speaker::{
    model::SpeakerModelManager,
    types::{DiarizeParams, DiarizationResult, SpeakerSegment},
};
use rubato::{Resampler, SincFixedIn, SincInterpolationParameters, SincInterpolationType};
use sherpa_rs::diarize::{Diarize, DiarizeConfig};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Instant;

/// Progress callback: (num_processed_chunks, num_total_chunks)
pub type DiarizeProgressCallback = Box<dyn Fn(i32, i32) + Send + Sync>;

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
    let start = Instant::now();

    // 1. Download / locate ONNX models
    tracing::info!("Ensuring speaker diarization models...");
    let (seg_path, emb_path) = SpeakerModelManager::ensure_models(&params.model)?;

    // 2. Check cancellation
    if cancel.as_ref().map(|c| c.load(Ordering::SeqCst)).unwrap_or(false) {
        return Err(AudioError::Cancelled);
    }

    // 3. Decode audio to f32 PCM
    tracing::info!("Decoding audio: {}", audio_path);
    let audio = decode_audio_file(audio_path)?;
    let audio_duration = audio.duration_seconds() as f32;

    // 4. Convert to mono
    let mono_samples = convert_to_mono(&audio.samples, audio.channels);

    // 5. Resample to 16kHz (sherpa-rs requirement)
    let samples_16k = resample_to_16khz(&mono_samples, audio.sample_rate)?;

    // 6. Check cancellation again
    if cancel.as_ref().map(|c| c.load(Ordering::SeqCst)).unwrap_or(false) {
        return Err(AudioError::Cancelled);
    }

    let device_name = params.device.provider_string().to_string();
    let model_name = params.model.display_name().to_string();

    // 7. Build DiarizeConfig
    let config = DiarizeConfig {
        num_clusters: params.num_speakers,
        threshold: Some(params.threshold),
        provider: Some(params.device.provider_string().to_string()),
        ..Default::default()
    };

    // 8. Construct Diarize and run inference
    tracing::info!(
        "Running diarization with {} model on {}...",
        model_name,
        device_name
    );

    let mut sd = Diarize::new(&seg_path, &emb_path, config)
        .map_err(|e| AudioError::DiarizationFailed(e.to_string()))?;

    // Wrap the optional progress callback into sherpa-rs's expected signature:
    // Box<dyn Fn(i32, i32) -> i32 + Send + 'static>
    let sherpa_callback = progress.map(|cb| {
        let cb = Arc::new(cb);
        Box::new(move |processed: i32, total: i32| -> i32 {
            cb(processed, total);
            0 // returning 0 tells sherpa-rs to continue
        }) as Box<dyn Fn(i32, i32) -> i32 + Send + 'static>
    });

    let segments = sd
        .compute(samples_16k, sherpa_callback)
        .map_err(|e| AudioError::DiarizationFailed(e.to_string()))?;

    // 9. Map sherpa_rs::Segment → SpeakerSegment, count unique speakers
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
        audio_duration,
        inference_time,
        model: model_name,
        device: device_name,
    })
}

/// Convert interleaved multi-channel PCM to mono by averaging channels.
fn convert_to_mono(samples: &[f32], channels: u16) -> Vec<f32> {
    if channels <= 1 {
        return samples.to_vec();
    }
    samples
        .chunks(channels as usize)
        .map(|chunk| chunk.iter().sum::<f32>() / channels as f32)
        .collect()
}

/// Resample mono PCM from `source_rate` to 16kHz.
fn resample_to_16khz(samples: &[f32], source_rate: u32) -> Result<Vec<f32>> {
    if source_rate == 16000 {
        return Ok(samples.to_vec());
    }

    let params = SincInterpolationParameters {
        sinc_len: 128,
        f_cutoff: 0.98,
        interpolation: SincInterpolationType::Linear,
        oversampling_factor: 128,
        window: rubato::WindowFunction::BlackmanHarris2,
    };

    let mut resampler = SincFixedIn::<f32>::new(
        16000.0 / source_rate as f64,
        2.0,
        params,
        samples.len(),
        1,
    )
    .map_err(|e| AudioError::AudioPreprocessing(format!("Resampler init: {}", e)))?;

    let waves_in = vec![samples.to_vec()];
    let waves_out = resampler
        .process(&waves_in, None)
        .map_err(|e| AudioError::AudioPreprocessing(format!("Resampling: {}", e)))?;

    waves_out
        .into_iter()
        .next()
        .ok_or_else(|| AudioError::AudioPreprocessing("Resampler produced no output".into()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_convert_to_mono_passthrough() {
        let samples = vec![0.1, 0.2, 0.3];
        let mono = convert_to_mono(&samples, 1);
        assert_eq!(mono, samples);
    }

    #[test]
    fn test_convert_to_mono_stereo() {
        // Stereo: L=0.0, R=1.0 → mono=0.5
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
