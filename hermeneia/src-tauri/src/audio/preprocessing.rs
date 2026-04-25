use rubato::{Resampler, SincFixedIn, SincInterpolationParameters, SincInterpolationType};

use crate::audio::types::{AudioData, SpeechAudio};
use crate::error::{AudioError, Result};

/// Prepare decoded audio for speech models (Whisper + diarization):
/// mono + 16kHz.
pub fn prepare_speech_audio(audio: &AudioData) -> Result<SpeechAudio> {
    if audio.sample_rate == 0 {
        return Err(AudioError::AudioPreprocessing(
            "Sample rate cannot be zero".to_string(),
        ));
    }

    let mono = convert_to_mono(&audio.samples, audio.channels);
    let samples_16k_mono = resample_to_16khz(&mono, audio.sample_rate)?;

    Ok(SpeechAudio {
        samples_16k_mono,
        duration_seconds: audio.duration_seconds(),
    })
}

/// Prepare decoded audio for speech models while consuming owned audio.
///
/// This avoids extra full-buffer copies for mono/16kHz passthrough paths.
pub fn prepare_speech_audio_owned(audio: AudioData) -> Result<SpeechAudio> {
    if audio.sample_rate == 0 {
        return Err(AudioError::AudioPreprocessing(
            "Sample rate cannot be zero".to_string(),
        ));
    }

    let duration_seconds = audio.duration_seconds();
    let mono = convert_to_mono_owned(audio.samples, audio.channels);
    let samples_16k_mono = resample_to_16khz_owned(mono, audio.sample_rate)?;

    Ok(SpeechAudio {
        samples_16k_mono,
        duration_seconds,
    })
}

/// Convert interleaved multi-channel PCM to mono by averaging channels.
pub fn convert_to_mono(samples: &[f32], channels: u16) -> Vec<f32> {
    if channels <= 1 {
        return samples.to_vec();
    }

    samples
        .chunks(channels as usize)
        .map(|chunk| chunk.iter().sum::<f32>() / channels as f32)
        .collect()
}

/// Convert interleaved multi-channel PCM to mono by averaging channels,
/// consuming owned input.
pub fn convert_to_mono_owned(samples: Vec<f32>, channels: u16) -> Vec<f32> {
    if channels <= 1 {
        return samples;
    }

    samples
        .chunks(channels as usize)
        .map(|chunk| chunk.iter().sum::<f32>() / channels as f32)
        .collect()
}

/// Resample mono PCM from `source_rate` to 16kHz.
pub fn resample_to_16khz(samples: &[f32], source_rate: u32) -> Result<Vec<f32>> {
    if source_rate == 0 {
        return Err(AudioError::AudioPreprocessing(
            "Source sample rate cannot be zero".to_string(),
        ));
    }

    if source_rate == 16000 || samples.is_empty() {
        return Ok(samples.to_vec());
    }

    let params = SincInterpolationParameters {
        sinc_len: 128,
        f_cutoff: 0.98,
        interpolation: SincInterpolationType::Linear,
        oversampling_factor: 128,
        window: rubato::WindowFunction::BlackmanHarris2,
    };

    let mut resampler =
        SincFixedIn::<f32>::new(16000.0 / source_rate as f64, 2.0, params, samples.len(), 1)
            .map_err(|e| AudioError::AudioPreprocessing(format!("Resampler init: {}", e)))?;

    let mut wave_out = resampler.output_buffer_allocate(true);
    let wave_in = [samples];
    let (_, output_frames) = resampler
        .process_into_buffer(&wave_in, &mut wave_out, None)
        .map_err(|e| AudioError::AudioPreprocessing(format!("Resampling: {}", e)))?;

    let mut output = wave_out.into_iter().next().unwrap_or_default();
    output.truncate(output_frames);
    Ok(output)
}

/// Resample mono PCM from `source_rate` to 16kHz, consuming owned input.
pub fn resample_to_16khz_owned(samples: Vec<f32>, source_rate: u32) -> Result<Vec<f32>> {
    if source_rate == 0 {
        return Err(AudioError::AudioPreprocessing(
            "Source sample rate cannot be zero".to_string(),
        ));
    }

    if source_rate == 16000 || samples.is_empty() {
        return Ok(samples);
    }

    resample_to_16khz(&samples, source_rate)
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
        let samples = vec![0.0f32, 1.0, 0.2, 0.6];
        let mono = convert_to_mono(&samples, 2);
        assert_eq!(mono.len(), 2);
        assert!((mono[0] - 0.5).abs() < 1e-6);
        assert!((mono[1] - 0.4).abs() < 1e-6);
    }

    #[test]
    fn test_resample_passthrough() {
        let samples = vec![0.1f32, 0.2, 0.3];
        let out = resample_to_16khz(&samples, 16000).unwrap();
        assert_eq!(out, samples);
    }

    #[test]
    fn test_prepare_speech_audio_preserves_duration() {
        let audio = AudioData {
            samples: vec![0.0; 48_000],
            sample_rate: 48_000,
            channels: 1,
        };

        let speech = prepare_speech_audio(&audio).unwrap();
        assert!((speech.duration_seconds - 1.0).abs() < 1e-6);
        assert!(!speech.samples_16k_mono.is_empty());
    }

    #[test]
    fn test_prepare_speech_audio_owned_passthrough_uses_same_buffer() {
        let samples = vec![0.1f32, 0.2, 0.3, 0.4];
        let samples_ptr = samples.as_ptr();

        let audio = AudioData {
            samples,
            sample_rate: 16_000,
            channels: 1,
        };

        let speech = prepare_speech_audio_owned(audio).unwrap();
        assert_eq!(speech.samples_16k_mono.as_ptr(), samples_ptr);
    }
}
