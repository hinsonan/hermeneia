use crate::audio::AudioData;
use crate::error::{AudioError, Result};
use candle_core::{Device, Tensor};
use rubato::{Resampler, SincFixedIn, SincInterpolationParameters, SincInterpolationType};

/// Convert stereo to mono by averaging channels
pub fn convert_to_mono(samples: &[f32], channels: u16) -> Vec<f32> {
    if channels == 1 {
        return samples.to_vec();
    }

    samples
        .chunks(channels as usize)
        .map(|chunk| chunk.iter().sum::<f32>() / channels as f32)
        .collect()
}

/// Resample audio to 16kHz (Whisper's required sample rate)
pub fn resample_to_16khz(samples: &[f32], source_rate: u32) -> Result<Vec<f32>> {
    if source_rate == 16000 {
        return Ok(samples.to_vec());
    }

    let params = SincInterpolationParameters {
        sinc_len: 256,
        f_cutoff: 0.95,
        interpolation: SincInterpolationType::Linear,
        oversampling_factor: 256,
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

    Ok(waves_out[0].clone())
}

/// Compute mel-spectrogram using Candle
pub fn compute_mel_spectrogram(_samples: &[f32]) -> Result<Tensor> {
    // Stub implementation - returns a tensor with placeholder dimensions
    // TODO: Implement proper mel-spectrogram using candle_transformers::models::whisper::audio
    let device = Device::Cpu;
    let mel_data = vec![0.0f32; 80 * 3000]; // 80 mel bins, 3000 frames
    // Whisper encoder expects shape [batch, features, time]
    Tensor::from_vec(mel_data, (1, 80, 3000), &device)
        .map_err(|e| AudioError::AudioPreprocessing(format!("Tensor creation: {}", e)))
}

/// Preprocess audio: mono, 16kHz, mel-spectrogram
pub fn preprocess_audio(audio: &AudioData) -> Result<Tensor> {
    // Step 1: Convert to mono
    let mono = convert_to_mono(&audio.samples, audio.channels);

    // Step 2: Resample to 16kHz
    let resampled = resample_to_16khz(&mono, audio.sample_rate)?;

    // Step 3: Compute mel-spectrogram
    compute_mel_spectrogram(&resampled)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mono_conversion() {
        let stereo = vec![0.5, 0.3, 0.7, 0.1]; // L, R, L, R
        let mono = convert_to_mono(&stereo, 2);
        assert_eq!(mono.len(), 2);
        assert!((mono[0] - 0.4).abs() < 0.001); // (0.5 + 0.3) / 2
        assert!((mono[1] - 0.4).abs() < 0.001); // (0.7 + 0.1) / 2
    }

    #[test]
    fn test_mono_passthrough() {
        let mono_input = vec![0.1, 0.2, 0.3];
        let mono_output = convert_to_mono(&mono_input, 1);
        assert_eq!(mono_input, mono_output);
    }

    #[test]
    fn test_resample_passthrough() {
        let samples = vec![0.1, 0.2, 0.3];
        let result = resample_to_16khz(&samples, 16000).unwrap();
        assert_eq!(samples, result);
    }
}
