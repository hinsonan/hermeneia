use crate::audio::{prepare_speech_audio, AudioData, SpeechAudio};
use crate::error::{AudioError, Result};
use byteorder::{ByteOrder, LittleEndian};
use candle_core::{Device, Tensor};
use candle_transformers::models::whisper::{audio, Config};

/// Convert stereo to mono by averaging channels.
/// Kept for backward compatibility; delegates to shared audio preprocessing.
pub fn convert_to_mono(samples: &[f32], channels: u16) -> Vec<f32> {
    crate::audio::convert_to_mono(samples, channels)
}

/// Resample audio to 16kHz (Whisper's required sample rate).
/// Kept for backward compatibility; delegates to shared audio preprocessing.
pub fn resample_to_16khz(samples: &[f32], source_rate: u32) -> Result<Vec<f32>> {
    crate::audio::resample_to_16khz(samples, source_rate)
}

/// Compute mel-spectrogram using Candle
pub fn compute_mel_spectrogram(
    samples: &[f32],
    config: &Config,
    device: &Device,
) -> Result<Tensor> {
    let mel_bytes = match config.num_mel_bins {
        80 => include_bytes!("../../assets/melfilters.bytes").as_slice(),
        128 => include_bytes!("../../assets/melfilters128.bytes").as_slice(),
        _ => {
            return Err(AudioError::AudioPreprocessing(format!(
                "Unsupported mel bins: {}",
                config.num_mel_bins
            )))
        }
    };

    let mut mel_filters = vec![0f32; mel_bytes.len() / 4];
    LittleEndian::read_f32_into(mel_bytes, &mut mel_filters);

    let mel = audio::pcm_to_mel(config, samples, &mel_filters);
    let mel_len = mel.len();

    Tensor::from_vec(
        mel,
        (1, config.num_mel_bins, mel_len / config.num_mel_bins),
        device,
    )
    .map_err(|e| AudioError::AudioPreprocessing(format!("Tensor creation: {}", e)))
}

/// Preprocess shared speech audio to mel-spectrogram.
pub fn preprocess_speech_audio(
    speech_audio: &SpeechAudio,
    config: &Config,
    device: &Device,
) -> Result<Tensor> {
    compute_mel_spectrogram(&speech_audio.samples_16k_mono, config, device)
}

/// Preprocess audio: mono, 16kHz, mel-spectrogram.
/// Kept for backward compatibility.
pub fn preprocess_audio(audio: &AudioData, config: &Config, device: &Device) -> Result<Tensor> {
    let speech_audio = prepare_speech_audio(audio)?;
    preprocess_speech_audio(&speech_audio, config, device)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mono_conversion() {
        let stereo = vec![0.5, 0.3, 0.7, 0.1];
        let mono = convert_to_mono(&stereo, 2);
        assert_eq!(mono.len(), 2);
        assert!((mono[0] - 0.4).abs() < 0.001);
        assert!((mono[1] - 0.4).abs() < 0.001);
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
