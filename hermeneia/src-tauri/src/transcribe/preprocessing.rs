use crate::error::{AudioError, Result};
use byteorder::{ByteOrder, LittleEndian};
use candle_core::{Device, Tensor};
use candle_transformers::models::whisper::{audio, Config};

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
    speech_audio: &crate::audio::SpeechAudio,
    config: &Config,
    device: &Device,
) -> Result<Tensor> {
    compute_mel_spectrogram(&speech_audio.samples_16k_mono, config, device)
}
