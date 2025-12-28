// src-tauri/src/audio/encoder.rs

use hound::{SampleFormat, WavSpec, WavWriter};
use std::path::Path;

use crate::audio::types::AudioData;
use crate::error::Result;

/// Encode PCM audio data to a WAV file
/// 
/// Outputs 32-bit float WAV files for maximum quality
/// 
/// # Arguments
/// * `audio` - The audio data to encode
/// * `output_path` - Where to save the WAV file
/// 
/// # Example
/// ```
/// use hermeneia_lib::audio::{AudioData, encode_wav};
/// 
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// // Create some test audio data
/// let audio = AudioData {
///     samples: vec![0.0, 0.5, -0.5, 1.0, -1.0],
///     sample_rate: 44100,
///     channels: 1,
/// };
/// 
/// // Encode to WAV file
/// # let temp_dir = std::env::temp_dir();
/// # let output_path = temp_dir.join("test_output.wav");
/// encode_wav(&audio, &output_path)?;
/// # std::fs::remove_file(&output_path).ok();
/// # Ok(())
/// # }
/// ```
pub fn encode_wav<P: AsRef<Path>>(audio: &AudioData, output_path: P) -> Result<()> {
    // Configure WAV file specification
    let spec = WavSpec {
        channels: audio.channels,
        sample_rate: audio.sample_rate,
        bits_per_sample: 32, // 32-bit float for maximum quality
        sample_format: SampleFormat::Float,
    };

    // Create WAV writer
    let mut writer = WavWriter::create(output_path, spec)?;

    // Write all samples
    for &sample in &audio.samples {
        writer.write_sample(sample)?;
    }

    // Finalize the file (writes headers, etc.)
    writer.finalize()?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use hound::WavReader;

    #[test]
    fn test_encode_and_decode_wav() {
        // Create test audio
        let test_audio = AudioData {
            samples: vec![0.0, 0.5, -0.5, 1.0, -1.0],
            sample_rate: 44100,
            channels: 1,
        };

        // Encode to temporary file
        let temp_path = std::env::temp_dir().join("test_encode.wav");
        encode_wav(&test_audio, &temp_path).unwrap();

        // Read it back
        let mut reader = WavReader::open(&temp_path).unwrap();
        let samples: Vec<f32> = reader.samples::<f32>().map(|s| s.unwrap()).collect();

        // Verify
        assert_eq!(samples.len(), test_audio.samples.len());
        for (original, decoded) in test_audio.samples.iter().zip(samples.iter()) {
            assert!((original - decoded).abs() < 0.0001);
        }

        // Cleanup
        std::fs::remove_file(temp_path).ok();
    }
}