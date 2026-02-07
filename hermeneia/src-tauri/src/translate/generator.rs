use crate::error::{AudioError, Result};
use crate::translate::logits_processor::LogitsProcessor;
use crate::translate::types::ProgressCallback;
use candle_core::{Device, IndexOp, Tensor};
use candle_transformers::models::t5;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

/// Configuration for text generation
#[derive(Debug, Clone)]
pub struct GenerationConfig {
    pub max_length: usize,
    pub temperature: f64,
    pub top_p: Option<f64>,
    pub repetition_penalty: f64,
}

impl Default for GenerationConfig {
    fn default() -> Self {
        Self {
            max_length: 512,
            temperature: 1.0,
            top_p: Some(0.9),
            repetition_penalty: 1.2,
        }
    }
}

/// Generate translation using encoder-decoder model
pub struct Generator {
    config: GenerationConfig,
    logits_processor: LogitsProcessor,
}

impl Generator {
    /// Create a new generator with configuration
    pub fn new(config: GenerationConfig) -> Self {
        let logits_processor = LogitsProcessor::new(
            299792458, // seed
            config.temperature,
            config.top_p,
            config.repetition_penalty,
        );

        Self {
            config,
            logits_processor,
        }
    }

    /// Generate translation tokens using T5 model
    ///
    /// This is a simplified generation loop for encoder-decoder models.
    /// The actual implementation will depend on the model architecture.
    pub fn generate(
        &mut self,
        model: &mut t5::T5ForConditionalGeneration,
        encoder_output: &Tensor,
        decoder_start_token_id: u32,
        eos_token_id: u32,
        use_cache: bool,
        progress_callback: Option<&ProgressCallback>,
        cancel_flag: Option<&Arc<AtomicBool>>,
    ) -> Result<Vec<u32>> {
        let mut tokens = vec![decoder_start_token_id];
        let device = encoder_output.device();

        for step in 0..self.config.max_length {
            // Check for cancellation
            if let Some(flag) = cancel_flag {
                if flag.load(Ordering::SeqCst) {
                    tracing::info!("T5 translation cancelled by user at step {}", step);
                    return Err(AudioError::Cancelled);
                }
            }

            // Report progress
            if let Some(callback) = progress_callback {
                callback(step + 1, self.config.max_length);
            }

            // Prepare decoder input
            // For T5 with KV caching: first iteration uses all tokens, subsequent use only last token
            let decoder_input_ids = if step == 0 || !use_cache {
                // First iteration or cache disabled: pass all tokens
                Tensor::new(&tokens[..], device).map_err(|e| {
                    AudioError::TranslationFailed(format!("Failed to create decoder input: {}", e))
                })?
            } else {
                // Subsequent iterations: only pass the last token (KV cache handles the rest)
                let last_token = tokens.last().copied().unwrap();
                Tensor::new(&[last_token], device).map_err(|e| {
                    AudioError::TranslationFailed(format!("Failed to create decoder input: {}", e))
                })?
            };

            let decoder_input_ids = decoder_input_ids.unsqueeze(0).map_err(|e| {
                AudioError::TranslationFailed(format!("Failed to unsqueeze decoder input: {}", e))
            })?;

            // Forward pass through decoder
            // Note: T5's decode() already extracts the last token, returning [batch_size, vocab_size]
            let logits = model
                .decode(&decoder_input_ids, encoder_output)
                .map_err(|e| {
                    AudioError::TranslationFailed(format!("Decoder forward pass failed: {}", e))
                })?;

            // Get logits for batch 0 (T5 decode already returns last token logits)
            let next_token_logits = logits.i(0).map_err(|e| {
                AudioError::TranslationFailed(format!("Failed to extract logits: {}", e))
            })?;

            // Sample next token
            let next_token = self.logits_processor.sample(&next_token_logits, &tokens)?;

            tracing::debug!("Step {}: generated token {}", step, next_token);

            // Stop if EOS token generated
            if next_token == eos_token_id {
                tracing::info!("EOS token {} generated at step {}", eos_token_id, step);
                break;
            }

            tokens.push(next_token);

            // Safety: prevent infinite loops
            if tokens.len() > self.config.max_length * 2 {
                tracing::warn!(
                    "Generation exceeded max_length*2 ({}), stopping",
                    self.config.max_length * 2
                );
                break;
            }
        }

        Ok(tokens)
    }

    /// Generate translation with custom encoder-decoder architecture
    ///
    /// This is a more generic version that works with any encoder-decoder model
    /// by accepting encoder/decoder functions as closures.
    #[allow(dead_code)]
    pub fn generate_generic<E, D>(
        &mut self,
        encode: E,
        decode: D,
        decoder_start_token_id: u32,
        eos_token_id: u32,
        device: &Device,
        progress_callback: Option<&ProgressCallback>,
    ) -> Result<Vec<u32>>
    where
        E: Fn() -> Result<Tensor>,
        D: Fn(&Tensor) -> Result<Tensor>,
    {
        // Encode input
        let _encoder_output = encode()?;

        let mut tokens = vec![decoder_start_token_id];

        for step in 0..self.config.max_length {
            // Report progress
            if let Some(callback) = progress_callback {
                callback(step + 1, self.config.max_length);
            }

            // Prepare decoder input
            let decoder_input_ids = Tensor::new(&tokens[..], device).map_err(|e| {
                AudioError::TranslationFailed(format!("Failed to create decoder input: {}", e))
            })?;

            // Forward pass
            let logits = decode(&decoder_input_ids)?;

            // Get last token logits
            let next_token_logits = logits.i((0, tokens.len() - 1)).map_err(|e| {
                AudioError::TranslationFailed(format!("Failed to extract logits: {}", e))
            })?;

            // Sample next token
            let next_token = self.logits_processor.sample(&next_token_logits, &tokens)?;

            if next_token == eos_token_id {
                break;
            }

            tokens.push(next_token);
        }

        Ok(tokens)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generation_config_default() {
        let config = GenerationConfig::default();
        assert_eq!(config.max_length, 512);
        assert_eq!(config.temperature, 1.0);
        assert_eq!(config.top_p, Some(0.9));
        assert_eq!(config.repetition_penalty, 1.2);
    }

    #[test]
    fn test_generator_creation() {
        let config = GenerationConfig::default();
        let generator = Generator::new(config);
        assert_eq!(generator.config.max_length, 512);
    }
}
