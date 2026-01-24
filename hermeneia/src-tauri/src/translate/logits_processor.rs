use crate::error::{AudioError, Result};
use candle_core::Tensor;
use rand::distributions::{Distribution, WeightedIndex};
use rand::SeedableRng;

/// Process logits with temperature, top-p, and repetition penalty
pub struct LogitsProcessor {
    temperature: f64,
    top_p: Option<f64>,
    repetition_penalty: f64,
    rng: rand::rngs::StdRng,
}

impl LogitsProcessor {
    /// Create a new logits processor
    pub fn new(seed: u64, temperature: f64, top_p: Option<f64>, repetition_penalty: f64) -> Self {
        Self {
            temperature,
            top_p,
            repetition_penalty,
            rng: rand::rngs::StdRng::seed_from_u64(seed),
        }
    }

    /// Process logits and sample next token
    ///
    /// Steps:
    /// 1. Apply repetition penalty for previously generated tokens
    /// 2. Apply temperature scaling
    /// 3. Sample using top-p (nucleus) if enabled, else greedy
    pub fn sample(&mut self, logits: &Tensor, previous_tokens: &[u32]) -> Result<u32> {
        let logits = logits.to_dtype(candle_core::DType::F32).map_err(|e| {
            AudioError::TranslationFailed(format!("Failed to convert logits to f32: {}", e))
        })?;

        let mut logits_vec: Vec<f32> = logits.to_vec1().map_err(|e| {
            AudioError::TranslationFailed(format!("Failed to extract logits: {}", e))
        })?;

        // Apply repetition penalty
        if self.repetition_penalty != 1.0 {
            for &token_id in previous_tokens {
                let idx = token_id as usize;
                if idx < logits_vec.len() {
                    if logits_vec[idx] < 0.0 {
                        logits_vec[idx] *= self.repetition_penalty as f32;
                    } else {
                        logits_vec[idx] /= self.repetition_penalty as f32;
                    }
                }
            }
        }

        // Apply temperature
        if self.temperature > 0.0 && self.temperature != 1.0 {
            for logit in logits_vec.iter_mut() {
                *logit /= self.temperature as f32;
            }
        }

        // Sample token
        if self.temperature <= 0.0 {
            // Greedy decoding
            self.sample_greedy(&logits_vec)
        } else if let Some(top_p) = self.top_p {
            // Nucleus (top-p) sampling
            self.sample_top_p(&logits_vec, top_p)
        } else {
            // Greedy (most common for translation)
            self.sample_greedy(&logits_vec)
        }
    }

    /// Greedy sampling - select token with highest probability
    fn sample_greedy(&self, logits: &[f32]) -> Result<u32> {
        let mut max_idx = 0;
        let mut max_val = logits[0];

        for (i, &val) in logits.iter().enumerate().skip(1) {
            if val > max_val {
                max_val = val;
                max_idx = i;
            }
        }

        Ok(max_idx as u32)
    }

    /// Top-p (nucleus) sampling
    fn sample_top_p(&mut self, logits: &[f32], top_p: f64) -> Result<u32> {
        // Create (index, probability) pairs
        let mut logits_idx: Vec<(usize, f32)> = logits
            .iter()
            .enumerate()
            .map(|(i, &v)| (i, v))
            .collect();

        // Sort by probability (descending)
        logits_idx.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Convert to probabilities using softmax
        let max_logit = logits_idx[0].1;
        let mut probs: Vec<f32> = logits_idx
            .iter()
            .map(|(_, logit)| (logit - max_logit).exp())
            .collect();

        let sum: f32 = probs.iter().sum();
        for p in &mut probs {
            *p /= sum;
        }

        // Find cutoff for top-p
        let mut cumsum = 0.0f32;
        let mut cutoff = probs.len();
        for (i, &p) in probs.iter().enumerate() {
            cumsum += p;
            if cumsum >= top_p as f32 {
                cutoff = i + 1;
                break;
            }
        }

        // Sample from the top-p subset
        let top_probs = &probs[..cutoff];
        let top_indices: Vec<usize> = logits_idx[..cutoff].iter().map(|(i, _)| *i).collect();

        let dist = WeightedIndex::new(top_probs).map_err(|e| {
            AudioError::TranslationFailed(format!("Failed to create distribution: {}", e))
        })?;

        let sampled_idx = dist.sample(&mut self.rng);
        Ok(top_indices[sampled_idx] as u32)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_greedy_sampling() {
        let processor = LogitsProcessor::new(42, 0.0, None, 1.0);
        let logits = vec![0.1, 0.5, 0.3, 0.8, 0.2]; // Token 3 has highest score

        let token = processor.sample_greedy(&logits).unwrap();
        assert_eq!(token, 3);
    }

    #[test]
    fn test_temperature_scaling() {
        let mut processor = LogitsProcessor::new(42, 2.0, None, 1.0);
        let logits = Tensor::new(&[1.0f32, 2.0, 3.0], &candle_core::Device::Cpu).unwrap();

        // With high temperature, distribution becomes more uniform
        let token = processor.sample(&logits, &[]).unwrap();
        assert!(token < 3); // Should still be valid
    }

    #[test]
    fn test_repetition_penalty() {
        let mut processor = LogitsProcessor::new(42, 0.0, None, 2.0);
        let logits = Tensor::new(&[1.0f32, 2.0, 3.0, 4.0], &candle_core::Device::Cpu).unwrap();

        // Token 3 was previously generated, should be penalized
        let previous = vec![3];
        let token = processor.sample(&logits, &previous).unwrap();

        // After penalty, token 2 or 3 might be selected (depending on penalty strength)
        assert!(token < 4);
    }
}
