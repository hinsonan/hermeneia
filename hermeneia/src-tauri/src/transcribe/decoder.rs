use crate::error::{AudioError, Result};
use crate::transcribe::types::{ProgressCallback, TranscribeParams, TranscriptionTask, TranscriptSegment};
use candle_core::{Device, IndexOp, Tensor};
use candle_nn::ops::{log_softmax, softmax};
use candle_transformers::models::whisper::{self as m, Config};
use rand::distributions::{Distribution, WeightedIndex};
use rand::SeedableRng;
use tokenizers::Tokenizer;

/// Decoding result with tokens and metadata
#[derive(Debug, Clone)]
pub struct DecodingResult {
    pub tokens: Vec<u32>,
    pub text: String,
    pub avg_logprob: f64,
    pub no_speech_prob: f64,
    pub _temperature: f64,
    pub compression_ratio: f64,
}

/// Audio segment with timing information
#[derive(Debug, Clone)]
pub struct Segment {
    pub start: f64,
    pub duration: f64,
    pub dr: DecodingResult,
}

/// Decoder for Whisper model
pub struct Decoder<'a> {
    model: &'a mut m::model::Whisper,
    rng: rand::rngs::StdRng,
    task: TranscriptionTask,
    timestamps: bool,
    max_initial_timestamp_index: Option<u32>,
    _verbose: bool,
    tokenizer: &'a Tokenizer,
    suppress_tokens: Tensor,
    sot_token: u32,
    transcribe_token: u32,
    translate_token: u32,
    eot_token: u32,
    no_speech_token: u32,
    no_timestamps_token: u32,
    language_token: Option<u32>,
    // Progress tracking
    progress_callback: Option<ProgressCallback>,
    total_frames: usize,
    current_segment_start: usize,
    current_segment_size: usize,
}

impl<'a> Decoder<'a> {
    #[allow(clippy::too_many_arguments)]
    pub fn new_with_language_token(
        model: &'a mut m::model::Whisper,
        tokenizer: &'a Tokenizer,
        config: &Config,
        device: &Device,
        params: &TranscribeParams,
        language_token: Option<u32>,
    ) -> Result<Self> {
        let no_timestamps_token = token_id(tokenizer, m::NO_TIMESTAMPS_TOKEN)?;

        // Suppress tokens that should not be generated
        let suppress_tokens: Vec<f32> = (0..config.vocab_size as u32)
            .map(|i| {
                if config.suppress_tokens.contains(&i)
                    || params.timestamps && i == no_timestamps_token
                {
                    f32::NEG_INFINITY
                } else {
                    0f32
                }
            })
            .collect();
        let suppress_tokens = Tensor::new(suppress_tokens.as_slice(), device)
            .map_err(|e| AudioError::TranscriptionFailed(format!("Suppress tokens: {}", e)))?;

        let sot_token = token_id(tokenizer, m::SOT_TOKEN)?;
        let transcribe_token = token_id(tokenizer, m::TRANSCRIBE_TOKEN)?;
        let translate_token = token_id(tokenizer, m::TRANSLATE_TOKEN)?;
        let eot_token = token_id(tokenizer, m::EOT_TOKEN)?;

        let no_speech_token = m::NO_SPEECH_TOKENS
            .iter()
            .find_map(|token| token_id(tokenizer, token).ok())
            .ok_or_else(|| AudioError::TranscriptionFailed("No speech token not found".to_string()))?;

        Ok(Self {
            model,
            rng: rand::rngs::StdRng::seed_from_u64(299792458),
            tokenizer,
            task: params.task,
            timestamps: params.timestamps,
            max_initial_timestamp_index: None,
            _verbose: false,
            suppress_tokens,
            sot_token,
            transcribe_token,
            translate_token,
            eot_token,
            no_speech_token,
            language_token,
            no_timestamps_token,
            progress_callback: None,
            total_frames: 0,
            current_segment_start: 0,
            current_segment_size: 0,
        })
    }

    /// Decode a single mel-spectrogram segment
    pub fn decode(&mut self, mel: &Tensor, temperature: f64) -> Result<DecodingResult> {
        let audio_features = self
            .model
            .encoder
            .forward(mel, true)
            .map_err(|e| AudioError::TranscriptionFailed(format!("Encoder failed: {}", e)))?;

        let sample_len = self.model.config.max_target_positions / 2;
        let mut sum_logprob = 0f64;
        let mut no_speech_prob = f64::NAN;

        // Initialize tokens with SOT and task tokens
        let mut tokens = vec![self.sot_token];
        if let Some(language_token) = self.language_token {
            tokens.push(language_token);
        }
        match self.task {
            TranscriptionTask::Transcribe => tokens.push(self.transcribe_token),
            TranscriptionTask::Translate => tokens.push(self.translate_token),
        }
        if !self.timestamps {
            tokens.push(self.no_timestamps_token);
        }

        // Token generation loop
        for i in 0..sample_len {
            // Report fine-grained progress every 10 tokens
            if i % 10 == 0 && self.total_frames > 0 {
                if let Some(ref callback) = self.progress_callback {
                    // Calculate progress: segment progress + token progress within segment
                    let segment_progress_ratio = i as f64 / sample_len as f64;
                    let current_frame = self.current_segment_start +
                        (self.current_segment_size as f64 * segment_progress_ratio) as usize;
                    callback(current_frame, self.total_frames);
                }
            }

            let tokens_t = Tensor::new(tokens.as_slice(), mel.device())
                .map_err(|e| AudioError::TranscriptionFailed(format!("Token tensor: {}", e)))?;
            let tokens_t = tokens_t
                .unsqueeze(0)
                .map_err(|e| AudioError::TranscriptionFailed(format!("Unsqueeze: {}", e)))?;

            let ys = self
                .model
                .decoder
                .forward(&tokens_t, &audio_features, i == 0)
                .map_err(|e| AudioError::TranscriptionFailed(format!("Decoder failed: {}", e)))?;

            // Extract no speech probability on first iteration
            if i == 0 {
                let logits = self
                    .model
                    .decoder
                    .final_linear(&ys.i(..1).map_err(|e| {
                        AudioError::TranscriptionFailed(format!("Index error: {}", e))
                    })?)
                    .map_err(|e| AudioError::TranscriptionFailed(format!("Final linear: {}", e)))?
                    .i(0)
                    .map_err(|e| AudioError::TranscriptionFailed(format!("Index error: {}", e)))?
                    .i(0)
                    .map_err(|e| AudioError::TranscriptionFailed(format!("Index error: {}", e)))?;

                no_speech_prob = softmax(&logits, 0)
                    .map_err(|e| AudioError::TranscriptionFailed(format!("Softmax: {}", e)))?
                    .i(self.no_speech_token as usize)
                    .map_err(|e| AudioError::TranscriptionFailed(format!("Index error: {}", e)))?
                    .to_scalar::<f32>()
                    .map_err(|e| AudioError::TranscriptionFailed(format!("To scalar: {}", e)))?
                    as f64;
            }

            let (_, seq_len, _) = ys
                .dims3()
                .map_err(|e| AudioError::TranscriptionFailed(format!("Dims3: {}", e)))?;
            let logits = self
                .model
                .decoder
                .final_linear(
                    &ys.i((..1, seq_len - 1..))
                        .map_err(|e| AudioError::TranscriptionFailed(format!("Index error: {}", e)))?,
                )
                .map_err(|e| AudioError::TranscriptionFailed(format!("Final linear: {}", e)))?
                .i(0)
                .map_err(|e| AudioError::TranscriptionFailed(format!("Index error: {}", e)))?
                .i(0)
                .map_err(|e| AudioError::TranscriptionFailed(format!("Index error: {}", e)))?;

            // Apply timestamp rules when timestamps are enabled
            let logits = if self.timestamps {
                self.apply_timestamp_rules(&logits, &tokens)?
            } else {
                logits
            };

            let logits = logits
                .broadcast_add(&self.suppress_tokens)
                .map_err(|e| AudioError::TranscriptionFailed(format!("Broadcast add: {}", e)))?;

            // Sample next token
            let next_token = if temperature > 0.0 {
                let prs = softmax(&(&logits / temperature).map_err(|e| {
                    AudioError::TranscriptionFailed(format!("Division: {}", e))
                })?, 0)
                .map_err(|e| AudioError::TranscriptionFailed(format!("Softmax: {}", e)))?;
                let logits_v: Vec<f32> = prs
                    .to_vec1()
                    .map_err(|e| AudioError::TranscriptionFailed(format!("To vec: {}", e)))?;
                let distr = WeightedIndex::new(&logits_v)
                    .map_err(|e| AudioError::TranscriptionFailed(format!("WeightedIndex: {}", e)))?;
                distr.sample(&mut self.rng) as u32
            } else {
                // Greedy sampling
                let logits_v: Vec<f32> = logits
                    .to_vec1()
                    .map_err(|e| AudioError::TranscriptionFailed(format!("To vec: {}", e)))?;
                logits_v
                    .iter()
                    .enumerate()
                    .max_by(|(_, u), (_, v)| u.total_cmp(v))
                    .map(|(i, _)| i as u32)
                    .unwrap()
            };

            tokens.push(next_token);
            let prob = softmax(&logits, candle_core::D::Minus1)
                .map_err(|e| AudioError::TranscriptionFailed(format!("Softmax: {}", e)))?
                .i(next_token as usize)
                .map_err(|e| AudioError::TranscriptionFailed(format!("Index error: {}", e)))?
                .to_scalar::<f32>()
                .map_err(|e| AudioError::TranscriptionFailed(format!("To scalar: {}", e)))?
                as f64;

            if next_token == self.eot_token || tokens.len() > self.model.config.max_target_positions {
                break;
            }
            sum_logprob += prob.ln();
        }

        let text = self
            .tokenizer
            .decode(&tokens, true)
            .map_err(|e| AudioError::TranscriptionFailed(format!("Tokenizer decode: {}", e)))?;
        let avg_logprob = sum_logprob / tokens.len() as f64;

        Ok(DecodingResult {
            tokens,
            text,
            avg_logprob,
            no_speech_prob,
            _temperature: temperature,
            compression_ratio: f64::NAN,
        })
    }

    /// Decode with temperature fallback
    pub fn decode_with_fallback(&mut self, segment: &Tensor) -> Result<DecodingResult> {
        for (i, &temperature) in m::TEMPERATURES.iter().enumerate() {
            let dr = self.decode(segment, temperature);

            if i == m::TEMPERATURES.len() - 1 {
                return dr;
            }

            match dr {
                Ok(dr) => {
                    let needs_fallback = dr.compression_ratio > m::COMPRESSION_RATIO_THRESHOLD
                        || dr.avg_logprob < m::LOGPROB_THRESHOLD;
                    if !needs_fallback || dr.no_speech_prob > m::NO_SPEECH_THRESHOLD {
                        return Ok(dr);
                    }
                }
                Err(_) => continue,
            }
        }

        Err(AudioError::TranscriptionFailed(
            "All temperature fallbacks failed".to_string(),
        ))
    }

    /// Apply timestamp rules to logits
    fn apply_timestamp_rules(&self, input_logits: &Tensor, tokens: &[u32]) -> Result<Tensor> {
        let device = input_logits.device().clone();
        let timestamp_begin = self.no_timestamps_token + 1;
        let vocab_size = self.model.config.vocab_size as u32;

        let sample_begin = if self.language_token.is_some() { 3 } else { 2 };
        let sampled_tokens = if tokens.len() > sample_begin {
            &tokens[sample_begin..]
        } else {
            &[]
        };

        let mut masks = Vec::new();
        let mut mask_buffer = vec![0.0f32; vocab_size as usize];

        // Rule 1: Timestamp pairing constraints
        if !sampled_tokens.is_empty() {
            let last_was_timestamp = sampled_tokens
                .last()
                .map(|&t| t >= timestamp_begin)
                .unwrap_or(false);

            let penultimate_was_timestamp = if sampled_tokens.len() >= 2 {
                sampled_tokens[sampled_tokens.len() - 2] >= timestamp_begin
            } else {
                false
            };

            if last_was_timestamp {
                if penultimate_was_timestamp {
                    // Must be non-timestamp
                    for i in 0..vocab_size {
                        mask_buffer[i as usize] = if i >= timestamp_begin {
                            f32::NEG_INFINITY
                        } else {
                            0.0
                        };
                    }
                    masks.push(
                        Tensor::new(mask_buffer.as_slice(), &device).map_err(|e| {
                            AudioError::TranscriptionFailed(format!("Tensor creation: {}", e))
                        })?,
                    );
                } else {
                    // Must be timestamp or EOT
                    for i in 0..vocab_size {
                        mask_buffer[i as usize] = if i < self.eot_token {
                            f32::NEG_INFINITY
                        } else {
                            0.0
                        };
                    }
                    masks.push(
                        Tensor::new(mask_buffer.as_slice(), &device).map_err(|e| {
                            AudioError::TranscriptionFailed(format!("Tensor creation: {}", e))
                        })?,
                    );
                }
            }

            // Rule 2: Non-decreasing timestamp constraint
            let timestamp_tokens: Vec<u32> = sampled_tokens
                .iter()
                .filter(|&&t| t >= timestamp_begin)
                .cloned()
                .collect();

            if !timestamp_tokens.is_empty() {
                let timestamp_last = if last_was_timestamp && !penultimate_was_timestamp {
                    *timestamp_tokens.last().unwrap()
                } else {
                    timestamp_tokens.last().unwrap() + 1
                };

                for i in 0..vocab_size {
                    mask_buffer[i as usize] = if i >= timestamp_begin && i < timestamp_last {
                        f32::NEG_INFINITY
                    } else {
                        0.0
                    };
                }
                masks.push(
                    Tensor::new(mask_buffer.as_slice(), &device).map_err(|e| {
                        AudioError::TranscriptionFailed(format!("Tensor creation: {}", e))
                    })?,
                );
            }
        }

        // Rule 3: Force initial timestamp
        if tokens.len() == sample_begin {
            for i in 0..vocab_size {
                mask_buffer[i as usize] = if i < timestamp_begin {
                    f32::NEG_INFINITY
                } else {
                    0.0
                };
            }
            masks.push(
                Tensor::new(mask_buffer.as_slice(), &device).map_err(|e| {
                    AudioError::TranscriptionFailed(format!("Tensor creation: {}", e))
                })?,
            );

            if let Some(max_initial_timestamp_index) = self.max_initial_timestamp_index {
                let last_allowed = timestamp_begin + max_initial_timestamp_index;
                if last_allowed < vocab_size {
                    for i in 0..vocab_size {
                        mask_buffer[i as usize] = if i > last_allowed {
                            f32::NEG_INFINITY
                        } else {
                            0.0
                        };
                    }
                    masks.push(
                        Tensor::new(mask_buffer.as_slice(), &device).map_err(|e| {
                            AudioError::TranscriptionFailed(format!("Tensor creation: {}", e))
                        })?,
                    );
                }
            }
        }

        // Apply all masks
        let mut logits = input_logits.clone();
        for mask in masks {
            logits = logits
                .broadcast_add(&mask)
                .map_err(|e| AudioError::TranscriptionFailed(format!("Broadcast add: {}", e)))?;
        }

        // Rule 4: Probability-based timestamp preference
        let log_probs = log_softmax(&logits, 0)
            .map_err(|e| AudioError::TranscriptionFailed(format!("Log softmax: {}", e)))?;

        let timestamp_log_probs = log_probs
            .narrow(
                0,
                timestamp_begin as usize,
                vocab_size as usize - timestamp_begin as usize,
            )
            .map_err(|e| AudioError::TranscriptionFailed(format!("Narrow: {}", e)))?;

        let text_log_probs = log_probs
            .narrow(0, 0, timestamp_begin as usize)
            .map_err(|e| AudioError::TranscriptionFailed(format!("Narrow: {}", e)))?;

        // Compute log-sum-exp for timestamp tokens
        let timestamp_logprob = {
            let max_val = timestamp_log_probs
                .max(0)
                .map_err(|e| AudioError::TranscriptionFailed(format!("Max: {}", e)))?;
            let shifted = timestamp_log_probs
                .broadcast_sub(&max_val)
                .map_err(|e| AudioError::TranscriptionFailed(format!("Broadcast sub: {}", e)))?;
            let exp_shifted = shifted
                .exp()
                .map_err(|e| AudioError::TranscriptionFailed(format!("Exp: {}", e)))?;
            let sum_exp = exp_shifted
                .sum(0)
                .map_err(|e| AudioError::TranscriptionFailed(format!("Sum: {}", e)))?;
            let log_sum = sum_exp
                .log()
                .map_err(|e| AudioError::TranscriptionFailed(format!("Log: {}", e)))?;
            max_val
                .broadcast_add(&log_sum)
                .map_err(|e| AudioError::TranscriptionFailed(format!("Broadcast add: {}", e)))?
                .to_scalar::<f32>()
                .map_err(|e| AudioError::TranscriptionFailed(format!("To scalar: {}", e)))?
        };

        let max_text_token_logprob: f32 = text_log_probs
            .max(0)
            .map_err(|e| AudioError::TranscriptionFailed(format!("Max: {}", e)))?
            .to_scalar::<f32>()
            .map_err(|e| AudioError::TranscriptionFailed(format!("To scalar: {}", e)))?;

        if timestamp_logprob > max_text_token_logprob {
            for i in 0..vocab_size {
                mask_buffer[i as usize] = if i < timestamp_begin {
                    f32::NEG_INFINITY
                } else {
                    0.0
                };
            }
            let mask_tensor = Tensor::new(mask_buffer.as_slice(), &device)
                .map_err(|e| AudioError::TranscriptionFailed(format!("Tensor creation: {}", e)))?;
            logits = logits
                .broadcast_add(&mask_tensor)
                .map_err(|e| AudioError::TranscriptionFailed(format!("Broadcast add: {}", e)))?;
        }

        Ok(logits)
    }

    /// Run decoding on full audio
    pub fn run(&mut self, mel: &Tensor, progress_callback: Option<ProgressCallback>) -> Result<Vec<Segment>> {
        let (_, _, content_frames) = mel
            .dims3()
            .map_err(|e| AudioError::TranscriptionFailed(format!("Dims3: {}", e)))?;
        let mut seek = 0;
        let mut segments = vec![];

        // Store progress info
        self.progress_callback = progress_callback;
        self.total_frames = content_frames;

        while seek < content_frames {

            let time_offset = (seek * m::HOP_LENGTH) as f64 / m::SAMPLE_RATE as f64;
            let segment_size = usize::min(content_frames - seek, m::N_FRAMES);
            let mel_segment = mel
                .narrow(2, seek, segment_size)
                .map_err(|e| AudioError::TranscriptionFailed(format!("Narrow: {}", e)))?;
            let segment_duration = (segment_size * m::HOP_LENGTH) as f64 / m::SAMPLE_RATE as f64;

            // Store segment info for progress tracking
            self.current_segment_start = seek;
            self.current_segment_size = segment_size;

            let dr = self.decode_with_fallback(&mel_segment)?;
            seek += segment_size;

            if dr.no_speech_prob > m::NO_SPEECH_THRESHOLD && dr.avg_logprob < m::LOGPROB_THRESHOLD {
                continue;
            }

            segments.push(Segment {
                start: time_offset,
                duration: segment_duration,
                dr,
            });
        }

        // Report 100% completion
        if let Some(ref callback) = self.progress_callback {
            callback(self.total_frames, self.total_frames);
        }

        Ok(segments)
    }

    /// Extract transcript segments with timestamps from decoder output
    pub fn extract_segments(&self, segments: Vec<Segment>) -> Vec<TranscriptSegment> {
        let mut result = Vec::new();

        for (seg_idx, segment) in segments.iter().enumerate() {
            if self.timestamps {
                let mut tokens_to_decode = vec![];
                let mut prev_timestamp_s = 0f32;
                let mut sub_segment_id = 0;

                for &token in segment.dr.tokens.iter() {
                    if token == self.sot_token || token == self.eot_token {
                        continue;
                    }

                    if token > self.no_timestamps_token {
                        let timestamp_s = (token - self.no_timestamps_token - 1) as f32 / 50.0;
                        if !tokens_to_decode.is_empty() {
                            if let Ok(text) = self.tokenizer.decode(&tokens_to_decode, true) {
                                result.push(TranscriptSegment {
                                    id: seg_idx * 100 + sub_segment_id,
                                    start: Some(segment.start + prev_timestamp_s as f64),
                                    end: Some(segment.start + timestamp_s as f64),
                                    text,
                                });
                                sub_segment_id += 1;
                            }
                            tokens_to_decode.clear();
                        }
                        prev_timestamp_s = timestamp_s;
                    } else {
                        tokens_to_decode.push(token);
                    }
                }

                if !tokens_to_decode.is_empty() {
                    if let Ok(text) = self.tokenizer.decode(&tokens_to_decode, true) {
                        if !text.trim().is_empty() {
                            result.push(TranscriptSegment {
                                id: seg_idx * 100 + sub_segment_id,
                                start: Some(segment.start + prev_timestamp_s as f64),
                                end: Some(segment.start + segment.duration),
                                text,
                            });
                        }
                    }
                }
            } else {
                result.push(TranscriptSegment {
                    id: seg_idx,
                    start: Some(segment.start),
                    end: Some(segment.start + segment.duration),
                    text: segment.dr.text.clone(),
                });
            }
        }

        result
    }
}

/// Get token ID from tokenizer
fn token_id(tokenizer: &Tokenizer, token: &str) -> Result<u32> {
    tokenizer
        .token_to_id(token)
        .ok_or_else(|| AudioError::TranscriptionFailed(format!("No token ID for: {}", token)))
}
