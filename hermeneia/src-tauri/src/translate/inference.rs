use crate::error::{AudioError, Result};
use crate::translate::{
    generator::{GenerationConfig, Generator},
    model::{get_device, ModelFiles, ModelManager},
    tokenization::TranslationTokenizer,
    types::{ProgressCallback, TranslateParams, TranslationModel, TranslationResult},
};
use candle_core::{Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::{marian, t5};
use std::time::Instant;

/// Callback for batch translation progress: (current_segment, total_segments, segment_text)
pub type BatchProgressCallback = Box<dyn Fn(usize, usize, &str) + Send + Sync>;

/// Enum to handle different model architectures
enum TranslationModelType {
    /// T5-based models (MADLAD-400 uses T5 architecture)
    T5 {
        model: t5::T5ForConditionalGeneration,
        config: t5::Config,
    },
    Marian {
        model: marian::MTModel,
        config: marian::Config,
    },
}

/// Get a human-readable name for the device
fn device_name(device: &Device) -> &'static str {
    match device {
        Device::Cpu => "CPU",
        Device::Cuda(_) => "CUDA",
        Device::Metal(_) => "Metal",
    }
}

/// Main translation function - simple API
pub fn translate_text(text: &str, params: TranslateParams) -> Result<TranslationResult> {
    translate_text_with_progress(text, params, None)
}

/// Translation with optional progress callback for CLI
pub fn translate_text_with_progress(
    text: &str,
    params: TranslateParams,
    progress_callback: Option<ProgressCallback>,
) -> Result<TranslationResult> {
    let start_time = Instant::now();

    tracing::info!(
        "Starting translation: {} -> {}",
        params.source_language,
        params.target_language
    );

    // 1. Select best model
    let model_manager = ModelManager::new()?;
    let selected_model = model_manager.select_model(&params)?;
    tracing::info!("Selected model: {}", selected_model.display_name());

    // 2. Download/load model (on-demand, cached for future use)
    let model_files = model_manager.ensure_model(selected_model, params.use_quantized)?;
    let device = get_device(params.force_cpu)?;
    tracing::info!("Using device: {}", device_name(&device));

    // 3. Load model and tokenizer
    let (tokenizer, mut model) = load_model(&model_files, selected_model, &device)?;

    // 4. Tokenize input
    let input_ids = tokenizer.encode(text, &params.source_language, &params.target_language)?;

    tracing::info!("Input tokenized: {} tokens", input_ids.len());
    tracing::debug!("Input token IDs: {:?}", input_ids);

    // 5. Generate translation
    let (output_ids, decoder_start_token_id) = generate_translation(
        &mut model,
        &input_ids,
        &params,
        &device,
        progress_callback.as_ref(),
    )?;

    tracing::info!("Generated {} tokens", output_ids.len());
    tracing::debug!("Output token IDs: {:?}", output_ids);

    // 6. Decode output
    // Skip the decoder start token (first token) which is used to prime the decoder
    // but is not part of the actual translation output
    let tokens_to_decode = if !output_ids.is_empty() && output_ids[0] == decoder_start_token_id {
        tracing::debug!(
            "Skipping decoder start token {} from output",
            decoder_start_token_id
        );
        &output_ids[1..]
    } else {
        &output_ids
    };
    let translated_text = tokenizer.decode(tokens_to_decode)?;

    let inference_time = start_time.elapsed().as_secs_f64();
    tracing::info!("Translation completed in {:.2}s", inference_time);

    Ok(TranslationResult {
        translated_text,
        source_language: params.source_language,
        target_language: params.target_language,
        model_used: selected_model,
        inference_time,
        token_count: output_ids.len(),
    })
}

/// Load model and tokenizer from files
fn load_model(
    model_files: &ModelFiles,
    model_type: TranslationModel,
    device: &Device,
) -> Result<(TranslationTokenizer, TranslationModelType)> {
    tracing::info!("Loading tokenizer...");
    let tokenizer = if model_type.is_marian() {
        // Load MarianMT tokenizer
        let spm_path = model_files
            .spm_model
            .as_ref()
            .ok_or_else(|| AudioError::ModelLoad {
                model: model_type.model_id().to_string(),
                details: "SentencePiece model path missing for MarianMT".to_string(),
            })?;
        TranslationTokenizer::from_marian_files(&model_files.tokenizer, spm_path, model_type)?
    } else {
        // Load MADLAD tokenizer (T5-based)
        TranslationTokenizer::from_file(&model_files.tokenizer, model_type)?
    };

    tracing::info!("Loading model weights...");
    let vb = if model_files.weights.extension().and_then(|s| s.to_str()) == Some("safetensors") {
        unsafe {
            VarBuilder::from_mmaped_safetensors(
                &[model_files.weights.clone()],
                candle_core::DType::F32,
                device,
            )
            .map_err(|e| AudioError::ModelLoad {
                model: model_type.model_id().to_string(),
                details: format!("Failed to load safetensors: {}", e),
            })?
        }
    } else {
        // Load pytorch pickle-based checkpoint using candle's pickle reader
        tracing::info!("Loading pytorch pickle checkpoint...");
        let tensors = candle_core::pickle::read_all(&model_files.weights).map_err(|e| {
            AudioError::ModelLoad {
                model: model_type.model_id().to_string(),
                details: format!("Failed to read pytorch pickle: {}", e),
            }
        })?;

        // Convert Vec to HashMap for VarBuilder
        let tensor_map: std::collections::HashMap<String, Tensor> = tensors.into_iter().collect();
        tracing::info!(
            "Loaded {} tensors from pytorch checkpoint",
            tensor_map.len()
        );

        VarBuilder::from_tensors(tensor_map, candle_core::DType::F32, device)
    };

    tracing::info!("Loading model config...");
    tracing::info!("Initializing model architecture...");

    let model_wrapper = if model_type.is_marian() {
        // Load Marian model
        let config_file = std::fs::File::open(&model_files.config)?;
        let mut config_value: serde_json::Value =
            serde_json::from_reader(config_file).map_err(|e| AudioError::ModelLoad {
                model: model_type.model_id().to_string(),
                details: format!("Failed to parse Marian config JSON: {}", e),
            })?;
        if let Some(obj) = config_value.as_object_mut() {
            if !obj.contains_key("share_encoder_decoder_embeddings") {
                obj.insert(
                    "share_encoder_decoder_embeddings".to_string(),
                    serde_json::Value::Bool(true),
                );
            }
        } else {
            return Err(AudioError::ModelLoad {
                model: model_type.model_id().to_string(),
                details: "Marian config JSON is not an object".to_string(),
            });
        }
        let config: marian::Config =
            serde_json::from_value(config_value).map_err(|e| AudioError::ModelLoad {
                model: model_type.model_id().to_string(),
                details: format!("Failed to parse Marian config: {}", e),
            })?;
        let model = marian::MTModel::new(&config, vb).map_err(|e| AudioError::ModelLoad {
            model: model_type.model_id().to_string(),
            details: format!("Failed to initialize Marian model: {}", e),
        })?;
        TranslationModelType::Marian { model, config }
    } else {
        // Load T5-based model (MADLAD-400 uses T5 architecture)
        let config: t5::Config = serde_json::from_reader(std::fs::File::open(&model_files.config)?)
            .map_err(|e| AudioError::ModelLoad {
                model: model_type.model_id().to_string(),
                details: format!("Failed to parse T5 config: {}", e),
            })?;
        let model = t5::T5ForConditionalGeneration::load(vb, &config).map_err(|e| {
            AudioError::ModelLoad {
                model: model_type.model_id().to_string(),
                details: format!("Failed to initialize T5-based model: {}", e),
            }
        })?;
        TranslationModelType::T5 { model, config }
    };

    Ok((tokenizer, model_wrapper))
}

/// Generate translation using encoder-decoder architecture
/// Returns (output_ids, decoder_start_token_id)
fn generate_translation(
    model: &mut TranslationModelType,
    input_ids: &[u32],
    params: &TranslateParams,
    device: &Device,
    progress_callback: Option<&ProgressCallback>,
) -> Result<(Vec<u32>, u32)> {
    // Create input tensor
    let input_tensor = Tensor::new(input_ids, device)
        .map_err(|e| {
            AudioError::TranslationFailed(format!("Failed to create input tensor: {}", e))
        })?
        .unsqueeze(0)
        .map_err(|e| AudioError::TranslationFailed(format!("Failed to unsqueeze input: {}", e)))?;

    match model {
        TranslationModelType::T5 {
            model: t5_model,
            config,
        } => {
            tracing::info!("Running T5-based encoder (MADLAD)...");
            let encoder_output = t5_model
                .encode(&input_tensor)
                .map_err(|e| AudioError::TranslationFailed(format!("Encoder failed: {}", e)))?;

            // Get special tokens
            let decoder_start_token_id =
                config.decoder_start_token_id.unwrap_or(config.pad_token_id) as u32;
            let eos_token_id = config.eos_token_id as u32;
            let use_cache = config.use_cache;

            tracing::info!(
                "Decoder start token: {}, EOS token: {}",
                decoder_start_token_id,
                eos_token_id
            );

            // Configure generation
            let gen_config = GenerationConfig {
                max_length: params.max_length.unwrap_or(512),
                temperature: params.temperature.unwrap_or(1.0),
                top_p: params.top_p,
                repetition_penalty: params.repetition_penalty.unwrap_or(1.2),
            };

            tracing::info!(
                "Starting generation (max {} tokens)...",
                gen_config.max_length
            );

            let mut generator = Generator::new(gen_config);
            let output_ids = generator.generate(
                t5_model,
                &encoder_output,
                decoder_start_token_id,
                eos_token_id,
                use_cache,
                progress_callback,
            )?;

            Ok((output_ids, decoder_start_token_id))
        }
        TranslationModelType::Marian {
            model: marian_model,
            config,
        } => {
            tracing::info!("Running Marian encoder...");
            let encoder_output = marian_model
                .encoder()
                .forward(&input_tensor, 0)
                .map_err(|e| {
                    AudioError::TranslationFailed(format!("Marian encoder failed: {}", e))
                })?;

            // Get special tokens from config (MarianMT uses decoder_start_token_id from config)
            // For Helsinki-NLP MarianMT models: decoder_start_token_id=65000, eos_token_id=0
            let decoder_start_token_id = config.decoder_start_token_id;
            let eos_token_id = config.eos_token_id;

            tracing::info!(
                "Decoder start token: {}, EOS token: {}",
                decoder_start_token_id,
                eos_token_id
            );

            // Greedy decoding with KV cache for Marian
            let max_length = params.max_length.unwrap_or(512);
            tracing::info!("Starting greedy decoding (max {} tokens)...", max_length);

            let mut token_ids = vec![decoder_start_token_id];

            for index in 0..max_length {
                // First iteration: full sequence, subsequent: only last token (KV cache)
                let context_size = if index >= 1 { 1 } else { token_ids.len() };
                let start_pos = token_ids.len().saturating_sub(context_size);

                let input_slice = &token_ids[start_pos..];
                let decoder_input = Tensor::new(input_slice, device)
                    .map_err(|e| {
                        AudioError::TranslationFailed(format!(
                            "Failed to create decoder input: {}",
                            e
                        ))
                    })?
                    .unsqueeze(0)
                    .map_err(|e| {
                        AudioError::TranslationFailed(format!("Failed to unsqueeze: {}", e))
                    })?;

                let logits = marian_model
                    .decode(&decoder_input, &encoder_output, start_pos)
                    .map_err(|e| AudioError::TranslationFailed(format!("Decoder failed: {}", e)))?;

                // Get logits for last position [batch_size, seq_len, vocab_size]
                let seq_len = logits.dim(1).map_err(|e| {
                    AudioError::TranslationFailed(format!("Failed to get seq_len: {}", e))
                })?;
                let last_logits = logits
                    .get(0)
                    .map_err(|e| {
                        AudioError::TranslationFailed(format!("Failed to get batch: {}", e))
                    })?
                    .get(seq_len - 1)
                    .map_err(|e| {
                        AudioError::TranslationFailed(format!("Failed to get last token: {}", e))
                    })?;

                let next_token = last_logits
                    .argmax(0)
                    .map_err(|e| AudioError::TranslationFailed(format!("Argmax failed: {}", e)))?
                    .to_scalar::<u32>()
                    .map_err(|e| {
                        AudioError::TranslationFailed(format!("Failed to convert to scalar: {}", e))
                    })?;

                token_ids.push(next_token);

                if next_token == eos_token_id || next_token == config.forced_eos_token_id {
                    break;
                }

                if let Some(callback) = progress_callback {
                    callback(index + 1, max_length);
                }
            }

            let output_ids = token_ids;

            tracing::info!("Generated {} tokens", output_ids.len());
            Ok((output_ids, decoder_start_token_id))
        }
    }
}

// ============================================================================
// Translator struct for batch translation (model loaded once)
// ============================================================================

/// A translator that holds a loaded model for efficient batch translation
///
/// This avoids reloading the model for each text segment, which is essential
/// for translating SRT files with many segments.
pub struct Translator {
    tokenizer: TranslationTokenizer,
    model: TranslationModelType,
    device: Device,
    selected_model: TranslationModel,
    params: TranslateParams,
}

impl Translator {
    /// Create a new translator by loading the model for the given parameters
    pub fn new(params: TranslateParams) -> Result<Self> {
        tracing::info!(
            "Initializing translator: {} -> {}",
            params.source_language,
            params.target_language
        );

        // 1. Select best model
        let model_manager = ModelManager::new()?;
        let selected_model = model_manager.select_model(&params)?;
        tracing::info!("Selected model: {}", selected_model.display_name());

        // 2. Download/load model (on-demand, cached for future use)
        let model_files = model_manager.ensure_model(selected_model, params.use_quantized)?;
        let device = get_device(params.force_cpu)?;
        tracing::info!("Using device: {}", device_name(&device));

        // 3. Load model and tokenizer
        let (tokenizer, model) = load_model(&model_files, selected_model, &device)?;

        Ok(Self {
            tokenizer,
            model,
            device,
            selected_model,
            params,
        })
    }

    /// Get the model that was selected
    pub fn model_used(&self) -> TranslationModel {
        self.selected_model
    }

    /// Translate a single text using the loaded model
    pub fn translate(&mut self, text: &str) -> Result<String> {
        let start_time = Instant::now();

        // Reset KV cache before each translation to avoid interference
        // from previous translations when reusing the model
        self.reset_kv_cache();

        // 1. Tokenize input
        let input_ids = self.tokenizer.encode(
            text,
            &self.params.source_language,
            &self.params.target_language,
        )?;

        tracing::debug!("Input tokenized: {} tokens", input_ids.len());

        // 2. Generate translation
        let (output_ids, decoder_start_token_id) = generate_translation(
            &mut self.model,
            &input_ids,
            &self.params,
            &self.device,
            None,
        )?;

        tracing::debug!("Generated {} tokens", output_ids.len());

        // 3. Decode output (skip decoder start token)
        let tokens_to_decode = if !output_ids.is_empty() && output_ids[0] == decoder_start_token_id
        {
            &output_ids[1..]
        } else {
            &output_ids
        };
        let translated_text = self.tokenizer.decode(tokens_to_decode)?;

        let inference_time = start_time.elapsed().as_secs_f64();
        tracing::debug!("Segment translated in {:.2}s", inference_time);

        Ok(translated_text)
    }

    /// Reset the KV cache for decoder models
    /// This must be called between translations when reusing the same model
    fn reset_kv_cache(&mut self) {
        match &mut self.model {
            TranslationModelType::Marian { model, .. } => {
                model.reset_kv_cache();
            }
            TranslationModelType::T5 { model, .. } => {
                model.clear_kv_cache();
            }
        }
    }

    /// Translate multiple texts efficiently (model loaded once)
    ///
    /// This is the recommended way to translate SRT segments or multiple paragraphs.
    pub fn translate_batch(
        &mut self,
        texts: &[String],
        progress_callback: Option<&BatchProgressCallback>,
    ) -> Result<Vec<String>> {
        let total = texts.len();
        let mut results = Vec::with_capacity(total);

        for (i, text) in texts.iter().enumerate() {
            if let Some(callback) = progress_callback {
                callback(i + 1, total, text);
            }

            let translated = self.translate(text)?;
            results.push(translated);
        }

        Ok(results)
    }
}

/// Translate multiple texts with a single model load
///
/// This is the main API for batch translation - use this for SRT files
/// or any case where you need to translate multiple texts with the same
/// language pair.
pub fn translate_texts_batch(
    texts: &[String],
    params: TranslateParams,
    progress_callback: Option<BatchProgressCallback>,
) -> Result<(Vec<String>, TranslationModel, f64)> {
    let start_time = Instant::now();

    // Create translator (loads model once)
    let mut translator = Translator::new(params)?;

    // Translate all texts
    let results = translator.translate_batch(texts, progress_callback.as_ref())?;

    let total_time = start_time.elapsed().as_secs_f64();
    let model_used = translator.model_used();

    tracing::info!(
        "Batch translation complete: {} segments in {:.2}s ({:.2}s/segment avg)",
        texts.len(),
        total_time,
        total_time / texts.len() as f64
    );

    Ok((results, model_used, total_time))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_device_name() {
        let cpu = Device::Cpu;
        assert_eq!(device_name(&cpu), "CPU");
    }
}
