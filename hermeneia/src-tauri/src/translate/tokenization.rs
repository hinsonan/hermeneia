use crate::error::{AudioError, Result};
use crate::translate::types::TranslationModel;
use std::path::Path;
use tokenizers::Tokenizer;

/// Tokenizer wrapper for translation models
pub struct TranslationTokenizer {
    inner: TokenizerInner,
    model_type: TranslationModel,
}

/// Internal enum to support both HuggingFace and MarianMT tokenizers
enum TokenizerInner {
    HuggingFace(Tokenizer),
    Marian(rust_tokenizers::tokenizer::MarianTokenizer),
}

impl TranslationTokenizer {
    /// Load tokenizer from file (for T5/mT5 models)
    pub fn from_file(path: &Path, model_type: TranslationModel) -> Result<Self> {
        let mut tokenizer = Tokenizer::from_file(path).map_err(|e| {
            AudioError::TokenizationError(format!("Failed to load tokenizer: {}", e))
        })?;

        tokenizer.with_padding(None);
        let _ = tokenizer.with_truncation(None);

        Ok(Self {
            inner: TokenizerInner::HuggingFace(tokenizer),
            model_type,
        })
    }

    /// Load MarianMT tokenizer from vocab and SentencePiece model
    pub fn from_marian_files(
        vocab_path: &Path,
        spm_path: &Path,
        model_type: TranslationModel,
    ) -> Result<Self> {
        use rust_tokenizers::tokenizer::MarianTokenizer;

        let tokenizer = MarianTokenizer::from_files(
            vocab_path
                .to_str()
                .ok_or_else(|| AudioError::TokenizationError("Invalid vocab path".to_string()))?,
            spm_path
                .to_str()
                .ok_or_else(|| AudioError::TokenizationError("Invalid spm path".to_string()))?,
            false, // lowercase
        )
        .map_err(|e| {
            AudioError::TokenizationError(format!("Failed to load MarianMT tokenizer: {}", e))
        })?;

        Ok(Self {
            inner: TokenizerInner::Marian(tokenizer),
            model_type,
        })
    }

    /// Encode input text with appropriate task prefix
    ///
    /// For MADLAD: Prepends "<2{target_lang}> {text}" (e.g., "<2de> Hello" for translation to German)
    /// For MarianMT: Direct encoding (no prefix)
    pub fn encode(&self, text: &str, source_lang: &str, target_lang: &str) -> Result<Vec<u32>> {
        match &self.inner {
            TokenizerInner::HuggingFace(tokenizer) => {
                let input = if self.model_type.is_madlad() {
                    // MADLAD models use target language prefix: "<2{lang}> {text}"
                    self.format_madlad_input(text, target_lang)
                } else if self.model_type.is_t5() {
                    // T5-family models use task prompts for translation
                    self.format_t5_input(text, source_lang, target_lang)
                } else {
                    text.to_string()
                };

                let encoding = tokenizer.encode(input, true).map_err(|e| {
                    AudioError::TokenizationError(format!("Encoding failed: {}", e))
                })?;

                Ok(encoding.get_ids().to_vec())
            }
            TokenizerInner::Marian(tokenizer) => {
                use rust_tokenizers::tokenizer::Tokenizer as RustTokenizer;
                use rust_tokenizers::tokenizer::TruncationStrategy;

                // MarianMT encodes directly without task prefix
                let tokens =
                    tokenizer.encode(text, None, 512, &TruncationStrategy::LongestFirst, 0);

                Ok(tokens.token_ids.iter().map(|&id| id as u32).collect())
            }
        }
    }

    /// Decode token IDs back to text
    pub fn decode(&self, token_ids: &[u32]) -> Result<String> {
        match &self.inner {
            TokenizerInner::HuggingFace(tokenizer) => {
                let text = tokenizer
                    .decode(token_ids, true) // skip_special_tokens = true
                    .map_err(|e| {
                        AudioError::TokenizationError(format!("Decoding failed: {}", e))
                    })?;

                Ok(text)
            }
            TokenizerInner::Marian(tokenizer) => {
                use rust_tokenizers::tokenizer::Tokenizer as RustTokenizer;

                let ids: Vec<i64> = token_ids.iter().map(|&id| id as i64).collect();
                let text = tokenizer.decode(&ids, true, true); // skip_special_tokens, clean_up_tokenization_spaces

                Ok(text)
            }
        }
    }

    /// Format input for MADLAD models with target language prefix
    fn format_madlad_input(&self, text: &str, target_lang: &str) -> String {
        // MADLAD models use target language prefixes like:
        // "<2de> {text}" for translation to German
        format!("<2{}> {}", target_lang, text)
    }

    /// Format input for T5-family translation models
    ///
    /// Example: "translate to German: A beautiful candle."
    fn format_t5_input(&self, text: &str, source_lang: &str, target_lang: &str) -> String {
        let _ = source_lang;
        let target_name = language_name(target_lang);
        format!("translate to {}: {}", target_name, text)
    }
    /// Get special token IDs
    pub fn get_bos_token_id(&self) -> Option<u32> {
        match &self.inner {
            TokenizerInner::HuggingFace(tokenizer) => tokenizer.get_vocab(true).get("<s>").copied(),
            TokenizerInner::Marian(_) => {
                // MarianMT typically uses </s> as BOS
                Some(0)
            }
        }
    }

    pub fn get_eos_token_id(&self) -> Option<u32> {
        match &self.inner {
            TokenizerInner::HuggingFace(tokenizer) => {
                tokenizer.get_vocab(true).get("</s>").copied()
            }
            TokenizerInner::Marian(_) => {
                // MarianMT uses </s> (ID 0) as EOS
                Some(0)
            }
        }
    }

    pub fn get_pad_token_id(&self) -> Option<u32> {
        match &self.inner {
            TokenizerInner::HuggingFace(tokenizer) => {
                let vocab = tokenizer.get_vocab(true);
                vocab.get("<pad>").or_else(|| vocab.get("[PAD]")).copied()
            }
            TokenizerInner::Marian(_) => {
                // MarianMT uses </s> as pad token
                Some(0)
            }
        }
    }
}

fn language_name(code: &str) -> String {
    match code.to_lowercase().as_str() {
        "en" => "English".to_string(),
        "es" => "Spanish".to_string(),
        "fr" => "French".to_string(),
        "de" => "German".to_string(),
        "pt" => "Portuguese".to_string(),
        "it" => "Italian".to_string(),
        "ru" => "Russian".to_string(),
        "zh" => "Chinese".to_string(),
        "ja" => "Japanese".to_string(),
        "ko" => "Korean".to_string(),
        "ar" => "Arabic".to_string(),
        _ => code.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_madlad_input_formatting() {
        // Use a simple dummy tokenizer
        use tokenizers::models::bpe::BPE;
        let model = BPE::default();
        let tokenizer = Tokenizer::new(model);

        let tok = TranslationTokenizer {
            inner: TokenizerInner::HuggingFace(tokenizer),
            model_type: TranslationModel::Madlad3B,
        };

        let formatted = tok.format_madlad_input("Hello world", "de");
        assert_eq!(formatted, "<2de> Hello world");

        let formatted = tok.format_madlad_input("Bonjour", "en");
        assert_eq!(formatted, "<2en> Bonjour");
    }
}
