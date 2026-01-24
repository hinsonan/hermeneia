/// Translation service for Hermeneia
///
/// This module provides text-to-text neural machine translation using
/// encoder-decoder models from HuggingFace (MADLAD-400, MarianMT).

// Public modules
pub mod generator;
pub mod inference;
pub mod language;
pub mod logits_processor;
pub mod model;
pub mod tokenization;
pub mod types;

// Re-export commonly used types
pub use types::{TranslateParams, TranslationModel, TranslationResult};

// Re-export main API functions
pub use inference::{translate_text, translate_text_with_progress};
