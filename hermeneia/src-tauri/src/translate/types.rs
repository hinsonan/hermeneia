use serde::{Deserialize, Serialize};

/// Available translation models from HuggingFace
///
/// All models listed here are fine-tuned for translation and ready to use:
/// - **MarianMT models**: Specialized for specific language pairs (fastest, best quality for supported pairs)
/// - **MADLAD-400 models**: Multilingual models supporting 450+ languages (use for unsupported pairs)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum TranslationModel {
    // Multilingual models (450+ languages)
    #[serde(rename = "madlad-3b")]
    Madlad3B,
    #[serde(rename = "madlad-7b")]
    Madlad7B,
    #[serde(rename = "madlad-10b")]
    Madlad10B,

    // T5 models (general text-to-text, translation via prompt)
    #[serde(rename = "t5-small")]
    T5Small,
    #[serde(rename = "t5-base")]
    T5Base,
    #[serde(rename = "t5-large")]
    T5Large,
    #[serde(rename = "flan-t5-small")]
    FlanT5Small,
    #[serde(rename = "flan-t5-base")]
    FlanT5Base,
    #[serde(rename = "flan-t5-large")]
    FlanT5Large,
    #[serde(rename = "flan-ul2")]
    FlanUl2,

    // Specialized MarianMT models (specific language pairs - fastest)
    #[serde(rename = "marian-en-es")]
    MarianEnEs,
    #[serde(rename = "marian-es-en")]
    MarianEsEn,
    #[serde(rename = "marian-en-fr")]
    MarianEnFr,
    #[serde(rename = "marian-fr-en")]
    MarianFrEn,
    #[serde(rename = "marian-en-de")]
    MarianEnDe,
    #[serde(rename = "marian-de-en")]
    MarianDeEn,
    #[serde(rename = "marian-en-pt")]
    MarianEnPt,
    #[serde(rename = "marian-pt-en")]
    MarianPtEn,
    #[serde(rename = "marian-en-it")]
    MarianEnIt,
    #[serde(rename = "marian-it-en")]
    MarianItEn,
    #[serde(rename = "marian-en-ru")]
    MarianEnRu,
    #[serde(rename = "marian-ru-en")]
    MarianRuEn,
    #[serde(rename = "marian-en-zh")]
    MarianEnZh,
    #[serde(rename = "marian-zh-en")]
    MarianZhEn,
    #[serde(rename = "marian-en-ja")]
    MarianEnJa,
    #[serde(rename = "marian-ja-en")]
    MarianJaEn,
    #[serde(rename = "marian-en-ko")]
    MarianEnKo,
    #[serde(rename = "marian-ko-en")]
    MarianKoEn,
    #[serde(rename = "marian-en-ar")]
    MarianEnAr,
    #[serde(rename = "marian-ar-en")]
    MarianArEn,
}

impl TranslationModel {
    /// Returns the HuggingFace model ID for downloading
    pub fn model_id(&self) -> &'static str {
        match self {
            // MADLAD-400 multilingual models
            Self::Madlad3B => "jbochi/madlad400-3b-mt",
            Self::Madlad7B => "jbochi/madlad400-7b-mt",
            Self::Madlad10B => "jbochi/madlad400-10b-mt",

            // T5 family models
            Self::T5Small => "t5-small",
            Self::T5Base => "t5-base",
            Self::T5Large => "t5-large",
            Self::FlanT5Small => "google/flan-t5-small",
            Self::FlanT5Base => "google/flan-t5-base",
            Self::FlanT5Large => "google/flan-t5-large",
            Self::FlanUl2 => "google/flan-ul2",

            // MarianMT specialized models
            Self::MarianEnEs => "Helsinki-NLP/opus-mt-en-es",
            Self::MarianEsEn => "Helsinki-NLP/opus-mt-es-en",
            Self::MarianEnFr => "Helsinki-NLP/opus-mt-en-fr",
            Self::MarianFrEn => "Helsinki-NLP/opus-mt-fr-en",
            Self::MarianEnDe => "Helsinki-NLP/opus-mt-en-de",
            Self::MarianDeEn => "Helsinki-NLP/opus-mt-de-en",
            Self::MarianEnPt => "Helsinki-NLP/opus-mt-en-roa",
            Self::MarianPtEn => "Helsinki-NLP/opus-mt-roa-en",
            Self::MarianEnIt => "Helsinki-NLP/opus-mt-en-it",
            Self::MarianItEn => "Helsinki-NLP/opus-mt-it-en",
            Self::MarianEnRu => "Helsinki-NLP/opus-mt-en-ru",
            Self::MarianRuEn => "Helsinki-NLP/opus-mt-ru-en",
            Self::MarianEnZh => "Helsinki-NLP/opus-mt-en-zh",
            Self::MarianZhEn => "Helsinki-NLP/opus-mt-zh-en",
            Self::MarianEnJa => "Helsinki-NLP/opus-mt-en-jap",
            Self::MarianJaEn => "Helsinki-NLP/opus-mt-jap-en",
            Self::MarianEnKo => "Helsinki-NLP/opus-mt-en-ko",
            Self::MarianKoEn => "Helsinki-NLP/opus-mt-ko-en",
            Self::MarianEnAr => "Helsinki-NLP/opus-mt-en-ar",
            Self::MarianArEn => "Helsinki-NLP/opus-mt-ar-en",
        }
    }

    /// Returns whether this is a multilingual model (supports any language pair)
    pub fn is_multilingual(&self) -> bool {
        matches!(self, Self::Madlad3B | Self::Madlad7B | Self::Madlad10B)
    }

    /// Returns whether this is a MADLAD model
    pub fn is_madlad(&self) -> bool {
        matches!(self, Self::Madlad3B | Self::Madlad7B | Self::Madlad10B)
    }

    /// Returns whether this is a MarianMT model
    pub fn is_marian(&self) -> bool {
        matches!(
            self,
            Self::MarianEnEs
                | Self::MarianEsEn
                | Self::MarianEnFr
                | Self::MarianFrEn
                | Self::MarianEnDe
                | Self::MarianDeEn
                | Self::MarianEnPt
                | Self::MarianPtEn
                | Self::MarianEnIt
                | Self::MarianItEn
                | Self::MarianEnRu
                | Self::MarianRuEn
                | Self::MarianEnZh
                | Self::MarianZhEn
                | Self::MarianEnJa
                | Self::MarianJaEn
                | Self::MarianEnKo
                | Self::MarianKoEn
                | Self::MarianEnAr
                | Self::MarianArEn
        )
    }

    /// Returns whether this is a T5-family model (non-MADLAD, non-Marian)
    pub fn is_t5(&self) -> bool {
        matches!(
            self,
            Self::T5Small
                | Self::T5Base
                | Self::T5Large
                | Self::FlanT5Small
                | Self::FlanT5Base
                | Self::FlanT5Large
                | Self::FlanUl2
        )
    }

    /// Returns the HuggingFace revision/branch that contains safetensors files
    /// Most models on main branch don't have safetensors, they're in PR branches
    pub fn safetensors_revision(&self) -> Option<&'static str> {
        match self {
            // MADLAD models - check if main branch has safetensors
            Self::Madlad3B | Self::Madlad7B | Self::Madlad10B => None, // main branch should have safetensors

            // Flan-UL2 uses a PR revision for safetensors in candle example
            Self::FlanUl2 => Some("refs/pr/25"),

            // All MarianMT models use refs/pr/4 for safetensors
            _ if self.is_marian() => Some("refs/pr/4"),

            _ => None,
        }
    }

    /// Returns the source and target languages for specialized models
    /// Returns None for multilingual models (they support all pairs)
    pub fn language_pair(&self) -> Option<(&'static str, &'static str)> {
        match self {
            // MarianMT models have fixed language pairs
            Self::MarianEnEs => Some(("en", "es")),
            Self::MarianEsEn => Some(("es", "en")),
            Self::MarianEnFr => Some(("en", "fr")),
            Self::MarianFrEn => Some(("fr", "en")),
            Self::MarianEnDe => Some(("en", "de")),
            Self::MarianDeEn => Some(("de", "en")),
            Self::MarianEnPt => Some(("en", "pt")),
            Self::MarianPtEn => Some(("pt", "en")),
            Self::MarianEnIt => Some(("en", "it")),
            Self::MarianItEn => Some(("it", "en")),
            Self::MarianEnRu => Some(("en", "ru")),
            Self::MarianRuEn => Some(("ru", "en")),
            Self::MarianEnZh => Some(("en", "zh")),
            Self::MarianZhEn => Some(("zh", "en")),
            Self::MarianEnJa => Some(("en", "ja")),
            Self::MarianJaEn => Some(("ja", "en")),
            Self::MarianEnKo => Some(("en", "ko")),
            Self::MarianKoEn => Some(("ko", "en")),
            Self::MarianEnAr => Some(("en", "ar")),
            Self::MarianArEn => Some(("ar", "en")),

            // Multilingual models support any pair
            _ => None,
        }
    }

    /// Check if this model supports the given language pair
    pub fn supports_pair(&self, source: &str, target: &str) -> bool {
        if self.is_marian() {
            if let Some((model_src, model_tgt)) = self.language_pair() {
                model_src == source && model_tgt == target
            } else {
                false
            }
        } else {
            // T5-family and MADLAD models accept any pair
            let _ = (source, target);
            true
        }
    }

    /// Returns approximate model size in MB
    pub fn approx_size_mb(&self) -> u64 {
        match self {
            Self::Madlad3B => 3000,  // ~3GB
            Self::Madlad7B => 7000,  // ~7GB
            Self::Madlad10B => 10000, // ~10GB
            Self::T5Small => 242,
            Self::T5Base => 892,
            Self::T5Large => 2950,
            Self::FlanT5Small => 248,
            Self::FlanT5Base => 909,
            Self::FlanT5Large => 3000,
            Self::FlanUl2 => 20000,
            // MarianMT models are all similar size
            _ if self.is_marian() => 298,
            _ => 0,
        }
    }

    /// Returns a human-friendly display name
    pub fn display_name(&self) -> &'static str {
        match self {
            Self::Madlad3B => "MADLAD-400 3B (3GB, 450+ languages)",
            Self::Madlad7B => "MADLAD-400 7B (7GB, 450+ languages)",
            Self::Madlad10B => "MADLAD-400 10B (10GB, 450+ languages)",
            Self::T5Small => "T5 Small (242MB)",
            Self::T5Base => "T5 Base (892MB)",
            Self::T5Large => "T5 Large (2.95GB)",
            Self::FlanT5Small => "Flan-T5 Small (248MB)",
            Self::FlanT5Base => "Flan-T5 Base (909MB)",
            Self::FlanT5Large => "Flan-T5 Large (3GB)",
            Self::FlanUl2 => "Flan-UL2 (20GB)",
            Self::MarianEnEs => "MarianMT EN→ES (298MB)",
            Self::MarianEsEn => "MarianMT ES→EN (298MB)",
            Self::MarianEnFr => "MarianMT EN→FR (298MB)",
            Self::MarianFrEn => "MarianMT FR→EN (298MB)",
            Self::MarianEnDe => "MarianMT EN→DE (298MB)",
            Self::MarianDeEn => "MarianMT DE→EN (298MB)",
            Self::MarianEnPt => "MarianMT EN→PT (298MB)",
            Self::MarianPtEn => "MarianMT PT→EN (298MB)",
            Self::MarianEnIt => "MarianMT EN→IT (298MB)",
            Self::MarianItEn => "MarianMT IT→EN (298MB)",
            Self::MarianEnRu => "MarianMT EN→RU (298MB)",
            Self::MarianRuEn => "MarianMT RU→EN (298MB)",
            Self::MarianEnZh => "MarianMT EN→ZH (298MB)",
            Self::MarianZhEn => "MarianMT ZH→EN (298MB)",
            Self::MarianEnJa => "MarianMT EN→JA (298MB)",
            Self::MarianJaEn => "MarianMT JA→EN (298MB)",
            Self::MarianEnKo => "MarianMT EN→KO (298MB)",
            Self::MarianKoEn => "MarianMT KO→EN (298MB)",
            Self::MarianEnAr => "MarianMT EN→AR (298MB)",
            Self::MarianArEn => "MarianMT AR→EN (298MB)",
        }
    }

    /// Returns CLI key for --model flag
    pub fn cli_key(&self) -> &'static str {
        match self {
            Self::Madlad3B => "madlad-3b",
            Self::Madlad7B => "madlad-7b",
            Self::Madlad10B => "madlad-10b",
            Self::T5Small => "t5-small",
            Self::T5Base => "t5-base",
            Self::T5Large => "t5-large",
            Self::FlanT5Small => "flan-t5-small",
            Self::FlanT5Base => "flan-t5-base",
            Self::FlanT5Large => "flan-t5-large",
            Self::FlanUl2 => "flan-ul2",
            Self::MarianEnEs => "marian-en-es",
            Self::MarianEsEn => "marian-es-en",
            Self::MarianEnFr => "marian-en-fr",
            Self::MarianFrEn => "marian-fr-en",
            Self::MarianEnDe => "marian-en-de",
            Self::MarianDeEn => "marian-de-en",
            Self::MarianEnPt => "marian-en-pt",
            Self::MarianPtEn => "marian-pt-en",
            Self::MarianEnIt => "marian-en-it",
            Self::MarianItEn => "marian-it-en",
            Self::MarianEnRu => "marian-en-ru",
            Self::MarianRuEn => "marian-ru-en",
            Self::MarianEnZh => "marian-en-zh",
            Self::MarianZhEn => "marian-zh-en",
            Self::MarianEnJa => "marian-en-ja",
            Self::MarianJaEn => "marian-ja-en",
            Self::MarianEnKo => "marian-en-ko",
            Self::MarianKoEn => "marian-ko-en",
            Self::MarianEnAr => "marian-en-ar",
            Self::MarianArEn => "marian-ar-en",
        }
    }
}

/// Parameters for translation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TranslateParams {
    /// Source language code (ISO 639-1, e.g., "en", "es", "fr")
    pub source_language: String,

    /// Target language code (ISO 639-1, e.g., "en", "es", "fr")
    pub target_language: String,

    /// Preferred model to use (auto-selects best available if None)
    pub preferred_model: Option<TranslationModel>,

    /// If true, fall back to MADLAD-400 if preferred model unavailable
    pub fallback_enabled: bool,

    /// Force CPU even if GPU available
    pub force_cpu: bool,

    /// Use quantized model for faster inference (if available)
    pub use_quantized: bool,

    /// Maximum length of generated translation in tokens
    pub max_length: Option<usize>,

    /// Temperature for sampling (1.0 = no change, lower = more conservative)
    pub temperature: Option<f64>,

    /// Top-p nucleus sampling threshold (0.0-1.0)
    pub top_p: Option<f64>,

    /// Repetition penalty (1.0 = no penalty, higher = less repetition)
    pub repetition_penalty: Option<f64>,
}

impl Default for TranslateParams {
    fn default() -> Self {
        Self {
            source_language: "en".to_string(),
            target_language: "es".to_string(),
            preferred_model: None, // Auto-select best available
            fallback_enabled: true,
            force_cpu: false,
            use_quantized: false,
            max_length: Some(512),
            temperature: Some(0.0),
            top_p: None,
            repetition_penalty: Some(1.0),
        }
    }
}

/// Result of a translation operation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TranslationResult {
    /// The translated text
    pub translated_text: String,

    /// Source language used
    pub source_language: String,

    /// Target language used
    pub target_language: String,

    /// Model that was actually used
    pub model_used: TranslationModel,

    /// Time taken for inference in seconds
    pub inference_time: f64,

    /// Number of tokens generated
    pub token_count: usize,
}

/// Progress callback for long-running operations
/// (current_step, total_steps)
pub type ProgressCallback = Box<dyn Fn(usize, usize) + Send + Sync>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_id_resolution() {
        assert_eq!(TranslationModel::Madlad3B.model_id(), "jbochi/madlad400-3b-mt");
        assert_eq!(
            TranslationModel::MarianEnEs.model_id(),
            "Helsinki-NLP/opus-mt-en-es"
        );
    }

    #[test]
    fn test_multilingual_detection() {
        assert!(TranslationModel::Madlad3B.is_multilingual());
        assert!(TranslationModel::Madlad7B.is_multilingual());
        assert!(!TranslationModel::MarianEnEs.is_multilingual());
        assert!(!TranslationModel::MarianFrEn.is_multilingual());
    }

    #[test]
    fn test_language_pair_extraction() {
        assert_eq!(
            TranslationModel::MarianEnEs.language_pair(),
            Some(("en", "es"))
        );
        assert_eq!(
            TranslationModel::MarianFrEn.language_pair(),
            Some(("fr", "en"))
        );
        assert_eq!(TranslationModel::Madlad3B.language_pair(), None);
    }

    #[test]
    fn test_supports_pair() {
        let madlad = TranslationModel::Madlad3B;
        assert!(madlad.supports_pair("en", "es"));
        assert!(madlad.supports_pair("zh", "ar"));
        assert!(madlad.supports_pair("fr", "de"));

        let marian = TranslationModel::MarianEnEs;
        assert!(marian.supports_pair("en", "es"));
        assert!(!marian.supports_pair("es", "en"));
        assert!(!marian.supports_pair("en", "fr"));
    }

    #[test]
    fn test_model_family() {
        assert!(TranslationModel::Madlad3B.is_madlad());
        assert!(TranslationModel::Madlad7B.is_madlad());
        assert!(!TranslationModel::MarianEnEs.is_madlad());

        assert!(!TranslationModel::Madlad3B.is_marian());
        assert!(TranslationModel::MarianEnEs.is_marian());
    }

    #[test]
    fn test_default_params() {
        let params = TranslateParams::default();
        assert_eq!(params.source_language, "en");
        assert_eq!(params.target_language, "es");
        assert!(params.preferred_model.is_none());
        assert!(params.fallback_enabled);
        assert!(!params.force_cpu);
        assert_eq!(params.max_length, Some(512));
    }

    #[test]
    fn test_model_sizes() {
        assert_eq!(TranslationModel::Madlad3B.approx_size_mb(), 3000);
        assert_eq!(TranslationModel::MarianEnEs.approx_size_mb(), 298);
        assert_eq!(TranslationModel::Madlad10B.approx_size_mb(), 10000);
    }

    #[test]
    fn test_safetensors_revision() {
        // MADLAD models should use main branch
        assert_eq!(TranslationModel::Madlad3B.safetensors_revision(), None);
        assert_eq!(TranslationModel::Madlad7B.safetensors_revision(), None);
        assert_eq!(TranslationModel::Madlad10B.safetensors_revision(), None);

        // All Marian models should use refs/pr/4
        assert_eq!(
            TranslationModel::MarianEnEs.safetensors_revision(),
            Some("refs/pr/4")
        );
        assert_eq!(
            TranslationModel::MarianFrEn.safetensors_revision(),
            Some("refs/pr/4")
        );
        assert_eq!(
            TranslationModel::MarianEnDe.safetensors_revision(),
            Some("refs/pr/4")
        );
    }
}
