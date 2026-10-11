use serde::{Deserialize, Serialize};

/// Available translation models from HuggingFace
///
/// All models listed here are fine-tuned for translation and ready to use:
/// - **MarianMT models**: Specialized for specific language pairs (fastest, best quality for supported pairs)
/// - **MADLAD-400 models**: Multilingual models supporting 450+ languages (use for unsupported pairs)
///
/// Every model publishes safetensors weights (monolithic or sharded), which is
/// the only checkpoint format the Candle loaders support.
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

    // Specialized MarianMT models (specific language pairs - fastest)
    // Romance Languages
    #[serde(rename = "marian-en-es")]
    MarianEnEs,
    #[serde(rename = "marian-es-en")]
    MarianEsEn,
    #[serde(rename = "marian-fr-en")]
    MarianFrEn,

    // Germanic Languages
    #[serde(rename = "marian-en-de")]
    MarianEnDe,
    #[serde(rename = "marian-de-en")]
    MarianDeEn,
    #[serde(rename = "marian-nl-en")]
    MarianNlEn,
    #[serde(rename = "marian-sv-en")]
    MarianSvEn,

    // Slavic Languages
    #[serde(rename = "marian-en-ru")]
    MarianEnRu,
    #[serde(rename = "marian-ru-en")]
    MarianRuEn,

    // East Asian Languages
    #[serde(rename = "marian-en-ko")]
    MarianEnKo,

    // Middle Eastern Languages
    #[serde(rename = "marian-en-ar")]
    MarianEnAr,
    #[serde(rename = "marian-ar-en")]
    MarianArEn,
    #[serde(rename = "marian-he-en")]
    MarianHeEn,

    // South Asian Languages
    #[serde(rename = "marian-bn-en")]
    MarianBnEn,

    // Other European Languages
    #[serde(rename = "marian-en-hu")]
    MarianEnHu,
    #[serde(rename = "marian-fi-en")]
    MarianFiEn,
}

impl TranslationModel {
    /// Returns the HuggingFace model ID for downloading
    pub fn model_id(&self) -> &'static str {
        match self {
            // MADLAD-400 multilingual models
            Self::Madlad3B => "jbochi/madlad400-3b-mt",
            Self::Madlad7B => "jbochi/madlad400-7b-mt",
            Self::Madlad10B => "jbochi/madlad400-10b-mt",

            // Romance Languages
            Self::MarianEnEs => "Helsinki-NLP/opus-mt-en-es",
            Self::MarianEsEn => "Helsinki-NLP/opus-mt-es-en",
            Self::MarianFrEn => "Helsinki-NLP/opus-mt-fr-en",

            // Germanic Languages
            Self::MarianEnDe => "Helsinki-NLP/opus-mt-en-de",
            Self::MarianDeEn => "Helsinki-NLP/opus-mt-de-en",
            Self::MarianNlEn => "Helsinki-NLP/opus-mt-nl-en",
            Self::MarianSvEn => "Helsinki-NLP/opus-mt-sv-en",

            // Slavic Languages
            Self::MarianEnRu => "Helsinki-NLP/opus-mt-en-ru",
            Self::MarianRuEn => "Helsinki-NLP/opus-mt-ru-en",

            // East Asian Languages
            Self::MarianEnKo => "Helsinki-NLP/opus-mt-tc-big-en-ko",

            // Middle Eastern Languages
            Self::MarianEnAr => "Helsinki-NLP/opus-mt-en-ar",
            Self::MarianArEn => "Helsinki-NLP/opus-mt-ar-en",
            Self::MarianHeEn => "Helsinki-NLP/opus-mt-tc-big-he-en",

            // South Asian Languages
            Self::MarianBnEn => "Helsinki-NLP/opus-mt-bn-en",

            // Other European Languages
            Self::MarianEnHu => "Helsinki-NLP/opus-mt-tc-big-en-hu",
            Self::MarianFiEn => "Helsinki-NLP/opus-mt-tc-big-fi-en",
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
        !self.is_madlad()
    }

    /// Returns the catalog entry for this model, looked up by its stable CLI key.
    ///
    /// The catalog (`models.toml`) is the single source of truth for repository
    /// ids and revisions.
    pub fn catalog_entry(&self) -> Option<crate::translate::catalog::CatalogModel> {
        let catalog = crate::translate::catalog::load_model_catalog().ok()?;
        catalog
            .into_iter()
            .find(|entry| entry.name == self.cli_key())
    }

    /// Returns the repository revision selected by the catalog.
    ///
    /// Both `"main"` and `None` select the repository's default branch.
    pub fn catalog_revision(&self) -> Option<String> {
        self.catalog_entry().and_then(|entry| entry.revision)
    }

    /// Returns the source and target languages for specialized models
    /// Returns None for multilingual models (they support all pairs)
    pub fn language_pair(&self) -> Option<(&'static str, &'static str)> {
        match self {
            // Romance Languages
            Self::MarianEnEs => Some(("en", "es")),
            Self::MarianEsEn => Some(("es", "en")),
            Self::MarianFrEn => Some(("fr", "en")),

            // Germanic Languages
            Self::MarianEnDe => Some(("en", "de")),
            Self::MarianDeEn => Some(("de", "en")),
            Self::MarianNlEn => Some(("nl", "en")),
            Self::MarianSvEn => Some(("sv", "en")),

            // Slavic Languages
            Self::MarianEnRu => Some(("en", "ru")),
            Self::MarianRuEn => Some(("ru", "en")),

            // East Asian Languages
            Self::MarianEnKo => Some(("en", "ko")),

            // Middle Eastern Languages
            Self::MarianEnAr => Some(("en", "ar")),
            Self::MarianArEn => Some(("ar", "en")),
            Self::MarianHeEn => Some(("he", "en")),

            // South Asian Languages
            Self::MarianBnEn => Some(("bn", "en")),

            // Other European Languages
            Self::MarianEnHu => Some(("en", "hu")),
            Self::MarianFiEn => Some(("fi", "en")),

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
            // MADLAD models accept any pair
            let _ = (source, target);
            true
        }
    }

    /// Returns approximate model size in MB
    pub fn approx_size_mb(&self) -> u64 {
        match self {
            Self::Madlad3B => 11800,  // 11.8 GB
            Self::Madlad7B => 20000,  // ~20 GB
            Self::Madlad10B => 38000, // ~38 GB
            // MarianMT models are all similar size
            _ if self.is_marian() => 298,
            _ => 0,
        }
    }

    /// Returns a human-friendly display name
    pub fn display_name(&self) -> &'static str {
        match self {
            Self::Madlad3B => "MADLAD-400 3B (11.8GB, 450+ languages)",
            Self::Madlad7B => "MADLAD-400 7B (20GB, 450+ languages)",
            Self::Madlad10B => "MADLAD-400 10B (38GB, 450+ languages)",
            Self::MarianEnEs => "MarianMT EN→ES (298MB)",
            Self::MarianEsEn => "MarianMT ES→EN (298MB)",
            Self::MarianFrEn => "MarianMT FR→EN (298MB)",
            Self::MarianEnDe => "MarianMT EN→DE (298MB)",
            Self::MarianDeEn => "MarianMT DE→EN (298MB)",
            Self::MarianNlEn => "MarianMT NL→EN (298MB)",
            Self::MarianSvEn => "MarianMT SV→EN (298MB)",
            Self::MarianEnRu => "MarianMT EN→RU (298MB)",
            Self::MarianRuEn => "MarianMT RU→EN (298MB)",
            Self::MarianEnKo => "MarianMT EN→KO (298MB)",
            Self::MarianEnAr => "MarianMT EN→AR (298MB)",
            Self::MarianArEn => "MarianMT AR→EN (298MB)",
            Self::MarianHeEn => "MarianMT HE→EN (298MB)",
            Self::MarianBnEn => "MarianMT BN→EN (298MB)",
            Self::MarianEnHu => "MarianMT EN→HU (298MB)",
            Self::MarianFiEn => "MarianMT FI→EN (298MB)",
        }
    }

    /// Returns CLI key for --model flag
    pub fn cli_key(&self) -> &'static str {
        match self {
            Self::Madlad3B => "madlad-3b",
            Self::Madlad7B => "madlad-7b",
            Self::Madlad10B => "madlad-10b",
            Self::MarianEnEs => "marian-en-es",
            Self::MarianEsEn => "marian-es-en",
            Self::MarianFrEn => "marian-fr-en",
            Self::MarianEnDe => "marian-en-de",
            Self::MarianDeEn => "marian-de-en",
            Self::MarianNlEn => "marian-nl-en",
            Self::MarianSvEn => "marian-sv-en",
            Self::MarianEnRu => "marian-en-ru",
            Self::MarianRuEn => "marian-ru-en",
            Self::MarianEnKo => "marian-en-ko",
            Self::MarianEnAr => "marian-en-ar",
            Self::MarianArEn => "marian-ar-en",
            Self::MarianHeEn => "marian-he-en",
            Self::MarianBnEn => "marian-bn-en",
            Self::MarianEnHu => "marian-en-hu",
            Self::MarianFiEn => "marian-fi-en",
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
        assert_eq!(
            TranslationModel::Madlad3B.model_id(),
            "jbochi/madlad400-3b-mt"
        );
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
        assert_eq!(TranslationModel::Madlad3B.approx_size_mb(), 11800);
        assert_eq!(TranslationModel::MarianEnEs.approx_size_mb(), 298);
        assert_eq!(TranslationModel::Madlad10B.approx_size_mb(), 38000);
    }

    #[test]
    fn test_catalog_revision() {
        // MADLAD models load from the default branch
        assert_eq!(
            TranslationModel::Madlad3B.catalog_revision(),
            Some("main".to_string())
        );
        assert_eq!(
            TranslationModel::Madlad7B.catalog_revision(),
            Some("main".to_string())
        );
        assert_eq!(
            TranslationModel::Madlad10B.catalog_revision(),
            Some("main".to_string())
        );

        // Marian models with safetensors in a PR ref
        assert_eq!(
            TranslationModel::MarianEnEs.catalog_revision(),
            Some("refs/pr/4".to_string())
        );
        assert_eq!(
            TranslationModel::MarianFrEn.catalog_revision(),
            Some("refs/pr/4".to_string())
        );
        assert_eq!(
            TranslationModel::MarianEnDe.catalog_revision(),
            Some("refs/pr/4".to_string())
        );

        // Marian models with safetensors on the default branch
        assert_eq!(
            TranslationModel::MarianHeEn.catalog_revision(),
            Some("main".to_string())
        );
        assert_eq!(
            TranslationModel::MarianEnKo.catalog_revision(),
            Some("main".to_string())
        );
    }
}
