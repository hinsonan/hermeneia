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

    // Specialized MarianMT models (specific language pairs - fastest)
    // Romance Languages
    #[serde(rename = "marian-en-es")]
    MarianEnEs,
    #[serde(rename = "marian-es-en")]
    MarianEsEn,
    #[serde(rename = "marian-en-fr")]
    MarianEnFr,
    #[serde(rename = "marian-fr-en")]
    MarianFrEn,
    #[serde(rename = "marian-en-pt")]
    MarianEnPt,
    #[serde(rename = "marian-pt-en")]
    MarianPtEn,
    #[serde(rename = "marian-en-it")]
    MarianEnIt,
    #[serde(rename = "marian-it-en")]
    MarianItEn,
    #[serde(rename = "marian-en-ro")]
    MarianEnRo,
    #[serde(rename = "marian-ro-en")]
    MarianRoEn,

    // Germanic Languages
    #[serde(rename = "marian-en-de")]
    MarianEnDe,
    #[serde(rename = "marian-de-en")]
    MarianDeEn,
    #[serde(rename = "marian-en-nl")]
    MarianEnNl,
    #[serde(rename = "marian-nl-en")]
    MarianNlEn,
    #[serde(rename = "marian-en-sv")]
    MarianEnSv,
    #[serde(rename = "marian-sv-en")]
    MarianSvEn,
    #[serde(rename = "marian-en-da")]
    MarianEnDa,
    #[serde(rename = "marian-da-en")]
    MarianDaEn,
    #[serde(rename = "marian-en-no")]
    MarianEnNo,
    #[serde(rename = "marian-no-en")]
    MarianNoEn,

    // Slavic Languages
    #[serde(rename = "marian-en-ru")]
    MarianEnRu,
    #[serde(rename = "marian-ru-en")]
    MarianRuEn,
    #[serde(rename = "marian-en-pl")]
    MarianEnPl,
    #[serde(rename = "marian-pl-en")]
    MarianPlEn,
    #[serde(rename = "marian-en-cs")]
    MarianEnCs,
    #[serde(rename = "marian-cs-en")]
    MarianCsEn,
    #[serde(rename = "marian-en-uk")]
    MarianEnUk,
    #[serde(rename = "marian-uk-en")]
    MarianUkEn,

    // East Asian Languages
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

    // Southeast Asian Languages
    #[serde(rename = "marian-en-vi")]
    MarianEnVi,
    #[serde(rename = "marian-vi-en")]
    MarianViEn,
    #[serde(rename = "marian-en-th")]
    MarianEnTh,
    #[serde(rename = "marian-th-en")]
    MarianThEn,
    #[serde(rename = "marian-en-id")]
    MarianEnId,
    #[serde(rename = "marian-id-en")]
    MarianIdEn,

    // Middle Eastern Languages
    #[serde(rename = "marian-en-ar")]
    MarianEnAr,
    #[serde(rename = "marian-ar-en")]
    MarianArEn,
    #[serde(rename = "marian-en-he")]
    MarianEnHe,
    #[serde(rename = "marian-he-en")]
    MarianHeEn,
    #[serde(rename = "marian-en-fa")]
    MarianEnFa,
    #[serde(rename = "marian-fa-en")]
    MarianFaEn,
    #[serde(rename = "marian-en-tr")]
    MarianEnTr,
    #[serde(rename = "marian-tr-en")]
    MarianTrEn,

    // South Asian Languages
    #[serde(rename = "marian-en-hi")]
    MarianEnHi,
    #[serde(rename = "marian-hi-en")]
    MarianHiEn,
    #[serde(rename = "marian-en-bn")]
    MarianEnBn,
    #[serde(rename = "marian-bn-en")]
    MarianBnEn,
    #[serde(rename = "marian-en-ur")]
    MarianEnUr,
    #[serde(rename = "marian-ur-en")]
    MarianUrEn,

    // Other European Languages
    #[serde(rename = "marian-en-hu")]
    MarianEnHu,
    #[serde(rename = "marian-hu-en")]
    MarianHuEn,
    #[serde(rename = "marian-en-fi")]
    MarianEnFi,
    #[serde(rename = "marian-fi-en")]
    MarianFiEn,
    #[serde(rename = "marian-en-el")]
    MarianEnEl,
    #[serde(rename = "marian-el-en")]
    MarianElEn,

    // African Languages
    #[serde(rename = "marian-en-sw")]
    MarianEnSw,
    #[serde(rename = "marian-sw-en")]
    MarianSwEn,
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
            Self::MarianEnFr => "Helsinki-NLP/opus-mt-en-fr",
            Self::MarianFrEn => "Helsinki-NLP/opus-mt-fr-en",
            Self::MarianEnPt => "Helsinki-NLP/opus-mt-en-roa",
            Self::MarianPtEn => "Helsinki-NLP/opus-mt-roa-en",
            Self::MarianEnIt => "Helsinki-NLP/opus-mt-en-it",
            Self::MarianItEn => "Helsinki-NLP/opus-mt-it-en",
            Self::MarianEnRo => "Helsinki-NLP/opus-mt-en-ro",
            Self::MarianRoEn => "Helsinki-NLP/opus-mt-ro-en",

            // Germanic Languages
            Self::MarianEnDe => "Helsinki-NLP/opus-mt-en-de",
            Self::MarianDeEn => "Helsinki-NLP/opus-mt-de-en",
            Self::MarianEnNl => "Helsinki-NLP/opus-mt-en-nl",
            Self::MarianNlEn => "Helsinki-NLP/opus-mt-nl-en",
            Self::MarianEnSv => "Helsinki-NLP/opus-mt-en-sv",
            Self::MarianSvEn => "Helsinki-NLP/opus-mt-sv-en",
            Self::MarianEnDa => "Helsinki-NLP/opus-mt-en-da",
            Self::MarianDaEn => "Helsinki-NLP/opus-mt-da-en",
            Self::MarianEnNo => "Helsinki-NLP/opus-mt-en-no",
            Self::MarianNoEn => "Helsinki-NLP/opus-mt-no-en",

            // Slavic Languages
            Self::MarianEnRu => "Helsinki-NLP/opus-mt-en-ru",
            Self::MarianRuEn => "Helsinki-NLP/opus-mt-ru-en",
            Self::MarianEnPl => "Helsinki-NLP/opus-mt-en-pl",
            Self::MarianPlEn => "Helsinki-NLP/opus-mt-pl-en",
            Self::MarianEnCs => "Helsinki-NLP/opus-mt-en-cs",
            Self::MarianCsEn => "Helsinki-NLP/opus-mt-cs-en",
            Self::MarianEnUk => "Helsinki-NLP/opus-mt-en-uk",
            Self::MarianUkEn => "Helsinki-NLP/opus-mt-uk-en",

            // East Asian Languages
            Self::MarianEnZh => "Helsinki-NLP/opus-mt-en-zh",
            Self::MarianZhEn => "Helsinki-NLP/opus-mt-zh-en",
            Self::MarianEnJa => "Helsinki-NLP/opus-mt-en-jap",
            Self::MarianJaEn => "Helsinki-NLP/opus-mt-jap-en",
            Self::MarianEnKo => "Helsinki-NLP/opus-mt-en-ko",
            Self::MarianKoEn => "Helsinki-NLP/opus-mt-ko-en",

            // Southeast Asian Languages
            Self::MarianEnVi => "Helsinki-NLP/opus-mt-en-vi",
            Self::MarianViEn => "Helsinki-NLP/opus-mt-vi-en",
            Self::MarianEnTh => "Helsinki-NLP/opus-mt-en-th",
            Self::MarianThEn => "Helsinki-NLP/opus-mt-th-en",
            Self::MarianEnId => "Helsinki-NLP/opus-mt-en-id",
            Self::MarianIdEn => "Helsinki-NLP/opus-mt-id-en",

            // Middle Eastern Languages
            Self::MarianEnAr => "Helsinki-NLP/opus-mt-en-ar",
            Self::MarianArEn => "Helsinki-NLP/opus-mt-ar-en",
            Self::MarianEnHe => "Helsinki-NLP/opus-mt-en-he",
            Self::MarianHeEn => "Helsinki-NLP/opus-mt-tc-big-he-en",
            Self::MarianEnFa => "Helsinki-NLP/opus-mt-en-fa",
            Self::MarianFaEn => "Helsinki-NLP/opus-mt-fa-en",
            Self::MarianEnTr => "Helsinki-NLP/opus-mt-tc-big-en-tr",
            Self::MarianTrEn => "Helsinki-NLP/opus-mt-tr-en",

            // South Asian Languages
            Self::MarianEnHi => "Helsinki-NLP/opus-mt-en-hi",
            Self::MarianHiEn => "Helsinki-NLP/opus-mt-hi-en",
            Self::MarianEnBn => "Helsinki-NLP/opus-mt-en-bn",
            Self::MarianBnEn => "Helsinki-NLP/opus-mt-bn-en",
            Self::MarianEnUr => "Helsinki-NLP/opus-mt-en-ur",
            Self::MarianUrEn => "Helsinki-NLP/opus-mt-ur-en",

            // Other European Languages
            Self::MarianEnHu => "Helsinki-NLP/opus-mt-tc-big-en-hu",
            Self::MarianHuEn => "Helsinki-NLP/opus-mt-hu-en",
            Self::MarianEnFi => "Helsinki-NLP/opus-mt-en-fi",
            Self::MarianFiEn => "Helsinki-NLP/opus-mt-tc-big-fi-en",
            Self::MarianEnEl => "Helsinki-NLP/opus-mt-en-el",
            Self::MarianElEn => "Helsinki-NLP/opus-mt-el-en",

            // African Languages
            Self::MarianEnSw => "Helsinki-NLP/opus-mt-en-sw",
            Self::MarianSwEn => "Helsinki-NLP/opus-mt-sw-en",
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

    /// Returns the HuggingFace revision/branch that contains safetensors files
    /// Most models on main branch don't have safetensors, they're in PR branches
    pub fn safetensors_revision(&self) -> Option<&'static str> {
        match self {
            // MADLAD models - main branch has safetensors
            Self::Madlad3B | Self::Madlad7B | Self::Madlad10B => None,

            // TC-Big variants use main branch
            Self::MarianHeEn | Self::MarianEnTr | Self::MarianEnHu | Self::MarianFiEn => None,

            // All other MarianMT models use refs/pr/4 for safetensors
            _ if self.is_marian() => Some("refs/pr/4"),

            _ => None,
        }
    }

    /// Returns the source and target languages for specialized models
    /// Returns None for multilingual models (they support all pairs)
    pub fn language_pair(&self) -> Option<(&'static str, &'static str)> {
        match self {
            // Romance Languages
            Self::MarianEnEs => Some(("en", "es")),
            Self::MarianEsEn => Some(("es", "en")),
            Self::MarianEnFr => Some(("en", "fr")),
            Self::MarianFrEn => Some(("fr", "en")),
            Self::MarianEnPt => Some(("en", "pt")),
            Self::MarianPtEn => Some(("pt", "en")),
            Self::MarianEnIt => Some(("en", "it")),
            Self::MarianItEn => Some(("it", "en")),
            Self::MarianEnRo => Some(("en", "ro")),
            Self::MarianRoEn => Some(("ro", "en")),

            // Germanic Languages
            Self::MarianEnDe => Some(("en", "de")),
            Self::MarianDeEn => Some(("de", "en")),
            Self::MarianEnNl => Some(("en", "nl")),
            Self::MarianNlEn => Some(("nl", "en")),
            Self::MarianEnSv => Some(("en", "sv")),
            Self::MarianSvEn => Some(("sv", "en")),
            Self::MarianEnDa => Some(("en", "da")),
            Self::MarianDaEn => Some(("da", "en")),
            Self::MarianEnNo => Some(("en", "no")),
            Self::MarianNoEn => Some(("no", "en")),

            // Slavic Languages
            Self::MarianEnRu => Some(("en", "ru")),
            Self::MarianRuEn => Some(("ru", "en")),
            Self::MarianEnPl => Some(("en", "pl")),
            Self::MarianPlEn => Some(("pl", "en")),
            Self::MarianEnCs => Some(("en", "cs")),
            Self::MarianCsEn => Some(("cs", "en")),
            Self::MarianEnUk => Some(("en", "uk")),
            Self::MarianUkEn => Some(("uk", "en")),

            // East Asian Languages
            Self::MarianEnZh => Some(("en", "zh")),
            Self::MarianZhEn => Some(("zh", "en")),
            Self::MarianEnJa => Some(("en", "ja")),
            Self::MarianJaEn => Some(("ja", "en")),
            Self::MarianEnKo => Some(("en", "ko")),
            Self::MarianKoEn => Some(("ko", "en")),

            // Southeast Asian Languages
            Self::MarianEnVi => Some(("en", "vi")),
            Self::MarianViEn => Some(("vi", "en")),
            Self::MarianEnTh => Some(("en", "th")),
            Self::MarianThEn => Some(("th", "en")),
            Self::MarianEnId => Some(("en", "id")),
            Self::MarianIdEn => Some(("id", "en")),

            // Middle Eastern Languages
            Self::MarianEnAr => Some(("en", "ar")),
            Self::MarianArEn => Some(("ar", "en")),
            Self::MarianEnHe => Some(("en", "he")),
            Self::MarianHeEn => Some(("he", "en")),
            Self::MarianEnFa => Some(("en", "fa")),
            Self::MarianFaEn => Some(("fa", "en")),
            Self::MarianEnTr => Some(("en", "tr")),
            Self::MarianTrEn => Some(("tr", "en")),

            // South Asian Languages
            Self::MarianEnHi => Some(("en", "hi")),
            Self::MarianHiEn => Some(("hi", "en")),
            Self::MarianEnBn => Some(("en", "bn")),
            Self::MarianBnEn => Some(("bn", "en")),
            Self::MarianEnUr => Some(("en", "ur")),
            Self::MarianUrEn => Some(("ur", "en")),

            // Other European Languages
            Self::MarianEnHu => Some(("en", "hu")),
            Self::MarianHuEn => Some(("hu", "en")),
            Self::MarianEnFi => Some(("en", "fi")),
            Self::MarianFiEn => Some(("fi", "en")),
            Self::MarianEnEl => Some(("en", "el")),
            Self::MarianElEn => Some(("el", "en")),

            // African Languages
            Self::MarianEnSw => Some(("en", "sw")),
            Self::MarianSwEn => Some(("sw", "en")),

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
            // All other MarianMT models - use generic format
            _ => "MarianMT (298MB)",
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

            // New models - Romance Languages
            Self::MarianEnRo => "marian-en-ro",
            Self::MarianRoEn => "marian-ro-en",

            // Germanic Languages
            Self::MarianEnNl => "marian-en-nl",
            Self::MarianNlEn => "marian-nl-en",
            Self::MarianEnSv => "marian-en-sv",
            Self::MarianSvEn => "marian-sv-en",
            Self::MarianEnDa => "marian-en-da",
            Self::MarianDaEn => "marian-da-en",
            Self::MarianEnNo => "marian-en-no",
            Self::MarianNoEn => "marian-no-en",

            // Slavic Languages
            Self::MarianEnPl => "marian-en-pl",
            Self::MarianPlEn => "marian-pl-en",
            Self::MarianEnCs => "marian-en-cs",
            Self::MarianCsEn => "marian-cs-en",
            Self::MarianEnUk => "marian-en-uk",
            Self::MarianUkEn => "marian-uk-en",

            // Southeast Asian
            Self::MarianEnVi => "marian-en-vi",
            Self::MarianViEn => "marian-vi-en",
            Self::MarianEnTh => "marian-en-th",
            Self::MarianThEn => "marian-th-en",
            Self::MarianEnId => "marian-en-id",
            Self::MarianIdEn => "marian-id-en",

            // Middle Eastern
            Self::MarianEnHe => "marian-en-he",
            Self::MarianHeEn => "marian-he-en",
            Self::MarianEnFa => "marian-en-fa",
            Self::MarianFaEn => "marian-fa-en",
            Self::MarianEnTr => "marian-en-tr",
            Self::MarianTrEn => "marian-tr-en",

            // South Asian
            Self::MarianEnHi => "marian-en-hi",
            Self::MarianHiEn => "marian-hi-en",
            Self::MarianEnBn => "marian-en-bn",
            Self::MarianBnEn => "marian-bn-en",
            Self::MarianEnUr => "marian-en-ur",
            Self::MarianUrEn => "marian-ur-en",

            // Other European
            Self::MarianEnHu => "marian-en-hu",
            Self::MarianHuEn => "marian-hu-en",
            Self::MarianEnFi => "marian-en-fi",
            Self::MarianFiEn => "marian-fi-en",
            Self::MarianEnEl => "marian-en-el",
            Self::MarianElEn => "marian-el-en",

            // African
            Self::MarianEnSw => "marian-en-sw",
            Self::MarianSwEn => "marian-sw-en",
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
    fn test_safetensors_revision() {
        // MADLAD models should use main branch
        assert_eq!(TranslationModel::Madlad3B.safetensors_revision(), None);
        assert_eq!(TranslationModel::Madlad7B.safetensors_revision(), None);
        assert_eq!(TranslationModel::Madlad10B.safetensors_revision(), None);

        // Typical Marian models should use refs/pr/4
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
        assert_eq!(TranslationModel::MarianHeEn.safetensors_revision(), None);
        assert_eq!(TranslationModel::MarianEnTr.safetensors_revision(), None);
    }
}
