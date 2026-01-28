use crate::translate::types::TranslationModel;

use crate::translate::catalog::{load_model_catalog, ModelFamily};
use tracing::warn;

/// Check if a Marian model with safetensors exists for a specific language pair
pub fn is_safetensors_marian_pair(source: &str, target: &str) -> bool {
    let catalog = match load_model_catalog() {
        Ok(models) => models,
        Err(err) => {
            warn!(error = %err, "Failed to load model catalog");
            return false;
        }
    };

    catalog.iter().any(|model| {
        model.family == ModelFamily::Marian
            && model.has_safetensors
            && model.source.as_deref() == Some(source)
            && model.target.as_deref() == Some(target)
    })
}

/// Get the best MarianMT model for a specific language pair, if available
pub fn get_marian_for_pair(source: &str, target: &str) -> Option<TranslationModel> {
    if !is_safetensors_marian_pair(source, target) {
        return None;
    }

    match (source, target) {
        // Romance Languages
        ("en", "es") => Some(TranslationModel::MarianEnEs),
        ("es", "en") => Some(TranslationModel::MarianEsEn),
        ("en", "fr") => Some(TranslationModel::MarianEnFr),
        ("fr", "en") => Some(TranslationModel::MarianFrEn),
        ("en", "pt") => Some(TranslationModel::MarianEnPt),
        ("pt", "en") => Some(TranslationModel::MarianPtEn),
        ("en", "it") => Some(TranslationModel::MarianEnIt),
        ("it", "en") => Some(TranslationModel::MarianItEn),
        ("en", "ro") => Some(TranslationModel::MarianEnRo),
        ("ro", "en") => Some(TranslationModel::MarianRoEn),

        // Germanic Languages
        ("en", "de") => Some(TranslationModel::MarianEnDe),
        ("de", "en") => Some(TranslationModel::MarianDeEn),
        ("en", "nl") => Some(TranslationModel::MarianEnNl),
        ("nl", "en") => Some(TranslationModel::MarianNlEn),
        ("en", "sv") => Some(TranslationModel::MarianEnSv),
        ("sv", "en") => Some(TranslationModel::MarianSvEn),
        ("en", "da") => Some(TranslationModel::MarianEnDa),
        ("da", "en") => Some(TranslationModel::MarianDaEn),
        ("en", "no") => Some(TranslationModel::MarianEnNo),
        ("no", "en") => Some(TranslationModel::MarianNoEn),

        // Slavic Languages
        ("en", "ru") => Some(TranslationModel::MarianEnRu),
        ("ru", "en") => Some(TranslationModel::MarianRuEn),
        ("en", "pl") => Some(TranslationModel::MarianEnPl),
        ("pl", "en") => Some(TranslationModel::MarianPlEn),
        ("en", "cs") => Some(TranslationModel::MarianEnCs),
        ("cs", "en") => Some(TranslationModel::MarianCsEn),
        ("en", "uk") => Some(TranslationModel::MarianEnUk),
        ("uk", "en") => Some(TranslationModel::MarianUkEn),

        // East Asian Languages
        ("en", "zh") => Some(TranslationModel::MarianEnZh),
        ("zh", "en") => Some(TranslationModel::MarianZhEn),
        ("en", "ja") => Some(TranslationModel::MarianEnJa),
        ("ja", "en") => Some(TranslationModel::MarianJaEn),
        ("en", "ko") => Some(TranslationModel::MarianEnKo),
        ("ko", "en") => Some(TranslationModel::MarianKoEn),

        // Southeast Asian Languages
        ("en", "vi") => Some(TranslationModel::MarianEnVi),
        ("vi", "en") => Some(TranslationModel::MarianViEn),
        ("en", "th") => Some(TranslationModel::MarianEnTh),
        ("th", "en") => Some(TranslationModel::MarianThEn),
        ("en", "id") => Some(TranslationModel::MarianEnId),
        ("id", "en") => Some(TranslationModel::MarianIdEn),

        // Middle Eastern Languages
        ("en", "ar") => Some(TranslationModel::MarianEnAr),
        ("ar", "en") => Some(TranslationModel::MarianArEn),
        ("en", "he") => Some(TranslationModel::MarianEnHe),
        ("he", "en") => Some(TranslationModel::MarianHeEn),
        ("en", "fa") => Some(TranslationModel::MarianEnFa),
        ("fa", "en") => Some(TranslationModel::MarianFaEn),
        ("en", "tr") => Some(TranslationModel::MarianEnTr),
        ("tr", "en") => Some(TranslationModel::MarianTrEn),

        // South Asian Languages
        ("en", "hi") => Some(TranslationModel::MarianEnHi),
        ("hi", "en") => Some(TranslationModel::MarianHiEn),
        ("en", "bn") => Some(TranslationModel::MarianEnBn),
        ("bn", "en") => Some(TranslationModel::MarianBnEn),
        ("en", "ur") => Some(TranslationModel::MarianEnUr),
        ("ur", "en") => Some(TranslationModel::MarianUrEn),

        // Other European Languages
        ("en", "hu") => Some(TranslationModel::MarianEnHu),
        ("hu", "en") => Some(TranslationModel::MarianHuEn),
        ("en", "fi") => Some(TranslationModel::MarianEnFi),
        ("fi", "en") => Some(TranslationModel::MarianFiEn),
        ("en", "el") => Some(TranslationModel::MarianEnEl),
        ("el", "en") => Some(TranslationModel::MarianElEn),

        // African Languages
        ("en", "sw") => Some(TranslationModel::MarianEnSw),
        ("sw", "en") => Some(TranslationModel::MarianSwEn),

        // No specialized model for this pair
        _ => None,
    }
}

/// Common language codes and their names (safetensors Marian support)
pub const SUPPORTED_LANGUAGES: &[(&str, &str)] = &[
    ("en", "English"),
    ("es", "Spanish"),
    ("fr", "French"),
    ("de", "German"),
    ("nl", "Dutch"),
    ("sv", "Swedish"),
    ("ru", "Russian"),
    ("ar", "Arabic"),
    ("he", "Hebrew"),
    ("bn", "Bengali"),
    ("hu", "Hungarian"),
    ("fi", "Finnish"),
];

/// Validate that a language code is recognized
pub fn is_valid_language_code(code: &str) -> bool {
    SUPPORTED_LANGUAGES
        .iter()
        .any(|(lang_code, _)| *lang_code == code)
}

/// Get the full name of a language from its code
pub fn get_language_name(code: &str) -> Option<&'static str> {
    SUPPORTED_LANGUAGES
        .iter()
        .find(|(lang_code, _)| *lang_code == code)
        .map(|(_, name)| *name)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_get_marian_for_pair() {
        assert_eq!(
            get_marian_for_pair("en", "es"),
            Some(TranslationModel::MarianEnEs)
        );
        assert_eq!(
            get_marian_for_pair("fr", "en"),
            Some(TranslationModel::MarianFrEn)
        );
        assert_eq!(
            get_marian_for_pair("sv", "en"),
            Some(TranslationModel::MarianSvEn)
        );
        assert_eq!(get_marian_for_pair("en", "fr"), None);
        assert_eq!(
            get_marian_for_pair("nl", "en"),
            Some(TranslationModel::MarianNlEn)
        );
        assert_eq!(get_marian_for_pair("en", "nl"), None);
        assert_eq!(get_marian_for_pair("en", "sw"), None);
        assert_eq!(get_marian_for_pair("en", "en"), None);
        assert_eq!(get_marian_for_pair("es", "fr"), None); // No direct ES->FR model
    }

    #[test]
    fn test_bidirectional_pairs() {
        // Ensure we have both directions for major pairs
        assert!(get_marian_for_pair("en", "es").is_some());
        assert!(get_marian_for_pair("es", "en").is_some());
        assert!(get_marian_for_pair("en", "de").is_some());
        assert!(get_marian_for_pair("de", "en").is_some());
        assert!(get_marian_for_pair("en", "ru").is_some());
        assert!(get_marian_for_pair("ru", "en").is_some());
        assert!(get_marian_for_pair("en", "ar").is_some());
        assert!(get_marian_for_pair("ar", "en").is_some());
    }

    #[test]
    fn test_language_validation() {
        assert!(is_valid_language_code("en"));
        assert!(is_valid_language_code("es"));
        assert!(is_valid_language_code("de"));
        assert!(!is_valid_language_code("xx"));
        assert!(!is_valid_language_code(""));
    }

    #[test]
    fn test_language_names() {
        assert_eq!(get_language_name("en"), Some("English"));
        assert_eq!(get_language_name("es"), Some("Spanish"));
        assert_eq!(get_language_name("de"), Some("German"));
        assert_eq!(get_language_name("xx"), None);
    }
}
