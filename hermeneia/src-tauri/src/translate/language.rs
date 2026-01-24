use crate::translate::types::TranslationModel;

/// Get the best MarianMT model for a specific language pair, if available
pub fn get_marian_for_pair(source: &str, target: &str) -> Option<TranslationModel> {
    match (source, target) {
        // English to other languages
        ("en", "es") => Some(TranslationModel::MarianEnEs),
        ("en", "fr") => Some(TranslationModel::MarianEnFr),
        ("en", "de") => Some(TranslationModel::MarianEnDe),
        ("en", "pt") => Some(TranslationModel::MarianEnPt),
        ("en", "it") => Some(TranslationModel::MarianEnIt),
        ("en", "ru") => Some(TranslationModel::MarianEnRu),
        ("en", "zh") => Some(TranslationModel::MarianEnZh),
        ("en", "ja") => Some(TranslationModel::MarianEnJa),
        ("en", "ko") => Some(TranslationModel::MarianEnKo),
        ("en", "ar") => Some(TranslationModel::MarianEnAr),

        // Other languages to English
        ("es", "en") => Some(TranslationModel::MarianEsEn),
        ("fr", "en") => Some(TranslationModel::MarianFrEn),
        ("de", "en") => Some(TranslationModel::MarianDeEn),
        ("pt", "en") => Some(TranslationModel::MarianPtEn),
        ("it", "en") => Some(TranslationModel::MarianItEn),
        ("ru", "en") => Some(TranslationModel::MarianRuEn),
        ("zh", "en") => Some(TranslationModel::MarianZhEn),
        ("ja", "en") => Some(TranslationModel::MarianJaEn),
        ("ko", "en") => Some(TranslationModel::MarianKoEn),
        ("ar", "en") => Some(TranslationModel::MarianArEn),

        // No specialized model for this pair
        _ => None,
    }
}

/// Common language codes and their names
pub const SUPPORTED_LANGUAGES: &[(&str, &str)] = &[
    // European languages
    ("en", "English"),
    ("es", "Spanish"),
    ("fr", "French"),
    ("de", "German"),
    ("it", "Italian"),
    ("pt", "Portuguese"),
    ("nl", "Dutch"),
    ("pl", "Polish"),
    ("ru", "Russian"),
    ("cs", "Czech"),
    ("sv", "Swedish"),
    ("fi", "Finnish"),
    ("da", "Danish"),
    ("no", "Norwegian"),
    ("ro", "Romanian"),
    ("el", "Greek"),
    ("hu", "Hungarian"),
    ("tr", "Turkish"),
    // Asian languages
    ("zh", "Chinese"),
    ("ja", "Japanese"),
    ("ko", "Korean"),
    ("hi", "Hindi"),
    ("bn", "Bengali"),
    ("ta", "Tamil"),
    ("te", "Telugu"),
    ("th", "Thai"),
    ("vi", "Vietnamese"),
    ("id", "Indonesian"),
    ("ms", "Malay"),
    // Middle Eastern languages
    ("ar", "Arabic"),
    ("he", "Hebrew"),
    ("fa", "Persian"),
    ("ur", "Urdu"),
    // African languages
    ("sw", "Swahili"),
    ("yo", "Yoruba"),
    ("ha", "Hausa"),
    ("zu", "Zulu"),
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
        assert_eq!(get_marian_for_pair("en", "en"), None);
        assert_eq!(get_marian_for_pair("es", "fr"), None); // No direct ES->FR model
    }

    #[test]
    fn test_bidirectional_pairs() {
        // Ensure we have both directions for major pairs
        assert!(get_marian_for_pair("en", "es").is_some());
        assert!(get_marian_for_pair("es", "en").is_some());
        assert!(get_marian_for_pair("en", "fr").is_some());
        assert!(get_marian_for_pair("fr", "en").is_some());
    }

    #[test]
    fn test_language_validation() {
        assert!(is_valid_language_code("en"));
        assert!(is_valid_language_code("es"));
        assert!(is_valid_language_code("zh"));
        assert!(!is_valid_language_code("xx"));
        assert!(!is_valid_language_code(""));
    }

    #[test]
    fn test_language_names() {
        assert_eq!(get_language_name("en"), Some("English"));
        assert_eq!(get_language_name("es"), Some("Spanish"));
        assert_eq!(get_language_name("zh"), Some("Chinese"));
        assert_eq!(get_language_name("xx"), None);
    }
}
