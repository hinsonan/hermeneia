use crate::translate::types::TranslationModel;

/// Get the best MarianMT model for a specific language pair, if available
pub fn get_marian_for_pair(source: &str, target: &str) -> Option<TranslationModel> {
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
        assert_eq!(
            get_marian_for_pair("en", "nl"),
            Some(TranslationModel::MarianEnNl)
        );
        assert_eq!(
            get_marian_for_pair("sv", "en"),
            Some(TranslationModel::MarianSvEn)
        );
        assert_eq!(
            get_marian_for_pair("en", "sw"),
            Some(TranslationModel::MarianEnSw)
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
