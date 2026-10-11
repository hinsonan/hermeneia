use crate::translate::types::TranslationModel;

/// Get the best MarianMT model for a specific language pair, if available.
///
/// Only pairs whose catalog entries publish safetensors are listed; the catalog
/// is kept in sync with this mapping.
pub fn get_marian_for_pair(source: &str, target: &str) -> Option<TranslationModel> {
    match (source, target) {
        // Romance Languages
        ("en", "es") => Some(TranslationModel::MarianEnEs),
        ("es", "en") => Some(TranslationModel::MarianEsEn),
        ("fr", "en") => Some(TranslationModel::MarianFrEn),

        // Germanic Languages
        ("en", "de") => Some(TranslationModel::MarianEnDe),
        ("de", "en") => Some(TranslationModel::MarianDeEn),
        ("nl", "en") => Some(TranslationModel::MarianNlEn),
        ("sv", "en") => Some(TranslationModel::MarianSvEn),

        // Slavic Languages
        ("en", "ru") => Some(TranslationModel::MarianEnRu),
        ("ru", "en") => Some(TranslationModel::MarianRuEn),

        // East Asian Languages
        ("en", "ko") => Some(TranslationModel::MarianEnKo),

        // Middle Eastern Languages
        ("en", "ar") => Some(TranslationModel::MarianEnAr),
        ("ar", "en") => Some(TranslationModel::MarianArEn),
        ("he", "en") => Some(TranslationModel::MarianHeEn),

        // South Asian Languages
        ("bn", "en") => Some(TranslationModel::MarianBnEn),

        // Other European Languages
        ("en", "hu") => Some(TranslationModel::MarianEnHu),
        ("fi", "en") => Some(TranslationModel::MarianFiEn),

        // No supported specialized model for this pair
        _ => None,
    }
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
        assert_eq!(
            get_marian_for_pair("nl", "en"),
            Some(TranslationModel::MarianNlEn)
        );
        assert_eq!(get_marian_for_pair("en", "fr"), None);
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
}
