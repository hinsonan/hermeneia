//! SRT Subtitle Parser and Renderer
//!
//! Parses SubRip (.srt) files into segments for translation,
//! then reassembles them with translated text while preserving timestamps.

use serde::{Deserialize, Serialize};

/// A single subtitle cue/segment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubtitleSegment {
    /// Cue index (1-based, as in SRT files)
    pub index: usize,
    /// Start time in the format "HH:MM:SS,mmm"
    pub start: String,
    /// End time in the format "HH:MM:SS,mmm"
    pub end: String,
    /// The text content (may contain multiple lines)
    pub text: String,
}

/// Parsed SRT file
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubtitleFile {
    pub segments: Vec<SubtitleSegment>,
}

impl SubtitleFile {
    /// Parse an SRT file from string content
    pub fn parse(content: &str) -> Result<Self, SubtitleParseError> {
        let mut segments = Vec::new();
        let content = content.replace('\r', ""); // Normalize line endings
        let blocks: Vec<&str> = content
            .split("\n\n")
            .filter(|b| !b.trim().is_empty())
            .collect();

        for block in blocks {
            let lines: Vec<&str> = block.lines().collect();
            if lines.len() < 3 {
                // Skip malformed blocks (need at least index, timestamp, and text)
                continue;
            }

            // Parse index
            let index: usize = lines[0]
                .trim()
                .parse()
                .map_err(|_| SubtitleParseError::InvalidIndex(lines[0].to_string()))?;

            // Parse timestamp line: "HH:MM:SS,mmm --> HH:MM:SS,mmm"
            let timestamp_line = lines[1].trim();
            let (start, end) = parse_timestamp_line(timestamp_line)?;

            // Remaining lines are text (join with newlines to preserve multi-line cues)
            let text = lines[2..].join("\n");

            segments.push(SubtitleSegment {
                index,
                start,
                end,
                text,
            });
        }

        Ok(SubtitleFile { segments })
    }

    /// Render the subtitle file back to SRT format
    pub fn render(&self) -> String {
        let mut out = String::new();
        let mut first = true;

        for seg in &self.segments {
            if !first {
                out.push('\n');
            }
            first = false;

            out.push_str(&seg.index.to_string());
            out.push('\n');
            out.push_str(&seg.start);
            out.push_str(" --> ");
            out.push_str(&seg.end);
            out.push('\n');
            out.push_str(&seg.text);
            out.push('\n');
        }

        out
    }

    /// Create a new SubtitleFile with translated text segments
    /// The translated_texts should be in the same order as self.segments
    pub fn with_translated_text<I, S>(&self, translated_texts: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        let mut translated_iter = translated_texts.into_iter();
        let segments = self
            .segments
            .iter()
            .map(|seg| SubtitleSegment {
                index: seg.index,
                start: seg.start.clone(),
                end: seg.end.clone(),
                text: translated_iter
                    .next()
                    .map(|t| t.as_ref().to_string())
                    .unwrap_or_else(|| seg.text.clone()),
            })
            .collect();

        SubtitleFile { segments }
    }

    /// Create a new SubtitleFile with translated text while preserving
    /// leading Hermeneia speaker labels (e.g., `[Speaker 1]`) verbatim.
    pub fn with_translated_text_preserving_labels<I, S>(&self, translated_texts: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        let mut translated_iter = translated_texts.into_iter();
        let segments = self
            .segments
            .iter()
            .map(|seg| {
                let translated = translated_iter.next().map(|t| t.as_ref().to_string());
                let (label_prefix, body) = split_leading_label_prefix(&seg.text);

                let text = match (label_prefix, translated) {
                    (Some(prefix), Some(translated)) => {
                        let translated_body = if body.trim().is_empty() {
                            body.to_string()
                        } else {
                            translated
                        };
                        format!("{}{}", prefix, translated_body)
                    }
                    (None, Some(translated)) => translated,
                    (_, None) => seg.text.clone(),
                };

                SubtitleSegment {
                    index: seg.index,
                    start: seg.start.clone(),
                    end: seg.end.clone(),
                    text,
                }
            })
            .collect();

        SubtitleFile { segments }
    }

    /// Get just the text content for translation (preserving order)
    pub fn get_texts(&self) -> Vec<String> {
        self.segments.iter().map(|s| s.text.clone()).collect()
    }

    /// Get text content prepared for translation while preserving leading
    /// Hermeneia speaker labels by excluding them from translation input.
    pub fn get_texts_for_translation(&self) -> Vec<String> {
        self.get_texts_for_translation_ref()
            .into_iter()
            .map(str::to_string)
            .collect()
    }

    /// Get borrowed text slices prepared for translation while preserving
    /// leading Hermeneia speaker labels by excluding them from translation input.
    pub fn get_texts_for_translation_ref(&self) -> Vec<&str> {
        self.segments
            .iter()
            .map(|s| {
                let (_, body) = split_leading_label_prefix(&s.text);
                body
            })
            .collect()
    }

    /// Get the number of segments
    pub fn len(&self) -> usize {
        self.segments.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.segments.is_empty()
    }
}

/// Parse a timestamp line like "00:01:23,456 --> 00:01:25,789"
fn parse_timestamp_line(line: &str) -> Result<(String, String), SubtitleParseError> {
    let parts: Vec<&str> = line.split("-->").collect();
    if parts.len() != 2 {
        return Err(SubtitleParseError::InvalidTimestamp(line.to_string()));
    }

    let start = parts[0].trim().to_string();
    let end = parts[1].trim().to_string();

    // Basic validation
    if !is_valid_timestamp(&start) || !is_valid_timestamp(&end) {
        return Err(SubtitleParseError::InvalidTimestamp(line.to_string()));
    }

    Ok((start, end))
}

/// Validate SRT timestamp format: "HH:MM:SS,mmm"
fn is_valid_timestamp(ts: &str) -> bool {
    // Should match pattern like "00:01:23,456"
    let parts: Vec<&str> = ts.split(',').collect();
    if parts.len() != 2 {
        return false;
    }

    let time_parts: Vec<&str> = parts[0].split(':').collect();
    if time_parts.len() != 3 {
        return false;
    }

    // All parts should be numeric
    time_parts
        .iter()
        .all(|p| p.chars().all(|c| c.is_ascii_digit()))
        && parts[1].chars().all(|c| c.is_ascii_digit())
}

fn split_leading_label_prefix(text: &str) -> (Option<&str>, &str) {
    if !text.starts_with('[') {
        return (None, text);
    }

    let mut closing_bracket_idx = None;
    for (idx, ch) in text.char_indices().skip(1) {
        if ch == '\n' || ch == '\r' {
            return (None, text);
        }
        if ch == ']' {
            if idx == 1 {
                return (None, text);
            }
            closing_bracket_idx = Some(idx);
            break;
        }
    }

    let Some(closing_bracket_idx) = closing_bracket_idx else {
        return (None, text);
    };

    let mut prefix_end = closing_bracket_idx + 1;
    while let Some(ch) = text[prefix_end..].chars().next() {
        if ch == ' ' || ch == '\t' {
            prefix_end += ch.len_utf8();
        } else {
            break;
        }
    }

    if text[prefix_end..].starts_with('\n') {
        prefix_end += '\n'.len_utf8();
        while let Some(ch) = text[prefix_end..].chars().next() {
            if ch == ' ' || ch == '\t' {
                prefix_end += ch.len_utf8();
            } else {
                break;
            }
        }
    }

    let label = &text[1..closing_bracket_idx];
    let body = &text[prefix_end..];

    if !is_preserved_speaker_label(label, body) {
        return (None, text);
    }

    (Some(&text[..prefix_end]), body)
}

fn is_preserved_speaker_label(label: &str, body: &str) -> bool {
    let trimmed = label.trim();

    if is_generated_speaker_label(trimmed) {
        return true;
    }

    if body.trim().is_empty()
        || is_known_bracketed_caption(trimmed)
        || is_caption_like_label(trimmed)
    {
        return false;
    }

    is_custom_speaker_name(trimmed)
}

fn is_generated_speaker_label(label: &str) -> bool {
    if let Some(speaker) = label.strip_prefix("Speaker ") {
        return !speaker.is_empty() && speaker.chars().all(|ch| ch.is_ascii_digit());
    }

    false
}

fn is_known_bracketed_caption(label: &str) -> bool {
    matches!(
        label.to_ascii_lowercase().as_str(),
        "music" | "laughter" | "applause" | "laughs" | "sighs" | "inaudible"
    )
}

fn is_caption_like_label(label: &str) -> bool {
    label.split_whitespace().any(|word| {
        let normalized = word
            .trim_matches(|ch: char| !ch.is_ascii_alphabetic())
            .to_ascii_lowercase();

        matches!(
            normalized.as_str(),
            "applause"
                | "audience"
                | "cheering"
                | "clapping"
                | "closes"
                | "closing"
                | "crash"
                | "crashing"
                | "cough"
                | "coughing"
                | "crowd"
                | "crying"
                | "door"
                | "doors"
                | "footsteps"
                | "laughing"
                | "laughs"
                | "music"
                | "opens"
                | "opening"
                | "phone"
                | "phones"
                | "rumbles"
                | "rumbling"
                | "scream"
                | "screaming"
                | "shout"
                | "shouting"
                | "rings"
                | "ringing"
                | "silence"
                | "sighing"
                | "sighs"
                | "sobbing"
                | "thunder"
                | "whispering"
        )
    })
}

fn is_custom_speaker_name(label: &str) -> bool {
    !label.is_empty() && label.split_whitespace().all(is_title_case_name_word)
}

fn is_title_case_name_word(word: &str) -> bool {
    let mut chars = word.chars();
    let Some(first_char) = chars.next() else {
        return false;
    };

    first_char.is_uppercase()
        && chars.all(|ch| ch.is_alphabetic() || ch == '-' || ch == '\'' || ch == '.')
}

/// Errors that can occur during SRT parsing
#[derive(Debug, Clone)]
pub enum SubtitleParseError {
    InvalidIndex(String),
    InvalidTimestamp(String),
    EmptyFile,
}

impl std::fmt::Display for SubtitleParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidIndex(s) => write!(f, "Invalid subtitle index: {}", s),
            Self::InvalidTimestamp(s) => write!(f, "Invalid timestamp format: {}", s),
            Self::EmptyFile => write!(f, "Empty or invalid SRT file"),
        }
    }
}

impl std::error::Error for SubtitleParseError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_and_render_roundtrip() {
        let srt_content = "1\n00:00:01,000 --> 00:00:03,000\nHello world\n\n2\n00:00:03,500 --> 00:00:05,500\nHow are you today?\n\n3\n00:00:06,000 --> 00:00:08,000\nThis is a test\n";

        let srt_file = SubtitleFile::parse(srt_content).unwrap();

        assert_eq!(srt_file.len(), 3);
        assert_eq!(srt_file.segments[0].index, 1);
        assert_eq!(srt_file.segments[0].text, "Hello world");
        assert_eq!(srt_file.segments[1].index, 2);
        assert_eq!(srt_file.segments[1].text, "How are you today?");
        assert_eq!(srt_file.segments[2].index, 3);
        assert_eq!(srt_file.segments[2].text, "This is a test");
    }

    #[test]
    fn test_get_texts_preserves_order() {
        let srt_content = "1\n00:00:01,000 --> 00:00:03,000\nFirst\n\n2\n00:00:03,500 --> 00:00:05,500\nSecond\n\n3\n00:00:06,000 --> 00:00:08,000\nThird\n";

        let srt_file = SubtitleFile::parse(srt_content).unwrap();
        let texts = srt_file.get_texts();

        assert_eq!(texts.len(), 3);
        assert_eq!(texts[0], "First");
        assert_eq!(texts[1], "Second");
        assert_eq!(texts[2], "Third");
    }

    #[test]
    fn test_with_translated_text_preserves_timestamps() {
        let srt_content =
            "1\n00:00:01,000 --> 00:00:03,000\nHello\n\n2\n00:00:03,500 --> 00:00:05,500\nWorld\n";

        let srt_file = SubtitleFile::parse(srt_content).unwrap();
        let translated = vec!["Hola".to_string(), "Mundo".to_string()];
        let new_srt = srt_file.with_translated_text(translated);

        assert_eq!(new_srt.segments[0].start, "00:00:01,000");
        assert_eq!(new_srt.segments[0].end, "00:00:03,000");
        assert_eq!(new_srt.segments[0].text, "Hola");

        assert_eq!(new_srt.segments[1].start, "00:00:03,500");
        assert_eq!(new_srt.segments[1].end, "00:00:05,500");
        assert_eq!(new_srt.segments[1].text, "Mundo");
    }

    #[test]
    fn test_render_format() {
        let srt_content =
            "1\n00:00:01,000 --> 00:00:03,000\nHello\n\n2\n00:00:03,500 --> 00:00:05,500\nWorld\n";

        let srt_file = SubtitleFile::parse(srt_content).unwrap();
        let rendered = srt_file.render();

        // Check that rendered output contains proper format
        assert!(rendered.contains("1\n00:00:01,000 --> 00:00:03,000\nHello"));
        assert!(rendered.contains("2\n00:00:03,500 --> 00:00:05,500\nWorld"));
    }

    #[test]
    fn test_multiline_text() {
        let srt_content = "1\n00:00:01,000 --> 00:00:03,000\nLine one\nLine two\n\n2\n00:00:03,500 --> 00:00:05,500\nSingle line\n";

        let srt_file = SubtitleFile::parse(srt_content).unwrap();

        assert_eq!(srt_file.segments[0].text, "Line one\nLine two");
        assert_eq!(srt_file.segments[1].text, "Single line");
    }

    #[test]
    fn test_get_texts_for_translation_strips_leading_speaker_labels() {
        let srt_content = "1\n00:00:01,000 --> 00:00:03,000\n[Speaker 1] Hello world\n\n2\n00:00:03,500 --> 00:00:05,500\nNo label here\n";
        let srt_file = SubtitleFile::parse(srt_content).unwrap();

        let texts = srt_file.get_texts_for_translation();
        assert_eq!(texts, vec!["Hello world", "No label here"]);
    }

    #[test]
    fn test_get_texts_for_translation_keeps_bracketed_captions_translatable() {
        let srt_content = "1\n00:00:01,000 --> 00:00:03,000\n[Music]\n\n2\n00:00:03,500 --> 00:00:05,500\n[laughter] continues\n";
        let srt_file = SubtitleFile::parse(srt_content).unwrap();

        let texts = srt_file.get_texts_for_translation();
        assert_eq!(texts, vec!["[Music]", "[laughter] continues"]);
    }

    #[test]
    fn test_title_case_bracketed_captions_remain_translatable() {
        let srt_content = "1\n00:00:01,000 --> 00:00:03,000\n[Door Opens]\n\n2\n00:00:03,500 --> 00:00:05,500\n[Phone Rings] Hello\n\n3\n00:00:06,000 --> 00:00:07,000\n[Audience Laughs]\n\n4\n00:00:07,500 --> 00:00:08,500\n[Thunder] rumbles\n\n5\n00:00:09,000 --> 00:00:10,000\n[Birds Chirping]\n";
        let srt_file = SubtitleFile::parse(srt_content).unwrap();

        let texts = srt_file.get_texts_for_translation();
        assert_eq!(
            texts,
            vec![
                "[Door Opens]",
                "[Phone Rings] Hello",
                "[Audience Laughs]",
                "[Thunder] rumbles",
                "[Birds Chirping]"
            ]
        );

        let rebuilt = srt_file.with_translated_text_preserving_labels(vec![
            "[Se abre la puerta]".to_string(),
            "[Suena el telefono] Hola".to_string(),
            "[El publico rie]".to_string(),
            "[Trueno] retumba".to_string(),
            "[Pajaros cantando]".to_string(),
        ]);

        assert_eq!(rebuilt.segments[0].text, "[Se abre la puerta]");
        assert_eq!(rebuilt.segments[1].text, "[Suena el telefono] Hola");
        assert_eq!(rebuilt.segments[2].text, "[El publico rie]");
        assert_eq!(rebuilt.segments[3].text, "[Trueno] retumba");
        assert_eq!(rebuilt.segments[4].text, "[Pajaros cantando]");
    }

    #[test]
    fn test_with_translated_text_preserving_labels_keeps_speaker_prefix() {
        let srt_content = "1\n00:00:01,000 --> 00:00:03,000\n[Speaker 1] Hello\n\n2\n00:00:03,500 --> 00:00:05,500\n[Speaker 23] World\n";
        let srt_file = SubtitleFile::parse(srt_content).unwrap();

        let translated = vec!["Hola".to_string(), "Mundo".to_string()];
        let rebuilt = srt_file.with_translated_text_preserving_labels(translated);

        assert_eq!(rebuilt.segments[0].text, "[Speaker 1] Hola");
        assert_eq!(rebuilt.segments[1].text, "[Speaker 23] Mundo");
    }

    #[test]
    fn test_with_translated_text_preserving_labels_does_not_preserve_bracketed_captions() {
        let srt_content = "1\n00:00:01,000 --> 00:00:03,000\n[Music]\n";
        let srt_file = SubtitleFile::parse(srt_content).unwrap();

        let rebuilt = srt_file.with_translated_text_preserving_labels(vec!["[Musica]".to_string()]);

        assert_eq!(rebuilt.segments[0].text, "[Musica]");
    }

    #[test]
    fn test_speaker_like_captions_remain_translatable() {
        let srt_content = "1\n00:00:01,000 --> 00:00:03,000\n[Speaker 1 laughs] Hello\n\n2\n00:00:03,500 --> 00:00:05,500\n[Speaker 1A] World\n";
        let srt_file = SubtitleFile::parse(srt_content).unwrap();

        let texts = srt_file.get_texts_for_translation();
        assert_eq!(
            texts,
            vec!["[Speaker 1 laughs] Hello", "[Speaker 1A] World"]
        );
    }

    #[test]
    fn test_custom_speaker_label_is_preserved_when_prefixed_to_dialogue() {
        let srt_content = "1\n00:00:01,000 --> 00:00:03,000\n[Alice] Hello\n\n2\n00:00:03,500 --> 00:00:05,000\n[Pastor John] Welcome\n";
        let srt_file = SubtitleFile::parse(srt_content).unwrap();

        let texts = srt_file.get_texts_for_translation();
        assert_eq!(texts, vec!["Hello", "Welcome"]);

        let rebuilt = srt_file.with_translated_text_preserving_labels(vec![
            "Hola".to_string(),
            "Bienvenido".to_string(),
        ]);
        assert_eq!(rebuilt.segments[0].text, "[Alice] Hola");
        assert_eq!(rebuilt.segments[1].text, "[Pastor John] Bienvenido");
    }

    #[test]
    fn test_custom_speaker_label_supports_unicode_and_name_punctuation() {
        let srt_content = "1\n00:00:01,000 --> 00:00:03,000\n[José] Hola\n\n2\n00:00:03,500 --> 00:00:05,000\n[Élodie] Bonjour\n\n3\n00:00:05,500 --> 00:00:07,000\n[Dr. Smith] Hello\n\n4\n00:00:07,500 --> 00:00:09,000\n[Pastor John Jr.] Welcome\n";
        let srt_file = SubtitleFile::parse(srt_content).unwrap();

        let texts = srt_file.get_texts_for_translation();
        assert_eq!(texts, vec!["Hola", "Bonjour", "Hello", "Welcome"]);

        let rebuilt = srt_file.with_translated_text_preserving_labels(vec![
            "Hola".to_string(),
            "Bonjour".to_string(),
            "Hola".to_string(),
            "Bienvenido".to_string(),
        ]);

        assert_eq!(rebuilt.segments[0].text, "[José] Hola");
        assert_eq!(rebuilt.segments[1].text, "[Élodie] Bonjour");
        assert_eq!(rebuilt.segments[2].text, "[Dr. Smith] Hola");
        assert_eq!(rebuilt.segments[3].text, "[Pastor John Jr.] Bienvenido");
    }

    #[test]
    fn test_lowercase_bracketed_caption_with_dialogue_stays_translatable() {
        let srt_content = "1\n00:00:01,000 --> 00:00:03,000\n[sighs] Hello\n";
        let srt_file = SubtitleFile::parse(srt_content).unwrap();

        let texts = srt_file.get_texts_for_translation();
        assert_eq!(texts, vec!["[sighs] Hello"]);
    }

    #[test]
    fn test_whitespace_only_bracketed_prefix_stays_translatable() {
        let srt_content = "1\n00:00:01,000 --> 00:00:03,000\n[   ] Hello\n";
        let srt_file = SubtitleFile::parse(srt_content).unwrap();

        let texts = srt_file.get_texts_for_translation();
        assert_eq!(texts, vec!["[   ] Hello"]);
    }

    #[test]
    fn test_missing_translation_does_not_duplicate_speaker_label() {
        let srt_content = "1\n00:00:01,000 --> 00:00:03,000\n[Speaker 1] Hello\n";
        let srt_file = SubtitleFile::parse(srt_content).unwrap();

        let rebuilt = srt_file.with_translated_text_preserving_labels(Vec::<String>::new());
        assert_eq!(rebuilt.segments[0].text, "[Speaker 1] Hello");
    }

    #[test]
    fn test_with_translated_text_preserving_labels_mixed_segments() {
        let srt_content = "1\n00:00:01,000 --> 00:00:03,000\n[Speaker 1] First line\n\n2\n00:00:03,500 --> 00:00:05,500\nSecond line\n";
        let srt_file = SubtitleFile::parse(srt_content).unwrap();

        let translated = vec!["Primera linea".to_string(), "Segunda linea".to_string()];
        let rebuilt = srt_file.with_translated_text_preserving_labels(translated);

        assert_eq!(rebuilt.segments[0].text, "[Speaker 1] Primera linea");
        assert_eq!(rebuilt.segments[1].text, "Segunda linea");
    }

    #[test]
    fn test_with_translated_text_preserving_labels_supports_label_line_then_text() {
        let srt_content = "1\n00:00:01,000 --> 00:00:03,000\n[Speaker 1]\nHello world\n";
        let srt_file = SubtitleFile::parse(srt_content).unwrap();

        let texts = srt_file.get_texts_for_translation();
        assert_eq!(texts, vec!["Hello world"]);

        let rebuilt =
            srt_file.with_translated_text_preserving_labels(vec!["Hola mundo".to_string()]);
        assert_eq!(rebuilt.segments[0].text, "[Speaker 1]\nHola mundo");
    }

    #[test]
    fn test_with_translated_text_preserving_labels_label_only_stays_label_only() {
        let srt_content = "1\n00:00:01,000 --> 00:00:03,000\n[Speaker 1]\n";
        let srt_file = SubtitleFile::parse(srt_content).unwrap();

        let texts = srt_file.get_texts_for_translation();
        assert_eq!(texts, vec![""]);

        let rebuilt =
            srt_file.with_translated_text_preserving_labels(vec!["Hallucinated text".to_string()]);
        assert_eq!(rebuilt.segments[0].text, "[Speaker 1]");
    }

    #[test]
    fn test_split_leading_label_prefix_ignores_malformed_prefix() {
        let (prefix, body) = split_leading_label_prefix("[Speaker 1 Hello");
        assert_eq!(prefix, None);
        assert_eq!(body, "[Speaker 1 Hello");
    }
}
