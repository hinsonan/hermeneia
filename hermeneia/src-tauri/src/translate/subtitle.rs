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
        self.segments
            .iter()
            .map(|seg| {
                format!(
                    "{}\n{} --> {}\n{}\n",
                    seg.index, seg.start, seg.end, seg.text
                )
            })
            .collect::<Vec<_>>()
            .join("\n")
    }

    /// Create a new SubtitleFile with translated text segments
    /// The translated_texts should be in the same order as self.segments
    pub fn with_translated_text(&self, translated_texts: Vec<String>) -> Self {
        let segments = self
            .segments
            .iter()
            .zip(translated_texts.into_iter())
            .map(|(seg, text)| SubtitleSegment {
                index: seg.index,
                start: seg.start.clone(),
                end: seg.end.clone(),
                text,
            })
            .collect();

        SubtitleFile { segments }
    }

    /// Get just the text content for translation (preserving order)
    pub fn get_texts(&self) -> Vec<String> {
        self.segments.iter().map(|s| s.text.clone()).collect()
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
