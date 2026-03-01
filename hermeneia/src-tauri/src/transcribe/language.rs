use crate::error::{AudioError, Result};
use candle_core::{Device, IndexOp, Tensor};
use candle_nn::ops::softmax;
use candle_transformers::models::whisper::{self as m, model::Whisper};
use tokenizers::Tokenizer;

/// Detect language from audio mel-spectrogram
pub fn detect_language(
    model: &mut Whisper,
    tokenizer: &Tokenizer,
    mel: &Tensor,
    device: &Device,
) -> Result<u32> {
    // Trim mel to max source positions
    let (_, _, content_frames) = mel
        .dims3()
        .map_err(|e| AudioError::TranscriptionFailed(format!("Dims3: {}", e)))?;
    let mel_segment = if content_frames > m::N_FRAMES {
        mel.narrow(2, 0, m::N_FRAMES)
            .map_err(|e| AudioError::TranscriptionFailed(format!("Narrow: {}", e)))?
    } else {
        mel.clone()
    };

    // Encode audio features
    let audio_features = model
        .encoder
        .forward(&mel_segment, true)
        .map_err(|e| AudioError::TranscriptionFailed(format!("Encoder failed: {}", e)))?;

    // Get SOT token
    let sot_token = token_id(tokenizer, m::SOT_TOKEN)?;
    let tokens = Tensor::new(&[sot_token], device)
        .map_err(|e| AudioError::TranscriptionFailed(format!("Token tensor: {}", e)))?
        .unsqueeze(0)
        .map_err(|e| AudioError::TranscriptionFailed(format!("Unsqueeze: {}", e)))?;

    // Run decoder forward pass
    let ys = model
        .decoder
        .forward(&tokens, &audio_features, true)
        .map_err(|e| AudioError::TranscriptionFailed(format!("Decoder failed: {}", e)))?;

    // Get logits for first token
    let logits = model
        .decoder
        .final_linear(
            &ys.i(..1)
                .map_err(|e| AudioError::TranscriptionFailed(format!("Index error: {}", e)))?,
        )
        .map_err(|e| AudioError::TranscriptionFailed(format!("Final linear: {}", e)))?
        .i(0)
        .map_err(|e| AudioError::TranscriptionFailed(format!("Index error: {}", e)))?
        .i(0)
        .map_err(|e| AudioError::TranscriptionFailed(format!("Index error: {}", e)))?;

    // Get all language token IDs
    let language_token_ids = LANGUAGES
        .iter()
        .filter_map(|(code, _)| token_id(tokenizer, &format!("<|{code}|>")).ok())
        .collect::<Vec<_>>();

    let language_token_ids_tensor = Tensor::new(language_token_ids.as_slice(), device)
        .map_err(|e| AudioError::TranscriptionFailed(format!("Language tensor: {}", e)))?;

    // Select logits for language tokens
    let language_logits = logits
        .index_select(&language_token_ids_tensor, 0)
        .map_err(|e| AudioError::TranscriptionFailed(format!("Index select: {}", e)))?;

    // Apply softmax to get probabilities
    let probs = softmax(&language_logits, 0)
        .map_err(|e| AudioError::TranscriptionFailed(format!("Softmax: {}", e)))?;
    let probs_vec: Vec<f32> = probs
        .to_vec1()
        .map_err(|e| AudioError::TranscriptionFailed(format!("To vec: {}", e)))?;

    // Find language with highest probability
    let mut lang_probs: Vec<_> = probs_vec
        .iter()
        .enumerate()
        .map(|(i, &p)| (p, language_token_ids[i], LANGUAGES[i].0))
        .collect();
    lang_probs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());

    // Log top 5 detected languages
    tracing::info!("Detected languages:");
    for (prob, _token_id, lang_code) in lang_probs.iter().take(5) {
        tracing::info!("  {}: {:.2}%", lang_code, prob * 100.0);
    }

    let detected_lang = lang_probs[0].1;
    Ok(detected_lang)
}

/// Get token ID from tokenizer
fn token_id(tokenizer: &Tokenizer, token: &str) -> Result<u32> {
    tokenizer
        .token_to_id(token)
        .ok_or_else(|| AudioError::TranscriptionFailed(format!("No token ID for: {}", token)))
}

/// Language codes and names
const LANGUAGES: [(&str, &str); 99] = [
    ("en", "english"),
    ("zh", "chinese"),
    ("de", "german"),
    ("es", "spanish"),
    ("ru", "russian"),
    ("ko", "korean"),
    ("fr", "french"),
    ("ja", "japanese"),
    ("pt", "portuguese"),
    ("tr", "turkish"),
    ("pl", "polish"),
    ("ca", "catalan"),
    ("nl", "dutch"),
    ("ar", "arabic"),
    ("sv", "swedish"),
    ("it", "italian"),
    ("id", "indonesian"),
    ("hi", "hindi"),
    ("fi", "finnish"),
    ("vi", "vietnamese"),
    ("he", "hebrew"),
    ("uk", "ukrainian"),
    ("el", "greek"),
    ("ms", "malay"),
    ("cs", "czech"),
    ("ro", "romanian"),
    ("da", "danish"),
    ("hu", "hungarian"),
    ("ta", "tamil"),
    ("no", "norwegian"),
    ("th", "thai"),
    ("ur", "urdu"),
    ("hr", "croatian"),
    ("bg", "bulgarian"),
    ("lt", "lithuanian"),
    ("la", "latin"),
    ("mi", "maori"),
    ("ml", "malayalam"),
    ("cy", "welsh"),
    ("sk", "slovak"),
    ("te", "telugu"),
    ("fa", "persian"),
    ("lv", "latvian"),
    ("bn", "bengali"),
    ("sr", "serbian"),
    ("az", "azerbaijani"),
    ("sl", "slovenian"),
    ("kn", "kannada"),
    ("et", "estonian"),
    ("mk", "macedonian"),
    ("br", "breton"),
    ("eu", "basque"),
    ("is", "icelandic"),
    ("hy", "armenian"),
    ("ne", "nepali"),
    ("mn", "mongolian"),
    ("bs", "bosnian"),
    ("kk", "kazakh"),
    ("sq", "albanian"),
    ("sw", "swahili"),
    ("gl", "galician"),
    ("mr", "marathi"),
    ("pa", "punjabi"),
    ("si", "sinhala"),
    ("km", "khmer"),
    ("sn", "shona"),
    ("yo", "yoruba"),
    ("so", "somali"),
    ("af", "afrikaans"),
    ("oc", "occitan"),
    ("ka", "georgian"),
    ("be", "belarusian"),
    ("tg", "tajik"),
    ("sd", "sindhi"),
    ("gu", "gujarati"),
    ("am", "amharic"),
    ("yi", "yiddish"),
    ("lo", "lao"),
    ("uz", "uzbek"),
    ("fo", "faroese"),
    ("ht", "haitian creole"),
    ("ps", "pashto"),
    ("tk", "turkmen"),
    ("nn", "nynorsk"),
    ("mt", "maltese"),
    ("sa", "sanskrit"),
    ("lb", "luxembourgish"),
    ("my", "myanmar"),
    ("bo", "tibetan"),
    ("tl", "tagalog"),
    ("mg", "malagasy"),
    ("as", "assamese"),
    ("tt", "tatar"),
    ("haw", "hawaiian"),
    ("ln", "lingala"),
    ("ha", "hausa"),
    ("ba", "bashkir"),
    ("jw", "javanese"),
    ("su", "sundanese"),
];
