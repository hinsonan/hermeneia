use std::path::PathBuf;

/// Resolve the Hugging Face hub cache directory using hf-hub's path semantics.
///
/// hf-hub uses:
/// - `HF_HOME/hub` when `HF_HOME` is set
/// - otherwise `<home>/.cache/huggingface/hub`
pub fn hf_hub_cache_dir() -> PathBuf {
    if let Some(home) = std::env::var_os("HF_HOME") {
        let mut path = PathBuf::from(home);
        path.push("hub");
        return path;
    }

    let mut path = dirs::home_dir().unwrap_or_else(|| PathBuf::from("."));
    path.push(".cache");
    path.push("huggingface");
    path.push("hub");
    path
}
