use clap::Parser;
use hermeneia_lib::translate::{
    model::ModelManager, translate_text_with_progress, TranslateParams, TranslationModel,
};
use indicatif::{ProgressBar, ProgressStyle};
use std::fs;
use tracing::info;

#[derive(Parser, Debug)]
#[command(name = "translate")]
#[command(about = "Translate text using neural machine translation")]
struct Args {
    /// Input text to translate (use --input-file for file input)
    #[arg(short, long, conflicts_with = "input_file")]
    text: Option<String>,

    /// Input text file
    #[arg(short = 'i', long)]
    input_file: Option<String>,

    /// Output file (optional, prints to stdout if not specified)
    #[arg(short, long)]
    output: Option<String>,

    /// Source language code (ISO 639-1, e.g., "en", "es", "fr")
    #[arg(long, default_value = "en")]
    source: String,

    /// Target language code (ISO 639-1, e.g., "en", "es", "fr")
    #[arg(long, default_value = "es")]
    target: String,

    /// Preferred model (auto-selects if not specified)
    /// Options: madlad-3b, madlad-7b, madlad-10b, t5-small, t5-base, t5-large,
    /// flan-t5-small, flan-t5-base, flan-t5-large, flan-ul2, marian-en-es, etc.
    #[arg(short, long)]
    model: Option<String>,

    /// Force CPU (disable GPU)
    #[arg(long)]
    cpu: bool,

    /// Maximum translation length in tokens
    #[arg(long, default_value = "512")]
    max_length: usize,

    /// List all cached models
    #[arg(long)]
    list_models: bool,

    /// Disable progress bar
    #[arg(long)]
    no_progress: bool,
}

fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    let args = Args::parse();

    // Handle --list-models command
    if args.list_models {
        return list_cached_models();
    }

    // Get input text
    let input_text = if let Some(text) = args.text {
        text
    } else if let Some(file_path) = args.input_file {
        fs::read_to_string(&file_path)
            .map_err(|e| anyhow::anyhow!("Failed to read input file '{}': {}", file_path, e))?
    } else {
        anyhow::bail!("Either --text or --input-file must be specified");
    };

    if input_text.trim().is_empty() {
        anyhow::bail!("Input text is empty");
    }

    // Parse model if specified
    let preferred_model = if let Some(model_str) = args.model {
        Some(parse_model(&model_str)?)
    } else {
        None
    };

    info!(
        "Translating: {} -> {}",
        args.source, args.target
    );
    if let Some(model) = preferred_model {
        info!("Using model: {}", model.display_name());
    } else {
        info!("Auto-selecting best available model");
    }

    // Configure parameters
    let params = TranslateParams {
        source_language: args.source.clone(),
        target_language: args.target.clone(),
        preferred_model,
        fallback_enabled: true,
        force_cpu: args.cpu,
        use_quantized: false,
        max_length: Some(args.max_length),
        temperature: Some(0.0),
        top_p: None,
        repetition_penalty: Some(1.0),
    };

    // Setup progress bar
    let progress_bar = if !args.no_progress {
        let pb = ProgressBar::new(args.max_length as u64);
        pb.set_style(
            ProgressStyle::default_bar()
                .template("[{elapsed_precise}] {bar:40.cyan/blue} {pos}/{len} tokens {msg}")
                .unwrap()
                .progress_chars("=>-"),
        );
        Some(pb)
    } else {
        None
    };

    // Create progress callback
    let progress_callback = if let Some(ref pb) = progress_bar {
        let pb_clone = pb.clone();
        Some(Box::new(move |current: usize, _total: usize| {
            pb_clone.set_position(current as u64);
        })
            as Box<dyn Fn(usize, usize) + Send + Sync>)
    } else {
        None
    };

    // Run translation
    let result = translate_text_with_progress(&input_text, params, progress_callback)?;

    // Finish progress bar
    if let Some(pb) = progress_bar {
        pb.finish_with_message("Done!");
    }

    // Output result
    let output_text = format!(
        "{}\n\n--- Translation Info ---\nModel: {}\nTokens: {}\nTime: {:.2}s\n",
        result.translated_text.trim(),
        result.model_used.display_name(),
        result.token_count,
        result.inference_time,
    );

    if let Some(output_path) = args.output {
        fs::write(&output_path, &output_text)
            .map_err(|e| anyhow::anyhow!("Failed to write output file: {}", e))?;
        info!("Translation saved to: {}", output_path);
    } else {
        // Print to stdout
        println!("\n{}", output_text);
    }

    Ok(())
}

/// Parse model string to TranslationModel enum
fn parse_model(s: &str) -> anyhow::Result<TranslationModel> {
    let model = match s.to_lowercase().as_str() {
        "madlad-3b" | "madlad3b" => TranslationModel::Madlad3B,
        "madlad-7b" | "madlad7b" => TranslationModel::Madlad7B,
        "madlad-10b" | "madlad10b" => TranslationModel::Madlad10B,
        "t5-small" | "t5small" => TranslationModel::T5Small,
        "t5-base" | "t5base" => TranslationModel::T5Base,
        "t5-large" | "t5large" => TranslationModel::T5Large,
        "flan-t5-small" | "flan-t5small" => TranslationModel::FlanT5Small,
        "flan-t5-base" | "flan-t5base" => TranslationModel::FlanT5Base,
        "flan-t5-large" | "flan-t5large" => TranslationModel::FlanT5Large,
        "flan-ul2" | "flanul2" => TranslationModel::FlanUl2,
        "marian-en-es" => TranslationModel::MarianEnEs,
        "marian-es-en" => TranslationModel::MarianEsEn,
        "marian-en-fr" => TranslationModel::MarianEnFr,
        "marian-fr-en" => TranslationModel::MarianFrEn,
        "marian-en-de" => TranslationModel::MarianEnDe,
        "marian-de-en" => TranslationModel::MarianDeEn,
        "marian-en-pt" => TranslationModel::MarianEnPt,
        "marian-pt-en" => TranslationModel::MarianPtEn,
        "marian-en-it" => TranslationModel::MarianEnIt,
        "marian-it-en" => TranslationModel::MarianItEn,
        "marian-en-ru" => TranslationModel::MarianEnRu,
        "marian-ru-en" => TranslationModel::MarianRuEn,
        "marian-en-zh" => TranslationModel::MarianEnZh,
        "marian-zh-en" => TranslationModel::MarianZhEn,
        "marian-en-ja" => TranslationModel::MarianEnJa,
        "marian-ja-en" => TranslationModel::MarianJaEn,
        "marian-en-ko" => TranslationModel::MarianEnKo,
        "marian-ko-en" => TranslationModel::MarianKoEn,
        "marian-en-ar" => TranslationModel::MarianEnAr,
        "marian-ar-en" => TranslationModel::MarianArEn,
        _ => anyhow::bail!("Invalid model: {}", s),
    };
    Ok(model)
}

/// List all cached translation models
fn list_cached_models() -> anyhow::Result<()> {
    let manager = ModelManager::new()
        .map_err(|e| anyhow::anyhow!("Failed to initialize model manager: {}", e))?;

    let cached = manager
        .list_cached_models()
        .map_err(|e| anyhow::anyhow!("Failed to list models: {}", e))?;

    if cached.is_empty() {
        println!("No cached models found.");
        println!("Models will be downloaded automatically on first use.");
        return Ok(());
    }

    println!("\nCached Translation Models:");
    println!("{:<20} {:<45} {:>10}", "Model Key", "Display Name", "Size (MB)");
    println!("{}", "-".repeat(80));

    let mut total_size = 0u64;
    for (model, size_mb) in cached {
        println!(
            "{:<20} {:<45} {:>10}",
            model.cli_key(),
            model.display_name(),
            size_mb
        );
        total_size += size_mb;
    }

    println!("{}", "-".repeat(80));
    println!("{:<20} {:<45} {:>10}", "Total", "", total_size);
    println!("\nCache location: {}", manager.cache_dir().display());

    Ok(())
}
