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
    /// Options: madlad-3b, madlad-7b, madlad-10b, marian-en-es, marian-es-en, etc.
    #[arg(short, long)]
    model: Option<String>,

    /// Force CPU (disable GPU)
    #[arg(long)]
    cpu: bool,

    /// Maximum translation length in tokens
    #[arg(long, default_value = "512")]
    max_length: usize,

    /// List models from the catalog
    #[arg(long)]
    list_models: bool,

    /// Only show cached models when listing
    #[arg(long)]
    cached_only: bool,

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
        return list_models(args.cached_only);
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

    info!("Translating: {} -> {}", args.source, args.target);
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
        }) as Box<dyn Fn(usize, usize) + Send + Sync>)
    } else {
        None
    };

    // Run translation
    let result = translate_text_with_progress(&input_text, params, progress_callback, None)?;

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
        "marian-en-ro" => TranslationModel::MarianEnRo,
        "marian-ro-en" => TranslationModel::MarianRoEn,
        "marian-en-nl" => TranslationModel::MarianEnNl,
        "marian-nl-en" => TranslationModel::MarianNlEn,
        "marian-en-sv" => TranslationModel::MarianEnSv,
        "marian-sv-en" => TranslationModel::MarianSvEn,
        "marian-en-da" => TranslationModel::MarianEnDa,
        "marian-da-en" => TranslationModel::MarianDaEn,
        "marian-en-no" => TranslationModel::MarianEnNo,
        "marian-no-en" => TranslationModel::MarianNoEn,
        "marian-en-ru" => TranslationModel::MarianEnRu,
        "marian-ru-en" => TranslationModel::MarianRuEn,
        "marian-en-pl" => TranslationModel::MarianEnPl,
        "marian-pl-en" => TranslationModel::MarianPlEn,
        "marian-en-cs" => TranslationModel::MarianEnCs,
        "marian-cs-en" => TranslationModel::MarianCsEn,
        "marian-en-uk" => TranslationModel::MarianEnUk,
        "marian-uk-en" => TranslationModel::MarianUkEn,
        "marian-en-zh" => TranslationModel::MarianEnZh,
        "marian-zh-en" => TranslationModel::MarianZhEn,
        "marian-en-ja" => TranslationModel::MarianEnJa,
        "marian-ja-en" => TranslationModel::MarianJaEn,
        "marian-en-ko" => TranslationModel::MarianEnKo,
        "marian-ko-en" => TranslationModel::MarianKoEn,
        "marian-en-vi" => TranslationModel::MarianEnVi,
        "marian-vi-en" => TranslationModel::MarianViEn,
        "marian-en-th" => TranslationModel::MarianEnTh,
        "marian-th-en" => TranslationModel::MarianThEn,
        "marian-en-id" => TranslationModel::MarianEnId,
        "marian-id-en" => TranslationModel::MarianIdEn,
        "marian-en-ar" => TranslationModel::MarianEnAr,
        "marian-ar-en" => TranslationModel::MarianArEn,
        "marian-en-he" => TranslationModel::MarianEnHe,
        "marian-he-en" => TranslationModel::MarianHeEn,
        "marian-en-fa" => TranslationModel::MarianEnFa,
        "marian-fa-en" => TranslationModel::MarianFaEn,
        "marian-en-tr" => TranslationModel::MarianEnTr,
        "marian-tr-en" => TranslationModel::MarianTrEn,
        "marian-en-hi" => TranslationModel::MarianEnHi,
        "marian-hi-en" => TranslationModel::MarianHiEn,
        "marian-en-bn" => TranslationModel::MarianEnBn,
        "marian-bn-en" => TranslationModel::MarianBnEn,
        "marian-en-ur" => TranslationModel::MarianEnUr,
        "marian-ur-en" => TranslationModel::MarianUrEn,
        "marian-en-hu" => TranslationModel::MarianEnHu,
        "marian-hu-en" => TranslationModel::MarianHuEn,
        "marian-en-fi" => TranslationModel::MarianEnFi,
        "marian-fi-en" => TranslationModel::MarianFiEn,
        "marian-en-el" => TranslationModel::MarianEnEl,
        "marian-el-en" => TranslationModel::MarianElEn,
        "marian-en-sw" => TranslationModel::MarianEnSw,
        "marian-sw-en" => TranslationModel::MarianSwEn,
        _ => anyhow::bail!("Invalid model: {}", s),
    };
    Ok(model)
}

/// List translation models from the catalog
fn list_models(cached_only: bool) -> anyhow::Result<()> {
    let manager = ModelManager::new()
        .map_err(|e| anyhow::anyhow!("Failed to initialize model manager: {}", e))?;

    let mut catalog = manager
        .list_catalog_models()
        .map_err(|e| anyhow::anyhow!("Failed to list models: {}", e))?;

    if cached_only {
        catalog.retain(|entry| entry.cached);
    }

    if catalog.is_empty() {
        if cached_only {
            println!("No cached models found.");
        } else {
            println!("No models found in catalog.");
        }
        println!("Models will be downloaded automatically on first use.");
        return Ok(());
    }

    println!("\nTranslation Models:");
    println!(
        "{:<20} {:<7} {:<11} {:>9} {:>8}",
        "Model Key", "Family", "Pair", "Size (MB)", "Cached"
    );
    println!("{}", "-".repeat(62));

    let mut total_size = 0u64;
    let mut cached_size = 0u64;
    let mut cached_count = 0u64;

    for entry in &catalog {
        let model = &entry.model;
        let pair = match (model.source.as_deref(), model.target.as_deref()) {
            (Some(source), Some(target)) => format!("{}→{}", source, target),
            _ => "-".to_string(),
        };
        println!(
            "{:<20} {:<7} {:<11} {:>9} {:>8}",
            model.name,
            model.family.as_str(),
            pair,
            model.size_mb,
            if entry.cached { "yes" } else { "no" }
        );

        total_size += model.size_mb;
        if entry.cached {
            cached_size += model.size_mb;
            cached_count += 1;
        }
    }

    println!("{}", "-".repeat(62));
    println!(
        "Total models: {} | Cached: {} | Total size: {} MB | Cached size: {} MB",
        catalog.len(),
        cached_count,
        total_size,
        cached_size
    );
    println!("Cache location: {}", manager.cache_dir().display());

    Ok(())
}
