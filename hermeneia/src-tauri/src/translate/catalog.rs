use crate::error::{AudioError, Result};
use serde::Deserialize;

const MODELS_TOML: &str = include_str!("models.toml");

#[derive(Debug, Deserialize)]
struct ModelsToml {
    #[serde(default)]
    madlad: Vec<MadladToml>,
    #[serde(default)]
    marian: Vec<MarianToml>,
}

#[derive(Debug, Deserialize)]
struct MadladToml {
    name: String,
    model_id: String,
    size_mb: u64,
    revision: Option<String>,
    description: Option<String>,
    has_safetensors: Option<bool>,
}

#[derive(Debug, Deserialize)]
struct MarianToml {
    name: String,
    model_id: String,
    source: String,
    target: String,
    size_mb: u64,
    revision: Option<String>,
    description: Option<String>,
    bleu_score: Option<f32>,
    has_safetensors: Option<bool>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelFamily {
    Madlad,
    Marian,
}

impl ModelFamily {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Madlad => "madlad",
            Self::Marian => "marian",
        }
    }
}

#[derive(Debug, Clone)]
pub struct CatalogModel {
    pub name: String,
    pub model_id: String,
    pub family: ModelFamily,
    pub source: Option<String>,
    pub target: Option<String>,
    pub size_mb: u64,
    pub revision: Option<String>,
    pub description: Option<String>,
    pub bleu_score: Option<f32>,
    pub has_safetensors: bool,
}

#[derive(Debug, Clone)]
pub struct CatalogModelStatus {
    pub model: CatalogModel,
    pub cached: bool,
}

pub fn load_model_catalog() -> Result<Vec<CatalogModel>> {
    let parsed: ModelsToml = toml::from_str(MODELS_TOML)
        .map_err(|e| AudioError::ModelCatalogLoad(format!("Failed to parse models.toml: {}", e)))?;

    let mut models = Vec::new();

    for madlad in parsed.madlad {
        models.push(CatalogModel {
            name: madlad.name,
            model_id: madlad.model_id,
            family: ModelFamily::Madlad,
            source: None,
            target: None,
            size_mb: madlad.size_mb,
            revision: madlad.revision,
            description: madlad.description,
            bleu_score: None,
            has_safetensors: madlad.has_safetensors.unwrap_or(true),
        });
    }

    for marian in parsed.marian {
        models.push(CatalogModel {
            name: marian.name,
            model_id: marian.model_id,
            family: ModelFamily::Marian,
            source: Some(marian.source),
            target: Some(marian.target),
            size_mb: marian.size_mb,
            revision: marian.revision,
            description: marian.description,
            bleu_score: marian.bleu_score,
            has_safetensors: marian.has_safetensors.unwrap_or(false),
        });
    }

    Ok(models)
}
