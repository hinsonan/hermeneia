//! Model validation against system capabilities
//!
//! Validates whether a Whisper model can run efficiently on the current system.
//! Provides warnings for suboptimal configurations and errors for impossible ones.

use super::WhisperModel;
use crate::system_info::{get_system_capabilities, SystemCapabilities};

/// Result of model validation
#[derive(Debug, Clone)]
pub enum ValidationResult {
    /// Model can run without issues
    Ok,
    /// Model can run but with warnings (e.g., slow performance)
    Warning(Vec<String>),
    /// Model cannot run (insufficient memory)
    Error(String),
}

/// Validator for checking model compatibility with system capabilities
pub struct ModelValidator {
    capabilities: SystemCapabilities,
}

impl ModelValidator {
    /// Create a new validator by detecting system capabilities
    ///
    /// # Errors
    /// Returns error if system detection fails
    pub fn new() -> Result<Self, String> {
        let capabilities = get_system_capabilities()?;
        Ok(Self { capabilities })
    }

    /// Create a validator with pre-detected capabilities (for testing)
    pub fn with_capabilities(capabilities: SystemCapabilities) -> Self {
        Self { capabilities }
    }

    /// Validate if a model can run on this system
    ///
    /// # Arguments
    /// * `model` - The Whisper model to validate
    /// * `force_cpu` - If true, validates for CPU-only inference
    ///
    /// # Returns
    /// * `Ok` - Model can run without issues
    /// * `Warning` - Model can run but may be slow or suboptimal
    /// * `Error` - Model cannot run (insufficient memory)
    pub fn validate_model(&self, model: WhisperModel, force_cpu: bool) -> ValidationResult {
        let reqs = model.requirements();
        let mut warnings = Vec::new();

        // Determine if GPU is available and should be used
        let use_gpu = !force_cpu && self.capabilities.gpu_info.is_some();

        if use_gpu {
            // GPU path validation
            let gpu = self.capabilities.gpu_info.as_ref().unwrap();

            // Check VRAM availability
            if let Some(vram_available) = gpu.vram_available_gb {
                if vram_available < reqs.min_vram_gb {
                    // Hard error: Not enough VRAM
                    return ValidationResult::Error(format!(
                        "Insufficient VRAM for {} model. Need {:.1}GB, have {:.1}GB available. Try a smaller model or use --cpu flag.",
                        model.size_category(),
                        reqs.min_vram_gb,
                        vram_available
                    ));
                } else if vram_available < reqs.min_vram_gb * 1.5 {
                    // Warning: VRAM is tight
                    warnings.push(format!(
                        "VRAM is close to minimum ({:.1}GB available, {:.1}GB required). Consider a smaller model for better performance.",
                        vram_available,
                        reqs.min_vram_gb
                    ));
                }
            } else {
                // VRAM info not available, warn user
                warnings.push(
                    "Could not detect VRAM. Ensure your GPU has sufficient memory.".to_string()
                );
            }

            // Check compute capability
            if let (Some(min_cc), Some(actual_cc)) = (reqs.min_compute_capability, gpu.compute_capability) {
                if actual_cc < min_cc {
                    warnings.push(format!(
                        "GPU compute capability {}.{} is below recommended {}.{}. Performance may be degraded.",
                        actual_cc.0, actual_cc.1, min_cc.0, min_cc.1
                    ));
                }
            }
        } else {
            // CPU path validation
            if self.capabilities.available_ram_gb < reqs.min_ram_gb {
                // Hard error: Not enough RAM
                return ValidationResult::Error(format!(
                    "Insufficient RAM for {} model on CPU. Need {:.1}GB, have {:.1}GB available. Try a smaller model.",
                    model.size_category(),
                    reqs.min_ram_gb,
                    self.capabilities.available_ram_gb
                ));
            } else if self.capabilities.available_ram_gb < reqs.min_ram_gb * 1.5 {
                // Warning: RAM is tight
                warnings.push(format!(
                    "RAM is close to minimum ({:.1}GB available, {:.1}GB required). Close other applications for better performance.",
                    self.capabilities.available_ram_gb,
                    reqs.min_ram_gb
                ));
            }

            // Warn about slow models on CPU (anything larger than base/tiny)
            match model {
                WhisperModel::Large | WhisperModel::LargeV2 | WhisperModel::LargeV3 => {
                    warnings.push("Large model on CPU will be extremely slow (10-100x slower than GPU). Strongly recommend 'tiny' or 'base'.".to_string());
                }
                WhisperModel::Medium | WhisperModel::MediumEn => {
                    warnings.push("Medium model on CPU will be slow. Consider 'tiny' or 'base' for faster results.".to_string());
                }
                WhisperModel::Small | WhisperModel::SmallEn => {
                    warnings.push("Small model on CPU may be slow. Consider 'tiny' or 'base' for better performance.".to_string());
                }
                // Tiny and Base models are fine on CPU
                _ => {}
            }
        }

        if warnings.is_empty() {
            ValidationResult::Ok
        } else {
            ValidationResult::Warning(warnings)
        }
    }

    /// Recommend the best model for this system
    ///
    /// For GPU systems: returns largest model that can run well
    /// For CPU-only systems: returns best model based on available RAM
    pub fn recommend_model(&self) -> WhisperModel {
        let has_gpu = self.capabilities.gpu_info.is_some();

        if has_gpu {
            // GPU systems: try largest models first
            let models = [
                WhisperModel::LargeV3,
                WhisperModel::Medium,
                WhisperModel::Small,
                WhisperModel::Base,
                WhisperModel::Tiny,
            ];

            for model in &models {
                let result = self.validate_model(*model, false);
                match result {
                    ValidationResult::Ok => return *model,
                    ValidationResult::Warning(warnings) => {
                        // Accept warnings that aren't about insufficient memory
                        let has_insufficient_warning = warnings.iter().any(|w| {
                            w.contains("Insufficient") || w.contains("Not enough")
                        });
                        if !has_insufficient_warning {
                            return *model;
                        }
                    }
                    ValidationResult::Error(_) => continue,
                }
            }
        } else {
            // CPU-only systems: be conservative, prioritize speed over quality
            let ram_gb = self.capabilities.available_ram_gb;

            if ram_gb >= 16.0 {
                // Plenty of RAM, can handle medium models but keep it fast
                return WhisperModel::Base; // Good balance, faster than Small on CPU
            } else if ram_gb >= 8.0 {
                // Decent RAM, base model is best
                return WhisperModel::Base;
            } else if ram_gb >= 4.0 {
                // Limited RAM, still prefer base for quality
                return WhisperModel::Base;
            } else if ram_gb >= 3.0 {
                // Very limited RAM, base might still work
                return WhisperModel::Base;
            } else {
                // Minimal RAM, tiny is safest
                return WhisperModel::Tiny;
            }
        }

        WhisperModel::Tiny
    }

    /// Get the system capabilities used by this validator
    pub fn capabilities(&self) -> &SystemCapabilities {
        &self.capabilities
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::system_info::{GpuDeviceType, GpuInfo, SystemCapabilities};

    fn mock_capabilities_cpu_only(ram_gb: f32) -> SystemCapabilities {
        SystemCapabilities {
            total_ram_gb: ram_gb,
            available_ram_gb: ram_gb * 0.8, // Assume 80% available
            gpu_info: None,
        }
    }

    fn mock_capabilities_with_gpu(ram_gb: f32, vram_gb: f32) -> SystemCapabilities {
        SystemCapabilities {
            total_ram_gb: ram_gb,
            available_ram_gb: ram_gb * 0.8,
            gpu_info: Some(GpuInfo {
                device_type: GpuDeviceType::NvidiaCuda,
                vram_total_gb: Some(vram_gb),
                vram_available_gb: Some(vram_gb * 0.9), // Assume 90% available
                compute_capability: Some((8, 6)), // RTX 30xx series
            }),
        }
    }

    #[test]
    fn test_tiny_on_low_ram() {
        let caps = mock_capabilities_cpu_only(4.0);
        let validator = ModelValidator::with_capabilities(caps);
        let result = validator.validate_model(WhisperModel::Tiny, true);
        // Tiny should work on 4GB RAM (needs 2GB)
        assert!(matches!(result, ValidationResult::Ok | ValidationResult::Warning(_)));
    }

    #[test]
    fn test_large_insufficient_ram() {
        let caps = mock_capabilities_cpu_only(8.0);
        let validator = ModelValidator::with_capabilities(caps);
        let result = validator.validate_model(WhisperModel::Large, true);
        // Large needs 16GB RAM, should error on 8GB
        assert!(matches!(result, ValidationResult::Error(_)));
    }

    #[test]
    fn test_large_with_sufficient_vram() {
        let caps = mock_capabilities_with_gpu(16.0, 12.0);
        let validator = ModelValidator::with_capabilities(caps);
        let result = validator.validate_model(WhisperModel::Large, false);
        // Large needs 10GB VRAM, should be OK with 12GB
        assert!(matches!(result, ValidationResult::Ok | ValidationResult::Warning(_)));
    }

    #[test]
    fn test_large_insufficient_vram() {
        let caps = mock_capabilities_with_gpu(32.0, 6.0);
        let validator = ModelValidator::with_capabilities(caps);
        let result = validator.validate_model(WhisperModel::Large, false);
        // Large needs 10GB VRAM, should error on 6GB
        assert!(matches!(result, ValidationResult::Error(_)));
    }

    #[test]
    fn test_recommend_model_high_end() {
        let caps = mock_capabilities_with_gpu(32.0, 24.0);
        let validator = ModelValidator::with_capabilities(caps);
        let recommended = validator.recommend_model();
        // Should recommend large with 24GB VRAM
        assert!(matches!(
            recommended,
            WhisperModel::LargeV3 | WhisperModel::Medium
        ));
    }

    #[test]
    fn test_recommend_model_low_end() {
        let caps = mock_capabilities_cpu_only(4.0);
        let validator = ModelValidator::with_capabilities(caps);
        let recommended = validator.recommend_model();
        // Should recommend tiny or base with 4GB RAM
        assert!(matches!(
            recommended,
            WhisperModel::Tiny | WhisperModel::Base
        ));
    }

    #[test]
    fn test_cpu_warning_when_no_gpu() {
        let caps = mock_capabilities_cpu_only(16.0);
        let validator = ModelValidator::with_capabilities(caps);
        let result = validator.validate_model(WhisperModel::Small, false);
        // Should warn about slow performance on CPU and recommend tiny/base
        if let ValidationResult::Warning(warnings) = result {
            assert!(warnings.iter().any(|w| w.contains("CPU may be slow") && w.contains("'tiny' or 'base'")));
        } else {
            panic!("Expected warning for Small model on CPU");
        }
    }
}
