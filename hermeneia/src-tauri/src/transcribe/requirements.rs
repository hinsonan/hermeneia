//! Model requirements for Whisper variants
//!
//! Defines memory and compute requirements for each model size to help
//! users select appropriate models for their hardware.

use super::WhisperModel;

/// Memory and compute requirements for a Whisper model
#[derive(Debug, Clone)]
pub struct ModelRequirements {
    /// Minimum VRAM in GB for GPU inference
    pub min_vram_gb: f32,
    /// Minimum RAM in GB for CPU inference
    pub min_ram_gb: f32,
    /// Minimum CUDA compute capability (major, minor) - None if no requirement
    pub min_compute_capability: Option<(u32, u32)>,
    /// Approximate disk space in GB for model files
    pub disk_size_gb: f32,
}

impl WhisperModel {
    /// Get memory and compute requirements for this model
    ///
    /// Requirements are based on typical memory usage during inference:
    /// - VRAM: For GPU inference (model weights + activations + KV cache)
    /// - RAM: For CPU inference (higher due to less efficient memory layout)
    /// - Compute: CUDA 7.0+ recommended for optimal performance
    ///
    /// # Note
    /// These are conservative estimates. Actual requirements may vary based on:
    /// - Audio file length (longer files need more memory for KV cache)
    /// - Batch size and beam search parameters
    /// - Quantization (when supported, reduces requirements by ~50%)
    pub fn requirements(&self) -> ModelRequirements {
        match self {
            // Tiny models: ~39M parameters
            Self::Tiny | Self::TinyEn => ModelRequirements {
                min_vram_gb: 1.0,
                min_ram_gb: 2.0,
                min_compute_capability: Some((7, 0)), // CUDA 7.0+
                disk_size_gb: 0.15,
            },

            // Base models: ~74M parameters
            Self::Base | Self::BaseEn => ModelRequirements {
                min_vram_gb: 1.5,
                min_ram_gb: 3.0,
                min_compute_capability: Some((7, 0)),
                disk_size_gb: 0.29,
            },

            // Small models: ~244M parameters
            Self::Small | Self::SmallEn => ModelRequirements {
                min_vram_gb: 2.0,
                min_ram_gb: 4.0,
                min_compute_capability: Some((7, 0)),
                disk_size_gb: 0.97,
            },

            // Medium models: ~769M parameters
            Self::Medium | Self::MediumEn => ModelRequirements {
                min_vram_gb: 5.0,
                min_ram_gb: 8.0,
                min_compute_capability: Some((7, 0)),
                disk_size_gb: 3.1,
            },

            // Large models: ~1550M parameters
            // Large-v2 and Large-v3 have similar requirements
            Self::Large | Self::LargeV2 | Self::LargeV3 => ModelRequirements {
                min_vram_gb: 10.0,
                min_ram_gb: 16.0,
                min_compute_capability: Some((7, 0)),
                disk_size_gb: 6.2,
            },
        }
    }

    /// Get a human-readable size category
    pub fn size_category(&self) -> &'static str {
        match self {
            Self::Tiny | Self::TinyEn => "tiny",
            Self::Base | Self::BaseEn => "base",
            Self::Small | Self::SmallEn => "small",
            Self::Medium | Self::MediumEn => "medium",
            Self::Large | Self::LargeV2 | Self::LargeV3 => "large",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tiny_requirements() {
        let reqs = WhisperModel::Tiny.requirements();
        assert_eq!(reqs.min_vram_gb, 1.0);
        assert_eq!(reqs.min_ram_gb, 2.0);
        assert_eq!(reqs.min_compute_capability, Some((7, 0)));
    }

    #[test]
    fn test_large_requirements() {
        let reqs = WhisperModel::Large.requirements();
        assert_eq!(reqs.min_vram_gb, 10.0);
        assert_eq!(reqs.min_ram_gb, 16.0);
    }

    #[test]
    fn test_requirements_increase_with_size() {
        let tiny = WhisperModel::Tiny.requirements();
        let small = WhisperModel::Small.requirements();
        let large = WhisperModel::Large.requirements();

        // Requirements should increase with model size
        assert!(tiny.min_vram_gb < small.min_vram_gb);
        assert!(small.min_vram_gb < large.min_vram_gb);
        assert!(tiny.min_ram_gb < small.min_ram_gb);
        assert!(small.min_ram_gb < large.min_ram_gb);
    }

    #[test]
    fn test_size_categories() {
        assert_eq!(WhisperModel::Tiny.size_category(), "tiny");
        assert_eq!(WhisperModel::TinyEn.size_category(), "tiny");
        assert_eq!(WhisperModel::Small.size_category(), "small");
        assert_eq!(WhisperModel::LargeV3.size_category(), "large");
    }
}
