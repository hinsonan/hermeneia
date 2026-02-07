//! System capability detection for model selection validation.
//!
//! This module detects available system resources (RAM, GPU, VRAM) to help
//! users select appropriate Whisper models. Uses caching to avoid repeated
//! system queries.

use once_cell::sync::Lazy;
use serde::Serialize;
use std::sync::Mutex;
use sysinfo::System;

/// Information about detected GPU capabilities
#[derive(Debug, Clone, Serialize)]
pub struct GpuInfo {
    /// Type of GPU detected
    pub device_type: GpuDeviceType,
    /// Total VRAM in GB (may be None if unavailable)
    pub vram_total_gb: Option<f32>,
    /// Available VRAM in GB (may be None if unavailable)
    pub vram_available_gb: Option<f32>,
    /// CUDA compute capability (major, minor) - None for non-NVIDIA
    pub compute_capability: Option<(u32, u32)>,
}

/// GPU device types
#[derive(Debug, Clone, Serialize)]
pub enum GpuDeviceType {
    NvidiaCuda,
    AmdRocm,
    AppleMetal,
    None,
}

/// Complete system capability information
#[derive(Debug, Clone, Serialize)]
pub struct SystemCapabilities {
    /// Total system RAM in GB
    pub total_ram_gb: f32,
    /// Currently available RAM in GB
    pub available_ram_gb: f32,
    /// GPU information if available
    pub gpu_info: Option<GpuInfo>,
}

/// Cached system capabilities (populated on first access)
static CAPABILITIES_CACHE: Lazy<Mutex<Option<SystemCapabilities>>> = Lazy::new(|| Mutex::new(None));

/// Get system capabilities (cached on first call)
///
/// This function detects RAM and GPU resources available on the system.
/// Results are cached to avoid repeated system queries.
///
/// # Errors
/// Returns error if system detection fails
pub fn get_system_capabilities() -> Result<SystemCapabilities, String> {
    let mut cache = CAPABILITIES_CACHE
        .lock()
        .map_err(|e| format!("Failed to lock cache: {}", e))?;

    if let Some(ref caps) = *cache {
        return Ok(caps.clone());
    }

    // Detect capabilities
    let (total_ram_gb, available_ram_gb) = detect_ram()?;
    let gpu_info = detect_gpu()?;

    let caps = SystemCapabilities {
        total_ram_gb,
        available_ram_gb,
        gpu_info,
    };

    *cache = Some(caps.clone());
    Ok(caps)
}

/// Detect system RAM
///
/// Returns (total_ram_gb, available_ram_gb)
fn detect_ram() -> Result<(f32, f32), String> {
    let mut sys = System::new_all();
    sys.refresh_memory();

    let total_bytes = sys.total_memory();
    let available_bytes = sys.available_memory();

    // Convert bytes to GB
    let total_gb = total_bytes as f32 / 1_073_741_824.0; // 1024^3
    let available_gb = available_bytes as f32 / 1_073_741_824.0;

    Ok((total_gb, available_gb))
}

/// Detect GPU capabilities
///
/// Platform-specific detection:
/// - Windows/Linux with CUDA feature: Use NVML for NVIDIA GPUs
/// - macOS: Check for Metal support (no VRAM info available)
/// - Fallback: Check if Candle detects CUDA
fn detect_gpu() -> Result<Option<GpuInfo>, String> {
    // Try NVML first (most accurate for NVIDIA)
    #[cfg(feature = "cuda")]
    {
        if let Ok(gpu_info) = detect_cuda_gpu() {
            return Ok(Some(gpu_info));
        }
    }

    // Fallback: Check if Candle can use CUDA
    #[cfg(feature = "cuda")]
    {
        if candle_core::utils::cuda_is_available() {
            // CUDA is available but we couldn't get detailed info via NVML
            return Ok(Some(GpuInfo {
                device_type: GpuDeviceType::NvidiaCuda,
                vram_total_gb: None,
                vram_available_gb: None,
                compute_capability: None,
            }));
        }
    }

    // Check for Metal (macOS with metal feature)
    #[cfg(all(target_os = "macos", feature = "metal"))]
    {
        if candle_core::utils::metal_is_available() {
            return Ok(Some(GpuInfo {
                device_type: GpuDeviceType::AppleMetal,
                vram_total_gb: None, // Unified memory, validated via system RAM
                vram_available_gb: None,
                compute_capability: None,
            }));
        }
    }

    // No GPU detected
    Ok(None)
}

/// Detect CUDA GPU using NVML (NVIDIA Management Library)
///
/// This provides the most accurate information about NVIDIA GPUs including
/// VRAM, compute capability, and utilization.
#[cfg(feature = "cuda")]
fn detect_cuda_gpu() -> Result<GpuInfo, String> {
    use nvml_wrapper::Nvml;

    let nvml = Nvml::init().map_err(|e| {
        format!("Failed to initialize NVML: {}", e)
    })?;

    // Get first GPU (device 0)
    let device = nvml.device_by_index(0).map_err(|e| {
        format!("Failed to get GPU device: {}", e)
    })?;

    // Get VRAM info
    let memory_info = device.memory_info().map_err(|e| {
        format!("Failed to get memory info: {}", e)
    })?;

    let vram_total_gb = memory_info.total as f32 / 1_073_741_824.0;
    let vram_available_gb = memory_info.free as f32 / 1_073_741_824.0;

    // Get compute capability
    let compute_capability = device.cuda_compute_capability().ok().map(|cc| {
        (cc.major as u32, cc.minor as u32)
    });

    Ok(GpuInfo {
        device_type: GpuDeviceType::NvidiaCuda,
        vram_total_gb: Some(vram_total_gb),
        vram_available_gb: Some(vram_available_gb),
        compute_capability,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detect_ram() {
        let (total, available) = detect_ram().unwrap();
        assert!(total > 0.0, "Total RAM should be > 0");
        assert!(available > 0.0, "Available RAM should be > 0");
        assert!(available <= total, "Available RAM should be <= total");
    }

    #[test]
    fn test_get_capabilities() {
        let caps = get_system_capabilities().unwrap();
        assert!(caps.total_ram_gb > 0.0);
        assert!(caps.available_ram_gb > 0.0);
        // GPU may or may not be present, so we don't assert on it
    }

    #[test]
    fn test_capabilities_cached() {
        let caps1 = get_system_capabilities().unwrap();
        let caps2 = get_system_capabilities().unwrap();

        // Should be identical (cached)
        assert_eq!(caps1.total_ram_gb, caps2.total_ram_gb);
    }
}
