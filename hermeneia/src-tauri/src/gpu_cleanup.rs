use crate::error::AudioError;
use candle_core::Device;
use candle_nn::VarBuilder;
use std::path::Path;

/// Synchronize the GPU device to ensure all pending operations complete
/// before model tensors are dropped.
///
/// - CUDA: Completes all async kernel launches and memory copies
/// - Metal: Finishes all command buffers
/// - CPU: No-op
pub fn synchronize_device(device: &Device) {
    match device.synchronize() {
        Ok(()) => {
            if !device.is_cpu() {
                tracing::debug!("GPU device synchronized before model cleanup");
            }
        }
        Err(e) => {
            tracing::warn!("Failed to synchronize device during cleanup: {}", e);
        }
    }
}

/// Check if an error string indicates an out-of-memory condition.
pub fn is_oom_error(err_str: &str) -> bool {
    err_str.contains("out of memory")
        || err_str.contains("OutOfMemory")
        || err_str.contains("OOM")
        || err_str.contains("CUDA_ERROR_OUT_OF_MEMORY")
        || err_str.contains("failed to allocate")
        || err_str.contains("Cannot allocate memory")
}

/// Convert a candle error to AudioError, detecting OOM conditions.
pub fn to_model_load_error(e: candle_core::Error, device: &Device, model_name: &str) -> AudioError {
    let err_str = e.to_string();
    if is_oom_error(&err_str) {
        let device_label = device_memory_label(device);
        AudioError::OutOfMemory {
            message: format!(
                "Failed to load model into {}. The model is too large for your system.",
                device_label
            ),
            device: device_label.to_string(),
            required_gb: 0.0,
            model_name: model_name.to_string(),
        }
    } else {
        AudioError::ModelLoad {
            model: model_name.to_string(),
            details: e.to_string(),
        }
    }
}

/// Convert a candle error during model initialization to AudioError with OOM detection.
pub fn to_model_init_error(
    e: candle_core::Error,
    device: &Device,
    model_name: &str,
) -> AudioError {
    let err_str = e.to_string();
    if is_oom_error(&err_str) {
        let device_label = device_memory_label(device);
        AudioError::OutOfMemory {
            message: format!(
                "Failed to initialize model in {}. The model is too large for your system.",
                device_label
            ),
            device: device_label.to_string(),
            required_gb: 0.0,
            model_name: model_name.to_string(),
        }
    } else {
        AudioError::ModelLoad {
            model: model_name.to_string(),
            details: e.to_string(),
        }
    }
}

/// Get the user-facing memory label for a device ("VRAM", "Unified Memory", or "RAM").
pub fn device_memory_label(device: &Device) -> &'static str {
    match device {
        Device::Cuda(_) => "VRAM",
        Device::Metal(_) => "Unified Memory",
        Device::Cpu => "RAM",
    }
}

/// Load safetensors weights with platform-appropriate strategy.
///
/// On Windows, uses buffered (in-memory) loading to avoid mmap file handle leaks.
/// On Linux/macOS, uses memory-mapped loading for efficiency.
pub fn load_safetensors_varbuilder(
    weights_path: &Path,
    dtype: candle_core::DType,
    device: &Device,
) -> std::result::Result<VarBuilder<'static>, candle_core::Error> {
    #[cfg(target_os = "windows")]
    {
        tracing::info!("Using buffered safetensors loading (Windows)");
        let data = std::fs::read(weights_path).map_err(|e| {
            candle_core::Error::Msg(format!("Failed to read weights file: {}", e))
        })?;
        VarBuilder::from_buffered_safetensors(data, dtype, device)
    }
    #[cfg(not(target_os = "windows"))]
    {
        unsafe { VarBuilder::from_mmaped_safetensors(&[weights_path.to_path_buf()], dtype, device) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_oom_error() {
        assert!(is_oom_error("CUDA_ERROR_OUT_OF_MEMORY"));
        assert!(is_oom_error("out of memory"));
        assert!(is_oom_error("failed to allocate 4GB"));
        assert!(is_oom_error("Cannot allocate memory"));
        assert!(!is_oom_error("file not found"));
        assert!(!is_oom_error("invalid config"));
    }

    #[test]
    fn test_device_memory_label() {
        assert_eq!(device_memory_label(&Device::Cpu), "RAM");
    }

    #[test]
    fn test_synchronize_cpu_is_noop() {
        synchronize_device(&Device::Cpu);
    }
}
