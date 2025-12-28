/// Automatically detect and apply GPU optimizations
pub fn apply_optimizations() {
    #[cfg(target_os = "linux")]
    {
        linux_nvidia_optimization();
    }

    #[cfg(not(target_os = "linux"))]
    {
        // No optimizations needed on Windows/macOS
    }
}

#[cfg(target_os = "linux")]
fn linux_nvidia_optimization() {
    // Check if environment variables are already set (manual override)
    if std::env::var("__NV_PRIME_RENDER_OFFLOAD").is_ok() {
        eprintln!("🎮 GPU: Manual NVIDIA settings detected");
        return;
    }

    // Detect NVIDIA GPU
    if !detect_nvidia_gpu() {
        return;
    }

    // Detect if this is a hybrid GPU setup
    let is_hybrid = detect_hybrid_gpu();

    if is_hybrid {
        eprintln!("🎮 GPU: Detected NVIDIA hybrid setup - enabling PRIME offload");
        
        std::env::set_var("__NV_PRIME_RENDER_OFFLOAD", "1");
        std::env::set_var("__GLX_VENDOR_LIBRARY_NAME", "nvidia");
        std::env::set_var("WEBKIT_DISABLE_DMABUF_RENDERER", "1");
    } else {
        eprintln!("🎮 GPU: NVIDIA discrete GPU detected");
        // Apply WebKit fix for rendering issues
        std::env::set_var("WEBKIT_DISABLE_DMABUF_RENDERER", "1");
    }
}

#[cfg(target_os = "linux")]
fn detect_nvidia_gpu() -> bool {
    match std::process::Command::new("lspci").output() {
        Ok(output) => {
            if let Ok(stdout) = String::from_utf8(output.stdout) {
                let lower = stdout.to_lowercase();
                return lower.contains("nvidia") 
                    && (lower.contains("vga") || lower.contains("3d"));
            }
            false
        }
        Err(e) => {
            eprintln!("🎮 GPU: Failed to detect GPU: {}", e);
            false
        }
    }
}

#[cfg(target_os = "linux")]
fn detect_hybrid_gpu() -> bool {
    match std::process::Command::new("lspci").output() {
        Ok(output) => {
            if let Ok(stdout) = String::from_utf8(output.stdout) {
                let lower = stdout.to_lowercase();
                
                // Check for integrated GPU alongside discrete
                let has_intel = lower.contains("intel") 
                    && lower.contains("vga");
                let has_amd_integrated = lower.contains("amd") 
                    && lower.contains("vga") 
                    && !lower.contains("radeon rx");
                
                return has_intel || has_amd_integrated;
            }
            false
        }
        Err(_) => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_apply_optimizations_doesnt_panic() {
        // Just verify it doesn't crash
        apply_optimizations();
    }
}