use crate::error::{AudioError, Result};
use crate::speaker::{SpeakerDevice, SpeakerModel};
use crate::transcribe::WhisperModel;
use candle_core::Device;
use candle_transformers::models::whisper::{self as m, Config};
use once_cell::sync::Lazy;
use serde::Serialize;
use sherpa_rs::diarize::Diarize;
use std::sync::{Arc, Mutex};
use std::time::Instant;
use sysinfo::System;
use tokenizers::Tokenizer;

#[derive(Debug, Clone)]
pub struct CachePolicy {
    pub ram_headroom_gb: f32,
    pub vram_headroom_gb: f32,
}

impl Default for CachePolicy {
    fn default() -> Self {
        Self {
            ram_headroom_gb: 1.5,
            vram_headroom_gb: 0.8,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WhisperRuntimeKey {
    pub model: WhisperModel,
    pub force_cpu: bool,
    pub use_quantized: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SpeakerRuntimeKey {
    pub model: SpeakerModel,
    pub device: SpeakerDevice,
}

pub struct WhisperRuntime {
    pub config: Config,
    pub tokenizer: Tokenizer,
    pub model: m::model::Whisper,
    pub device: Device,
}

pub struct SpeakerRuntime {
    pub diarize: Diarize,
    pub provider: String,
    pub warmed_up: bool,
}

struct WhisperCacheEntry {
    key: WhisperRuntimeKey,
    runtime: WhisperRuntime,
    loaded_at: Instant,
    last_used: Instant,
}

struct SpeakerCacheEntry {
    key: SpeakerRuntimeKey,
    runtime: SpeakerRuntime,
    loaded_at: Instant,
    last_used: Instant,
}

#[derive(Debug, Clone, Serialize)]
pub struct RuntimeCacheStats {
    pub whisper_loaded: bool,
    pub speaker_loaded: bool,
    pub whisper_key: Option<String>,
    pub speaker_key: Option<String>,
    pub whisper_loaded_for_sec: Option<u64>,
    pub speaker_loaded_for_sec: Option<u64>,
}

pub struct RuntimeCacheManager {
    whisper_slot: Mutex<Option<WhisperCacheEntry>>,
    speaker_slot: Mutex<Option<SpeakerCacheEntry>>,
    policy: CachePolicy,
}

static GLOBAL_RUNTIME_CACHE: Lazy<Arc<RuntimeCacheManager>> =
    Lazy::new(|| Arc::new(RuntimeCacheManager::default()));

pub fn global_runtime_cache() -> Arc<RuntimeCacheManager> {
    Arc::clone(&GLOBAL_RUNTIME_CACHE)
}

impl Default for RuntimeCacheManager {
    fn default() -> Self {
        Self::new(CachePolicy::default())
    }
}

impl RuntimeCacheManager {
    pub fn new(policy: CachePolicy) -> Self {
        Self {
            whisper_slot: Mutex::new(None),
            speaker_slot: Mutex::new(None),
            policy,
        }
    }

    pub fn with_whisper_runtime<R, L, U>(
        &self,
        key: WhisperRuntimeKey,
        load: L,
        use_runtime: U,
    ) -> Result<R>
    where
        L: FnOnce() -> Result<WhisperRuntime>,
        U: FnOnce(&mut WhisperRuntime) -> Result<R>,
    {
        let mut slot = self
            .whisper_slot
            .lock()
            .map_err(|e| AudioError::ModelLoad {
                model: key.model.model_id().to_string(),
                details: format!("Whisper cache lock failed: {}", e),
            })?;

        if let Some(entry) = slot.as_mut() {
            if entry.key == key {
                entry.last_used = Instant::now();
                tracing::info!(model = %key.model.model_id(), "Whisper runtime cache hit");
                return use_runtime(&mut entry.runtime);
            }

            tracing::info!(
                old_model = %entry.key.model.model_id(),
                new_model = %key.model.model_id(),
                "Evicting Whisper runtime due to key mismatch"
            );
            crate::gpu_cleanup::synchronize_device(&entry.runtime.device);
            *slot = None;
        }

        self.ensure_whisper_capacity(&key)?;

        let started = Instant::now();
        let runtime = load()?;
        let load_ms = started.elapsed().as_millis();
        let now = Instant::now();

        *slot = Some(WhisperCacheEntry {
            key,
            runtime,
            loaded_at: now,
            last_used: now,
        });

        tracing::info!(
            model = %key.model.model_id(),
            load_ms,
            "Whisper runtime cache miss: loaded and cached"
        );

        let entry = slot.as_mut().expect("Whisper cache entry inserted");
        use_runtime(&mut entry.runtime)
    }

    pub fn with_speaker_runtime<R, L, U>(
        &self,
        key: SpeakerRuntimeKey,
        load: L,
        use_runtime: U,
    ) -> Result<R>
    where
        L: FnOnce() -> Result<SpeakerRuntime>,
        U: FnOnce(&mut SpeakerRuntime) -> Result<R>,
    {
        let mut slot = self.speaker_slot.lock().map_err(|e| {
            AudioError::DiarizationFailed(format!("Speaker cache lock failed: {}", e))
        })?;

        if let Some(entry) = slot.as_mut() {
            if entry.key == key {
                entry.last_used = Instant::now();
                tracing::info!(provider = %entry.runtime.provider, "Speaker runtime cache hit");
                return use_runtime(&mut entry.runtime);
            }

            tracing::info!(
                old_model = %entry.key.model.display_name(),
                new_model = %key.model.display_name(),
                old_device = %entry.key.device.provider_string(),
                new_device = %key.device.provider_string(),
                "Evicting speaker runtime due to key mismatch"
            );
            *slot = None;
        }

        self.ensure_speaker_capacity(&key)?;

        let started = Instant::now();
        let runtime = load()?;
        let load_ms = started.elapsed().as_millis();
        let now = Instant::now();

        *slot = Some(SpeakerCacheEntry {
            key: key.clone(),
            runtime,
            loaded_at: now,
            last_used: now,
        });

        tracing::info!(
            model = %key.model.display_name(),
            provider = %key.device.provider_string(),
            load_ms,
            "Speaker runtime cache miss: loaded and cached"
        );

        let entry = slot.as_mut().expect("Speaker cache entry inserted");
        use_runtime(&mut entry.runtime)
    }

    pub fn clear_whisper(&self) {
        if let Ok(mut slot) = self.whisper_slot.lock() {
            if let Some(entry) = slot.as_ref() {
                tracing::info!("Clearing Whisper runtime cache");
                crate::gpu_cleanup::synchronize_device(&entry.runtime.device);
            }
            *slot = None;
        }
    }

    pub fn clear_speaker(&self) {
        if let Ok(mut slot) = self.speaker_slot.lock() {
            if slot.is_some() {
                tracing::info!("Clearing speaker runtime cache");
            }
            *slot = None;
        }
    }

    pub fn clear_all(&self) {
        self.clear_whisper();
        self.clear_speaker();
    }

    pub fn stats(&self) -> RuntimeCacheStats {
        let now = Instant::now();

        let whisper = self
            .whisper_slot
            .lock()
            .ok()
            .and_then(|slot| slot.as_ref().map(|e| (e.key, e.loaded_at)));

        let speaker = self
            .speaker_slot
            .lock()
            .ok()
            .and_then(|slot| slot.as_ref().map(|e| (e.key.clone(), e.loaded_at)));

        let speaker_key = speaker
            .as_ref()
            .map(|(k, _)| format!("{}:{}", k.model.cli_key(), k.device.provider_string()));
        let speaker_loaded_for_sec = speaker
            .as_ref()
            .map(|(_, t)| now.duration_since(*t).as_secs());

        RuntimeCacheStats {
            whisper_loaded: whisper.is_some(),
            speaker_loaded: speaker.is_some(),
            whisper_key: whisper.map(|(k, _)| {
                format!(
                    "{}:{}:{}",
                    k.model.model_id(),
                    if k.force_cpu { "cpu" } else { "auto" },
                    if k.use_quantized { "q" } else { "fp" }
                )
            }),
            speaker_key,
            whisper_loaded_for_sec: whisper.map(|(_, t)| now.duration_since(t).as_secs()),
            speaker_loaded_for_sec,
        }
    }

    fn ensure_whisper_capacity(&self, key: &WhisperRuntimeKey) -> Result<()> {
        let reqs = key.model.requirements();
        let required_ram = reqs.min_ram_gb + self.policy.ram_headroom_gb;
        let available_ram = live_available_ram_gb();

        if available_ram < required_ram {
            return Err(AudioError::OutOfMemory {
                message: format!(
                    "Insufficient RAM to load Whisper model '{}'. Need {:.1}GB available, have {:.1}GB.",
                    key.model.model_id(),
                    required_ram,
                    available_ram
                ),
                device: "RAM".to_string(),
                required_gb: required_ram,
                model_name: key.model.model_id().to_string(),
            });
        }

        if key.force_cpu {
            return Ok(());
        }

        let required_vram = reqs.min_vram_gb + self.policy.vram_headroom_gb;
        if let Some(available_vram) = live_available_vram_gb() {
            if available_vram < required_vram {
                return Err(AudioError::OutOfMemory {
                    message: format!(
                        "Insufficient VRAM to load Whisper model '{}'. Need {:.1}GB available, have {:.1}GB.",
                        key.model.model_id(),
                        required_vram,
                        available_vram
                    ),
                    device: "VRAM".to_string(),
                    required_gb: required_vram,
                    model_name: key.model.model_id().to_string(),
                });
            }
        } else {
            tracing::debug!("Skipping Whisper VRAM preflight (NVML unavailable)");
        }

        Ok(())
    }

    fn ensure_speaker_capacity(&self, key: &SpeakerRuntimeKey) -> Result<()> {
        let (min_ram_gb, min_vram_gb) = speaker_runtime_requirements(&key.model, &key.device);

        let required_ram = min_ram_gb + self.policy.ram_headroom_gb;
        let available_ram = live_available_ram_gb();
        if available_ram < required_ram {
            return Err(AudioError::OutOfMemory {
                message: format!(
                    "Insufficient RAM to initialize speaker runtime '{}'. Need {:.1}GB available, have {:.1}GB.",
                    key.model.display_name(),
                    required_ram,
                    available_ram
                ),
                device: "RAM".to_string(),
                required_gb: required_ram,
                model_name: key.model.display_name().to_string(),
            });
        }

        if !matches!(key.device, SpeakerDevice::Cuda) {
            return Ok(());
        }

        let required_vram = min_vram_gb + self.policy.vram_headroom_gb;
        if let Some(available_vram) = live_available_vram_gb() {
            if available_vram < required_vram {
                return Err(AudioError::OutOfMemory {
                    message: format!(
                        "Insufficient VRAM to initialize speaker runtime '{}'. Need {:.1}GB available, have {:.1}GB.",
                        key.model.display_name(),
                        required_vram,
                        available_vram
                    ),
                    device: "VRAM".to_string(),
                    required_gb: required_vram,
                    model_name: key.model.display_name().to_string(),
                });
            }
        } else {
            tracing::debug!("Skipping speaker VRAM preflight (NVML unavailable)");
        }

        Ok(())
    }
}

fn live_available_ram_gb() -> f32 {
    let mut sys = System::new_all();
    sys.refresh_memory();
    sys.available_memory() as f32 / 1_073_741_824.0
}

#[cfg(feature = "cuda")]
fn live_available_vram_gb() -> Option<f32> {
    use nvml_wrapper::Nvml;

    let nvml = Nvml::init().ok()?;
    let device = nvml.device_by_index(0).ok()?;
    let mem = device.memory_info().ok()?;
    Some(mem.free as f32 / 1_073_741_824.0)
}

#[cfg(not(feature = "cuda"))]
fn live_available_vram_gb() -> Option<f32> {
    None
}

fn speaker_runtime_requirements(model: &SpeakerModel, device: &SpeakerDevice) -> (f32, f32) {
    let base_ram = match model {
        SpeakerModel::English => 0.8,
        SpeakerModel::Multilingual => 1.0,
    };

    let base_vram = match model {
        SpeakerModel::English => 0.7,
        SpeakerModel::Multilingual => 1.0,
    };

    match device {
        SpeakerDevice::Cpu => (base_ram, 0.0),
        SpeakerDevice::Cuda => (base_ram, base_vram),
        SpeakerDevice::CoreMl => (base_ram + 0.3, 0.0),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicBool, Ordering};

    #[test]
    fn test_default_policy_values() {
        let policy = CachePolicy::default();
        assert_eq!(policy.ram_headroom_gb, 1.5);
        assert_eq!(policy.vram_headroom_gb, 0.8);
    }

    #[test]
    fn test_global_runtime_cache_singleton() {
        let a = global_runtime_cache();
        let b = global_runtime_cache();
        assert!(Arc::ptr_eq(&a, &b));
    }

    #[test]
    fn test_stats_empty_on_new_cache() {
        let cache = RuntimeCacheManager::new(CachePolicy::default());
        let stats = cache.stats();

        assert!(!stats.whisper_loaded);
        assert!(!stats.speaker_loaded);
        assert!(stats.whisper_key.is_none());
        assert!(stats.speaker_key.is_none());
    }

    #[test]
    fn test_clear_all_on_empty_cache_is_noop() {
        let cache = RuntimeCacheManager::new(CachePolicy::default());
        cache.clear_all();

        let stats = cache.stats();
        assert!(!stats.whisper_loaded);
        assert!(!stats.speaker_loaded);
    }

    #[test]
    fn test_whisper_preflight_oom_short_circuits_loader() {
        let cache = RuntimeCacheManager::new(CachePolicy {
            ram_headroom_gb: 1_000_000.0,
            vram_headroom_gb: 1_000_000.0,
        });

        let loader_called = AtomicBool::new(false);
        let key = WhisperRuntimeKey {
            model: WhisperModel::Tiny,
            force_cpu: true,
            use_quantized: false,
        };

        let result = cache.with_whisper_runtime(
            key,
            || {
                loader_called.store(true, Ordering::SeqCst);
                Err(AudioError::ModelLoad {
                    model: "test".to_string(),
                    details: "loader should not be called".to_string(),
                })
            },
            |_runtime| Ok(()),
        );

        assert!(matches!(result, Err(AudioError::OutOfMemory { .. })));
        assert!(!loader_called.load(Ordering::SeqCst));
    }

    #[test]
    fn test_speaker_preflight_oom_short_circuits_loader() {
        let cache = RuntimeCacheManager::new(CachePolicy {
            ram_headroom_gb: 1_000_000.0,
            vram_headroom_gb: 1_000_000.0,
        });

        let loader_called = AtomicBool::new(false);
        let key = SpeakerRuntimeKey {
            model: SpeakerModel::English,
            device: SpeakerDevice::Cpu,
        };

        let result = cache.with_speaker_runtime(
            key,
            || {
                loader_called.store(true, Ordering::SeqCst);
                Err(AudioError::DiarizationFailed(
                    "loader should not be called".to_string(),
                ))
            },
            |_runtime| Ok(()),
        );

        assert!(matches!(result, Err(AudioError::OutOfMemory { .. })));
        assert!(!loader_called.load(Ordering::SeqCst));
    }

    #[test]
    fn test_speaker_runtime_requirements_by_device() {
        let (cpu_ram, cpu_vram) =
            speaker_runtime_requirements(&SpeakerModel::English, &SpeakerDevice::Cpu);
        let (cuda_ram, cuda_vram) =
            speaker_runtime_requirements(&SpeakerModel::English, &SpeakerDevice::Cuda);
        let (coreml_ram, coreml_vram) =
            speaker_runtime_requirements(&SpeakerModel::English, &SpeakerDevice::CoreMl);

        assert!(cpu_ram > 0.0);
        assert_eq!(cpu_vram, 0.0);
        assert!(cuda_ram >= cpu_ram);
        assert!(cuda_vram > 0.0);
        assert!(coreml_ram > cpu_ram);
        assert_eq!(coreml_vram, 0.0);
    }
}
