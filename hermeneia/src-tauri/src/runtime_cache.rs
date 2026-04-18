use crate::error::{AudioError, Result};
use crate::runtime_pool::RuntimePool;
use crate::speaker::{SpeakerDevice, SpeakerModel};
use crate::transcribe::model::{get_device, ModelManager};
use crate::transcribe::WhisperModel;
use candle_core::Device;
use candle_transformers::models::whisper::{self as m, Config};
use once_cell::sync::Lazy;
use serde::Serialize;
use sherpa_rs::diarize::Diarize;
use std::collections::HashMap;
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct WhisperRuntimeKey {
    pub model: WhisperModel,
    pub force_cpu: bool,
    pub use_quantized: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
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

impl WhisperRuntime {
    pub fn clone_whisper_runtime(&self) -> Result<Self> {
        Ok(Self {
            config: self.config.clone(),
            tokenizer: self.tokenizer.clone(),
            model: self.model.clone(),
            device: self.device.clone(),
        })
    }

    pub fn reset_kv_cache(&mut self) {
        self.model.reset_kv_cache();
    }
}

pub struct SpeakerRuntime {
    pub diarize: Diarize,
    pub provider: String,
    pub warmed_up: bool,
}

struct WhisperCacheEntry {
    key: WhisperRuntimeKey,
    base_runtime: Arc<WhisperRuntime>,
    pool: Arc<RuntimePool<WhisperRuntime>>,
    loaded_at: Instant,
}

struct SpeakerCacheEntry {
    pool: Arc<RuntimePool<SpeakerRuntime>>,
    loaded_at: Instant,
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
    whisper_entries: Mutex<HashMap<WhisperRuntimeKey, WhisperCacheEntry>>,
    speaker_entries: Mutex<HashMap<SpeakerRuntimeKey, SpeakerCacheEntry>>,
    policy: CachePolicy,
    whisper_pool_limit: usize,
    speaker_pool_limit: usize,
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
            whisper_entries: Mutex::new(HashMap::new()),
            speaker_entries: Mutex::new(HashMap::new()),
            policy,
            whisper_pool_limit: 2,
            speaker_pool_limit: 2,
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
        let (pool, base_runtime) = {
            let mut entries = self
                .whisper_entries
                .lock()
                .map_err(|e| AudioError::ModelLoad {
                    model: key.model.model_id().to_string(),
                    details: format!("Whisper cache lock failed: {}", e),
                })?;

            if let Some(entry) = entries.get(&key) {
                tracing::info!(model = %key.model.model_id(), "Whisper runtime cache hit");
                (Arc::clone(&entry.pool), Arc::clone(&entry.base_runtime))
            } else {
                self.ensure_whisper_capacity(&key)?;

                let started = Instant::now();
                let runtime = load()?;
                let load_ms = started.elapsed().as_millis();

                let base_runtime = Arc::new(runtime);
                let pool = Arc::new(RuntimePool::new(self.whisper_pool_limit));

                entries.insert(
                    key,
                    WhisperCacheEntry {
                        key,
                        base_runtime: Arc::clone(&base_runtime),
                        pool: Arc::clone(&pool),
                        loaded_at: Instant::now(),
                    },
                );

                tracing::info!(
                    model = %key.model.model_id(),
                    load_ms,
                    "Whisper runtime cache miss: loaded keyed pool"
                );

                (pool, base_runtime)
            }
        };

        let mut lease = pool.checkout(|| {
            let mut worker = base_runtime.clone_whisper_runtime()?;
            worker.reset_kv_cache();
            Ok::<WhisperRuntime, AudioError>(worker)
        })?;

        lease.reset_kv_cache();
        let result = use_runtime(&mut lease);
        lease.reset_kv_cache();

        if let Device::Cuda(_) | Device::Metal(_) = lease.device {
            crate::gpu_cleanup::synchronize_device(&lease.device);
        }

        result
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
        let pool = {
            let mut entries = self.speaker_entries.lock().map_err(|e| {
                AudioError::DiarizationFailed(format!("Speaker cache lock failed: {}", e))
            })?;

            if let Some(entry) = entries.get(&key) {
                tracing::info!(
                    provider = %key.device.provider_string(),
                    "Speaker runtime cache hit"
                );
                Arc::clone(&entry.pool)
            } else {
                self.ensure_speaker_capacity(&key)?;

                let pool = Arc::new(RuntimePool::new(self.speaker_pool_limit));

                entries.insert(
                    key.clone(),
                    SpeakerCacheEntry {
                        pool: Arc::clone(&pool),
                        loaded_at: Instant::now(),
                    },
                );

                tracing::info!(
                    model = %key.model.display_name(),
                    provider = %key.device.provider_string(),
                    "Speaker runtime cache miss: initialized keyed pool"
                );

                Arc::clone(&pool)
            }
        };

        let mut load_opt = Some(load);
        let mut lease = pool.checkout(|| {
            let loader = load_opt.take().ok_or_else(|| {
                AudioError::DiarizationFailed("Speaker loader already consumed".to_string())
            })?;
            loader()
        })?;

        use_runtime(&mut lease)
    }

    pub fn clear_whisper(&self) {
        if let Ok(mut entries) = self.whisper_entries.lock() {
            for entry in entries.values() {
                tracing::info!(model = %entry.key.model.model_id(), "Clearing Whisper runtime pool");
                crate::gpu_cleanup::synchronize_device(&entry.base_runtime.device);
            }
            entries.clear();
        }
    }

    pub fn clear_speaker(&self) {
        if let Ok(mut entries) = self.speaker_entries.lock() {
            if !entries.is_empty() {
                tracing::info!(count = entries.len(), "Clearing speaker runtime pools");
            }
            entries.clear();
        }
    }

    pub fn clear_all(&self) {
        self.clear_whisper();
        self.clear_speaker();
    }

    pub fn stats(&self) -> RuntimeCacheStats {
        let now = Instant::now();

        let whisper = self.whisper_entries.lock().ok().and_then(|entries| {
            entries
                .iter()
                .max_by_key(|(_, entry)| entry.loaded_at)
                .map(|(key, entry)| (*key, entry.loaded_at))
        });

        let speaker = self.speaker_entries.lock().ok().and_then(|entries| {
            entries
                .iter()
                .max_by_key(|(_, entry)| entry.loaded_at)
                .map(|(key, entry)| (key.clone(), entry.loaded_at))
        });

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
            speaker_key: speaker
                .as_ref()
                .map(|(k, _)| format!("{}:{}", k.model.cli_key(), k.device.provider_string())),
            whisper_loaded_for_sec: whisper.map(|(_, t)| now.duration_since(t).as_secs()),
            speaker_loaded_for_sec: speaker.map(|(_, t)| now.duration_since(t).as_secs()),
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

pub fn load_whisper_runtime_by_key(key: WhisperRuntimeKey) -> Result<WhisperRuntime> {
    let model_manager = ModelManager::new()?;
    let model_files = model_manager.ensure_model(key.model, key.use_quantized)?;
    let device = get_device(key.force_cpu)?;

    let config_str =
        std::fs::read_to_string(&model_files.config).map_err(|e| AudioError::ModelLoad {
            model: "config".to_string(),
            details: e.to_string(),
        })?;
    let config: Config = serde_json::from_str(&config_str).map_err(|e| AudioError::ModelLoad {
        model: "config".to_string(),
        details: e.to_string(),
    })?;

    let tokenizer =
        Tokenizer::from_file(&model_files.tokenizer).map_err(|e| AudioError::ModelLoad {
            model: "tokenizer".to_string(),
            details: e.to_string(),
        })?;

    let vb =
        crate::gpu_cleanup::load_safetensors_varbuilder(&model_files.weights, m::DTYPE, &device)
            .map_err(|e| crate::gpu_cleanup::to_model_load_error(e, &device, "weights"))?;

    let model = m::model::Whisper::load(&vb, config.clone())
        .map_err(|e| crate::gpu_cleanup::to_model_init_error(e, &device, "whisper"))?;

    Ok(WhisperRuntime {
        config,
        tokenizer,
        model,
        device,
    })
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
    fn test_keyed_whisper_pool_behavior() {
        let cache = RuntimeCacheManager::new(CachePolicy::default());
        let key_a = WhisperRuntimeKey {
            model: WhisperModel::Tiny,
            force_cpu: true,
            use_quantized: false,
        };
        let key_b = WhisperRuntimeKey {
            model: WhisperModel::Base,
            force_cpu: true,
            use_quantized: false,
        };

        let _ = cache.with_whisper_runtime(
            key_a,
            || {
                Err(AudioError::ModelLoad {
                    model: "test".to_string(),
                    details: "expected in test without model files".to_string(),
                })
            },
            |_runtime| Ok(()),
        );

        let _ = cache.with_whisper_runtime(
            key_b,
            || {
                Err(AudioError::ModelLoad {
                    model: "test".to_string(),
                    details: "expected in test without model files".to_string(),
                })
            },
            |_runtime| Ok(()),
        );

        let stats = cache.stats();
        let _ = stats.whisper_key;
    }

    #[test]
    fn test_speaker_cache_hit_skips_preflight_capacity_check() {
        let cache = RuntimeCacheManager::new(CachePolicy {
            ram_headroom_gb: 1_000_000.0,
            vram_headroom_gb: 1_000_000.0,
        });

        let key = SpeakerRuntimeKey {
            model: SpeakerModel::English,
            device: SpeakerDevice::Cpu,
        };

        {
            let mut entries = cache.speaker_entries.lock().unwrap();
            entries.insert(
                key.clone(),
                SpeakerCacheEntry {
                    pool: Arc::new(RuntimePool::new(1)),
                    loaded_at: Instant::now(),
                },
            );
        }

        let loader_called = AtomicBool::new(false);
        let result = cache.with_speaker_runtime(
            key,
            || {
                loader_called.store(true, Ordering::SeqCst);
                Err(AudioError::DiarizationFailed(
                    "loader invoked on cache hit".to_string(),
                ))
            },
            |_runtime| Ok(()),
        );

        assert!(loader_called.load(Ordering::SeqCst));
        assert!(matches!(
            result,
            Err(AudioError::DiarizationFailed(msg)) if msg.contains("loader invoked on cache hit")
        ));
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
