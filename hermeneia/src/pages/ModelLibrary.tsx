import { Component, createSignal, For, Show, onCleanup, onMount, createMemo } from "solid-js";
import { useNavigate } from "@solidjs/router";
import { invoke } from "@tauri-apps/api/core";
import { listen, UnlistenFn } from "@tauri-apps/api/event";
import { useTheme } from "../utils/theme";
import DownloadProgressBar from "../components/DownloadProgressBar";
import ConfirmDialog from "../components/ConfirmDialog";
import type { ModelInfo, DownloadProgress } from "../types/models";
import "./ModelLibrary.css";

function formatSize(mb: number): string {
  if (mb >= 1024) return `${(mb / 1024).toFixed(1)} GB`;
  return `${mb} MB`;
}

function formatBytes(bytes: number): string {
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(0)} KB`;
  if (bytes < 1024 * 1024 * 1024) return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  return `${(bytes / (1024 * 1024 * 1024)).toFixed(2)} GB`;
}

const ModelLibrary: Component = () => {
  const navigate = useNavigate();
  const { toggleTheme } = useTheme();

  const [models, setModels] = createSignal<ModelInfo[]>([]);
  const [cacheSize, setCacheSize] = createSignal<number>(0);
  const [activeTab, setActiveTab] = createSignal<"transcription" | "translation">("transcription");
  const [downloadProgress, setDownloadProgress] = createSignal<DownloadProgress | null>(null);
  const [downloadingModelId, setDownloadingModelId] = createSignal<string | null>(null);
  const [searchQuery, setSearchQuery] = createSignal("");
  const [deleteTarget, setDeleteTarget] = createSignal<ModelInfo | null>(null);
  const [loading, setLoading] = createSignal(true);

  let progressUnlisten: UnlistenFn | null = null;

  const whisperModels = createMemo(() =>
    models().filter(m => m.category === "whisper")
  );

  const madladModels = createMemo(() =>
    models().filter(m => m.category === "madlad")
  );

  const marianModels = createMemo(() => {
    const query = searchQuery().toLowerCase();
    return models()
      .filter(m => m.category === "marian")
      .filter(m =>
        !query ||
        m.display_name.toLowerCase().includes(query) ||
        m.source_lang?.toLowerCase().includes(query) ||
        m.target_lang?.toLowerCase().includes(query) ||
        m.model_id.toLowerCase().includes(query)
      );
  });

  const cachedCount = createMemo(() => models().filter(m => m.is_cached).length);

  const loadModels = async () => {
    try {
      setLoading(true);
      const [modelList, size] = await Promise.all([
        invoke<ModelInfo[]>("list_models"),
        invoke<number>("get_cache_size"),
      ]);
      setModels(modelList);
      setCacheSize(size);
    } catch (err) {
      console.error("Failed to load models:", err);
    } finally {
      setLoading(false);
    }
  };

  onMount(async () => {
    await loadModels();

    progressUnlisten = await listen<DownloadProgress>("download-progress", (event) => {
      const p = event.payload;
      setDownloadProgress(p);
      if (p.phase === "complete" || p.phase === "cancelled") {
        setDownloadingModelId(null);
        setDownloadProgress(null);
        loadModels();
      }
    });
  });

  onCleanup(() => {
    // Cancel any in-progress download when navigating away
    if (downloadingModelId()) {
      invoke("cancel_download").catch(() => {});
    }
    if (progressUnlisten) {
      progressUnlisten();
      progressUnlisten = null;
    }
  });

  const handleDownload = async (model: ModelInfo) => {
    if (downloadingModelId()) return;
    setDownloadingModelId(model.model_id);
    try {
      await invoke("download_model", {
        modelId: model.model_id,
        modelName: model.display_name,
      });
    } catch (err) {
      const errStr = String(err);
      if (!errStr.includes("cancelled")) {
        console.error("Download failed:", err);
      }
    } finally {
      setDownloadingModelId(null);
      setDownloadProgress(null);
      loadModels();
    }
  };

  const handleCancelDownload = async () => {
    try {
      await invoke("cancel_download");
    } catch (err) {
      console.error("Cancel failed:", err);
    }
  };

  const handleDelete = async () => {
    const target = deleteTarget();
    if (!target) return;
    setDeleteTarget(null);
    try {
      await invoke("delete_model", { modelId: target.model_id });
      loadModels();
    } catch (err) {
      console.error("Delete failed:", err);
    }
  };

  const renderModelCard = (model: ModelInfo) => {
    const isDownloading = () => downloadingModelId() === model.model_id;
    const isAnyDownloading = () => downloadingModelId() !== null;

    return (
      <div class={`model-card ${model.is_cached ? "cached" : ""}`}>
        <div class="model-card-header">
          <span class="model-card-name">{model.display_name}</span>
          <span class={`model-badge ${model.is_cached ? "badge-cached" : "badge-available"}`}>
            {model.is_cached ? "Downloaded" : "Available"}
          </span>
        </div>
        <div class="model-card-meta">
          <span class="model-card-size">{formatSize(model.size_mb)}</span>
          <Show when={model.source_lang && model.target_lang}>
            <span class="model-card-langs">
              {model.source_lang?.toUpperCase()} → {model.target_lang?.toUpperCase()}
            </span>
          </Show>
        </div>

        <Show when={isDownloading() && downloadProgress()}>
          <DownloadProgressBar
            progress={downloadProgress()}
            onCancel={handleCancelDownload}
          />
        </Show>

        <Show when={!isDownloading()}>
          <div class="model-card-actions">
            <Show when={!model.is_cached}>
              <button
                class="model-action-btn download-btn"
                onClick={() => handleDownload(model)}
                disabled={isAnyDownloading()}
              >
                <svg viewBox="0 0 24 24" width="16" height="16">
                  <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
                  <polyline points="7 10 12 15 17 10" />
                  <line x1="12" y1="15" x2="12" y2="3" />
                </svg>
                Download
              </button>
            </Show>
            <Show when={model.is_cached}>
              <button
                class="model-action-btn delete-btn"
                onClick={() => setDeleteTarget(model)}
                disabled={isAnyDownloading()}
              >
                <svg viewBox="0 0 24 24" width="16" height="16">
                  <polyline points="3 6 5 6 21 6" />
                  <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2" />
                </svg>
                Delete
              </button>
            </Show>
          </div>
        </Show>
      </div>
    );
  };

  return (
    <>
      {/* Theme Toggle */}
      <button
        class="theme-toggle"
        onClick={toggleTheme}
        aria-label="Toggle dark mode"
      >
        <svg class="sun-icon" viewBox="0 0 24 24">
          <circle cx="12" cy="12" r="5" />
          <line x1="12" y1="1" x2="12" y2="3" />
          <line x1="12" y1="21" x2="12" y2="23" />
          <line x1="4.22" y1="4.22" x2="5.64" y2="5.64" />
          <line x1="18.36" y1="18.36" x2="19.78" y2="19.78" />
          <line x1="1" y1="12" x2="3" y2="12" />
          <line x1="21" y1="12" x2="23" y2="12" />
          <line x1="4.22" y1="19.78" x2="5.64" y2="18.36" />
          <line x1="18.36" y1="5.64" x2="19.78" y2="4.22" />
        </svg>
        <svg class="moon-icon" viewBox="0 0 24 24">
          <path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z" />
        </svg>
      </button>

      <div class="scroll-container">
        <div class="scroll-rod"></div>

        <main class="parchment">
          {/* Header */}
          <header class="page-header">
            <button class="back-button" onClick={() => navigate("/")}>
              <svg viewBox="0 0 24 24" width="20" height="20">
                <path d="M19 12H5M12 19l-7-7 7-7" />
              </svg>
              <span>Home</span>
            </button>
            <h1>Model Library</h1>
          </header>

          {/* Cache Summary */}
          <div class="cache-summary">
            <div class="cache-stat">
              <span class="cache-label">Total Cache</span>
              <span class="cache-value">{formatBytes(cacheSize())}</span>
            </div>
            <div class="cache-stat">
              <span class="cache-label">Models Cached</span>
              <span class="cache-value">{cachedCount()}</span>
            </div>
          </div>

          {/* Tabs */}
          <div class="model-tabs">
            <button
              class={`model-tab ${activeTab() === "transcription" ? "active" : ""}`}
              onClick={() => setActiveTab("transcription")}
            >
              Transcription
            </button>
            <button
              class={`model-tab ${activeTab() === "translation" ? "active" : ""}`}
              onClick={() => setActiveTab("translation")}
            >
              Translation
            </button>
          </div>

          {/* Loading state */}
          <Show when={loading()}>
            <div class="models-loading">Loading models...</div>
          </Show>

          {/* Transcription Tab */}
          <Show when={!loading() && activeTab() === "transcription"}>
            <section class="model-section">
              <h2 class="section-title">Whisper Models</h2>
              <p class="section-desc">OpenAI Whisper models for speech-to-text transcription.</p>
              <Show when={whisperModels().length > 0} fallback={
                <p class="no-results">No transcription models found. Check your connection and try refreshing.</p>
              }>
                <div class="model-grid">
                  <For each={whisperModels()}>
                    {(model) => renderModelCard(model)}
                  </For>
                </div>
              </Show>
            </section>
          </Show>

          {/* Translation Tab */}
          <Show when={!loading() && activeTab() === "translation"}>
            {/* MADLAD section */}
            <section class="model-section">
              <h2 class="section-title">Multilingual (MADLAD-400)</h2>
              <p class="section-desc">Large multilingual models supporting 450+ languages. Use when no specialized model is available.</p>
              <div class="model-grid">
                <For each={madladModels()}>
                  {(model) => renderModelCard(model)}
                </For>
              </div>
            </section>

            {/* MarianMT section */}
            <section class="model-section">
              <h2 class="section-title">Language Pairs (MarianMT)</h2>
              <p class="section-desc">Specialized models for specific language pairs. Faster and higher quality for supported pairs.</p>

              {/* Search/Filter */}
              <div class="model-search">
                <svg viewBox="0 0 24 24" width="16" height="16">
                  <circle cx="11" cy="11" r="8" />
                  <line x1="21" y1="21" x2="16.65" y2="16.65" />
                </svg>
                <input
                  type="text"
                  placeholder="Filter by language..."
                  value={searchQuery()}
                  onInput={(e) => setSearchQuery(e.currentTarget.value)}
                />
              </div>

              <div class="model-grid compact">
                <For each={marianModels()}>
                  {(model) => renderModelCard(model)}
                </For>
              </div>

              <Show when={marianModels().length === 0 && searchQuery()}>
                <p class="no-results">No models matching "{searchQuery()}"</p>
              </Show>
            </section>
          </Show>
        </main>

        <div class="scroll-rod"></div>
      </div>

      <ConfirmDialog
        open={deleteTarget() !== null}
        title="Delete Model?"
        message={`This will delete ${deleteTarget()?.display_name || "this model"} (${formatSize(deleteTarget()?.size_mb || 0)}) from cache. You can re-download it later.`}
        confirmLabel="Delete"
        cancelLabel="Keep"
        onConfirm={handleDelete}
        onCancel={() => setDeleteTarget(null)}
      />
    </>
  );
};

export default ModelLibrary;
