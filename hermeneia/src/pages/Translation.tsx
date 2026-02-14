import { Component, createSignal, For, Show, onCleanup, createEffect, createMemo } from "solid-js";
import { useNavigate } from "@solidjs/router";
import { invoke } from "@tauri-apps/api/core";
import { listen, UnlistenFn } from "@tauri-apps/api/event";
import { save } from "@tauri-apps/plugin-dialog";
import { useTheme } from "../utils/theme";
import TextFileUploader from "../components/TextFileUploader";
import GreekScrollLoader from "../components/GreekScrollLoader";
import TranslationProgressBar from "../components/TranslationProgressBar";
import InfoIcon from "../components/InfoIcon";
import ConfirmDialog from "../components/ConfirmDialog";
import DownloadProgressBar from "../components/DownloadProgressBar";
import type {
  TranslationProgress,
  TextTranslationResult,
} from "../types/translation";
import type { DownloadProgress } from "../types/models";
import { MADLAD_LANGUAGES, getLanguageName } from "../types/translation";
import "./Translation.css";

const Translation: Component = () => {
  const navigate = useNavigate();
  const { toggleTheme } = useTheme();

  // File state
  const [filePath, setFilePath] = createSignal<string>("");
  const [fileName, setFileName] = createSignal<string>("");
  const [isSubtitleFile, setIsSubtitleFile] = createSignal(false);

  // Language settings
  const [sourceLang, setSourceLang] = createSignal<string>("en");
  const [targetLang, setTargetLang] = createSignal<string>("es");

  // Processing state
  const [isTranslating, setIsTranslating] = createSignal(false);
  const [error, setError] = createSignal<string | null>(null);
  const [result, setResult] = createSignal<TextTranslationResult | null>(null);
  const [translationProgress, setTranslationProgress] = createSignal<TranslationProgress | null>(null);

  // Cancel dialog
  const [showCancelDialog, setShowCancelDialog] = createSignal(false);

  // Model download state
  const [isDownloading, setIsDownloading] = createSignal(false);
  const [modelDownloadProgress, setModelDownloadProgress] = createSignal<DownloadProgress | null>(null);

  // Pair validation
  const [marianSupported, setMarianSupported] = createSignal(true);
  const isValidPair = createMemo(() => sourceLang() !== targetLang());

  // Track unlisten functions for cleanup
  let progressUnlisten: UnlistenFn | null = null;
  let downloadUnlisten: UnlistenFn | null = null;

  // Filter target languages - must be different from source
  const availableTargetLanguages = createMemo(() => {
    return MADLAD_LANGUAGES.filter(lang => lang.code !== sourceLang());
  });

  // Filter source languages - must be different from target
  const availableSourceLanguages = createMemo(() => {
    return MADLAD_LANGUAGES.filter(lang => lang.code !== targetLang());
  });

  // Check if the selected pair has high-quality support
  createEffect(async () => {
    const src = sourceLang();
    const tgt = targetLang();
    if (src && tgt && src !== tgt) {
      try {
        const supported = await invoke<boolean>("check_marian_pair_supported", {
          sourceLang: src,
          targetLang: tgt,
        });
        setMarianSupported(supported);
      } catch {
        setMarianSupported(false);
      }
    }
    if (src && tgt && src === tgt) {
      setMarianSupported(false);
    }
  });

  // Auto-select a valid target if source changes to same as target
  createEffect(() => {
    if (sourceLang() === targetLang()) {
      const available = MADLAD_LANGUAGES.filter(l => l.code !== sourceLang());
      if (available.length > 0) {
        setTargetLang(available[0].code);
      }
    }
  });

  // Cleanup on component unmount
  onCleanup(() => {
    if (progressUnlisten) {
      progressUnlisten();
      progressUnlisten = null;
    }
    if (downloadUnlisten) {
      downloadUnlisten();
      downloadUnlisten = null;
    }
  });

  // Handle file selection
  const handleFileSelected = (path: string) => {
    setFilePath(path);
    const name = path.split("/").pop() || path.split("\\").pop() || "Unknown";
    setFileName(name);
    setIsSubtitleFile(name.toLowerCase().endsWith('.srt'));
    setResult(null);
    setError(null);
  };

  // Get quality label based on language pair
  const getQualityLabel = createMemo(() => {
    return marianSupported() ? "High" : "Standard";
  });

  // Pre-download the translation model (MarianMT or MADLAD) with progress UI.
  const ensureTranslationModelDownloaded = async (): Promise<boolean> => {
    // Resolve which model the backend will use for this language pair
    const [modelId, modelName] = await invoke<[string, string]>(
      "resolve_translation_model",
      { sourceLang: sourceLang(), targetLang: targetLang(), allowMadlad: true }
    );

    // Check if it's already cached
    const cached = await invoke<boolean>("is_model_cached", { modelId });
    if (cached) return true;

    setIsDownloading(true);
    setModelDownloadProgress(null);

    // Set up listener BEFORE invoking download
    try {
      downloadUnlisten = await listen<DownloadProgress>("download-progress", (event) => {
        setModelDownloadProgress(event.payload);
        if (event.payload.phase === "complete" || event.payload.phase === "cancelled") {
          setModelDownloadProgress(null);
        }
      });
    } catch (err) {
      console.warn("Failed to set up download listener:", err);
    }

    try {
      await invoke("download_model", { modelId, modelName });
      return true;
    } catch (err) {
      const errStr = String(err);
      if (errStr.includes("cancelled") || errStr.includes("Download cancelled")) return false;
      setError(`Model download failed: ${errStr}`);
      return false;
    } finally {
      setIsDownloading(false);
      setModelDownloadProgress(null);
      if (downloadUnlisten) {
        downloadUnlisten();
        downloadUnlisten = null;
      }
    }
  };

  // Start translation
  const handleTranslate = async () => {
    if (!filePath()) return;
    if (!isValidPair()) {
      setError("Source and target languages must be different.");
      return;
    }

    setError(null);

    // Check if model needs downloading
    const modelReady = await ensureTranslationModelDownloaded();
    if (!modelReady) return;

    setIsTranslating(true);
    setTranslationProgress(null);

    // Set up progress event listener
    try {
      progressUnlisten = await listen<TranslationProgress>('translation-progress', (event) => {
        setTranslationProgress(event.payload);
      });
    } catch (err) {
      console.warn('Failed to set up progress listener:', err);
    }

    try {
      const translationResult = await invoke<TextTranslationResult>("translate_text_file", {
        filePath: filePath(),
        sourceLang: sourceLang(),
        targetLang: targetLang(),
        allowMadlad: true,
      });

      setResult(translationResult);
    } catch (err) {
      const errStr = String(err);
      if (errStr.includes("Operation cancelled")) {
        navigate("/");
        return;
      }
      setError(errStr);
    } finally {
      // Clean up progress listener
      if (progressUnlisten) {
        progressUnlisten();
        progressUnlisten = null;
      }
      setIsTranslating(false);
      setTranslationProgress(null);
    }
  };

  // Copy translated text to clipboard
  const handleCopyTranslation = async () => {
    const res = result();
    if (!res) return;

    try {
      await navigator.clipboard.writeText(res.translated_text);
    } catch (err) {
      console.error("Failed to copy to clipboard:", err);
    }
  };

  // Copy original text to clipboard
  const handleCopyOriginal = async () => {
    const res = result();
    if (!res) return;

    try {
      await navigator.clipboard.writeText(res.original_text);
    } catch (err) {
      console.error("Failed to copy to clipboard:", err);
    }
  };

  // Export translated text
  const handleExportTranslation = async () => {
    const res = result();
    if (!res) return;

    try {
      const extension = res.is_subtitle ? "srt" : "txt";
      const filterName = res.is_subtitle ? "SRT Subtitle Files" : "Text Files";
      const baseName = fileName().replace(/\.[^/.]+$/, "");

      const outputPath = await save({
        filters: [{
          name: filterName,
          extensions: [extension],
        }],
        defaultPath: `${baseName}_${targetLang()}.${extension}`,
      });

      if (!outputPath) return;

      await invoke("write_text_file", {
        path: outputPath,
        content: res.translated_text,
      });
    } catch (err) {
      console.error("Failed to export translation:", err);
    }
  };

  // Handle back button - show confirm dialog if translating
  const handleBack = () => {
    if (isTranslating()) {
      setShowCancelDialog(true);
    } else {
      navigate("/");
    }
  };

  // Confirm cancellation and navigate home immediately
  const handleConfirmCancel = () => {
    setShowCancelDialog(false);
    // Fire-and-forget: set the cancel flag, then navigate immediately
    // The backend spawn_blocking will detect the flag, return Err(Cancelled),
    // and drop all model data (freeing CPU RAM / GPU VRAM automatically)
    invoke("cancel_inference").catch(() => {});
    navigate("/");
  };

  // Reset for new file
  const handleNewFile = () => {
    setFilePath("");
    setFileName("");
    setIsSubtitleFile(false);
    setResult(null);
    setError(null);
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
          <circle cx="12" cy="12" r="5"/>
          <line x1="12" y1="1" x2="12" y2="3"/>
          <line x1="12" y1="21" x2="12" y2="23"/>
          <line x1="4.22" y1="4.22" x2="5.64" y2="5.64"/>
          <line x1="18.36" y1="18.36" x2="19.78" y2="19.78"/>
          <line x1="1" y1="12" x2="3" y2="12"/>
          <line x1="21" y1="12" x2="23" y2="12"/>
          <line x1="4.22" y1="19.78" x2="5.64" y2="18.36"/>
          <line x1="18.36" y1="5.64" x2="19.78" y2="4.22"/>
        </svg>
        <svg class="moon-icon" viewBox="0 0 24 24">
          <path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"/>
        </svg>
      </button>

      <div class="scroll-container">
        <div class="scroll-rod"></div>

        <main class="parchment">
          {/* Header */}
          <header class="page-header">
            <button class="back-button" onClick={handleBack}>
              <svg viewBox="0 0 24 24" width="20" height="20">
                <path d="M19 12H5M12 19l-7-7 7-7"/>
              </svg>
              <span>Home</span>
            </button>
            <h1>Translation</h1>
          </header>

          {/* File Upload Section - shown when no file */}
          <Show when={!filePath()}>
            <section class="upload-section">
              <TextFileUploader onFileSelected={handleFileSelected} />
            </section>
          </Show>

          {/* Main Content - shown after file selected */}
          <Show when={filePath()}>
            {/* File info bar */}
            <div class="file-bar">
              <div class="file-info">
                <svg viewBox="0 0 24 24" width="20" height="20">
                  <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
                  <polyline points="14 2 14 8 20 8" />
                </svg>
                <span class="file-name">{fileName()}</span>
                <Show when={isSubtitleFile()}>
                  <span class="file-badge">SRT</span>
                </Show>
              </div>
              <button class="change-file-btn" onClick={handleNewFile} disabled={isTranslating()}>
                Change
              </button>
            </div>

            {/* Error display */}
            <Show when={error()}>
              <div class="error-banner">
                <div class="error-content">
                  <span>{error()}</span>
                </div>
                <button onClick={() => setError(null)}>Dismiss</button>
              </div>
            </Show>

            {/* Settings panel - hidden during translation, download, and when showing results */}
            <Show when={!isTranslating() && !isDownloading() && !result()}>
              <section class="settings-panel">
                {/* Source Language Selection */}
                <div class="setting-group">
                  <label for="source-lang-select" class="label-with-info">
                    Source Language
                    <InfoIcon
                      content="The language of the original text. Select the language your file is written in."
                      position="right"
                    />
                  </label>
                  <div class="select-wrapper">
                    <select
                      id="source-lang-select"
                      value={sourceLang()}
                      onChange={(e) => setSourceLang(e.currentTarget.value)}
                    >
                      <For each={availableSourceLanguages()}>
                        {(lang) => (
                          <option value={lang.code}>
                            {lang.name}
                          </option>
                        )}
                      </For>
                    </select>
                    <svg class="select-arrow" viewBox="0 0 24 24">
                      <path d="M6 9l6 6 6-6"/>
                    </svg>
                  </div>
                </div>

                {/* Target Language Selection */}
                <div class="setting-group">
                  <label for="target-lang-select" class="label-with-info">
                    Target Language
                    <InfoIcon
                      content="The language you want to translate into."
                      position="right"
                    />
                  </label>
                  <div class="select-wrapper">
                    <select
                      id="target-lang-select"
                      value={targetLang()}
                      onChange={(e) => setTargetLang(e.currentTarget.value)}
                    >
                      <For each={availableTargetLanguages()}>
                        {(lang) => (
                          <option value={lang.code}>
                            {lang.name}
                          </option>
                        )}
                      </For>
                    </select>
                    <svg class="select-arrow" viewBox="0 0 24 24">
                      <path d="M6 9l6 6 6-6"/>
                    </svg>
                  </div>
                </div>

                {/* Model info display */}
                <div class="model-info">
                  <span class="model-label">Quality:</span>
                  <span class="model-name">{getQualityLabel()}</span>
                  <Show when={!marianSupported()}>
                    <span class="model-note">(More languages supported)</span>
                  </Show>
                </div>

                {/* Subtitle preservation notice */}
                <Show when={isSubtitleFile()}>
                  <div class="subtitle-notice">
                    <svg viewBox="0 0 24 24" width="16" height="16">
                      <circle cx="12" cy="12" r="10"/>
                      <line x1="12" y1="16" x2="12" y2="12"/>
                      <line x1="12" y1="8" x2="12.01" y2="8"/>
                    </svg>
                    <span>Timestamps will be preserved in the translated subtitle file.</span>
                  </div>
                </Show>

                {/* Start button */}
                <button
                  class="start-btn"
                  onClick={handleTranslate}
                  disabled={!isValidPair()}
                  title={!isValidPair() ? 'Source and target must be different' : ''}
                >
                  <svg viewBox="0 0 24 24" width="20" height="20">
                    <path d="M12.87 15.07l-2.54-2.51.03-.03A17.52 17.52 0 0 0 14.07 6H17V4h-7V2H8v2H1v2h11.17C11.5 7.92 10.44 9.75 9 11.35 8.07 10.32 7.3 9.19 6.69 8h-2c.73 1.63 1.73 3.17 2.98 4.56l-5.09 5.02L4 19l5-5 3.11 3.11.76-2.04zM18.5 10h-2L12 22h2l1.12-3h4.75L21 22h2l-4.5-12zm-2.62 7l1.62-4.33L19.12 17h-3.24z"/>
                  </svg>
                  Begin Translation
                </button>
              </section>
            </Show>

            {/* Model download progress */}
            <Show when={isDownloading()}>
              <section class="processing-section">
                <h2>Downloading Model...</h2>
                <p>Downloading translation model before starting</p>
                <DownloadProgressBar
                  progress={modelDownloadProgress()}
                  onCancel={() => invoke("cancel_download").catch(() => {})}
                />
              </section>
            </Show>

            {/* Processing state */}
            <Show when={isTranslating()}>
              <section class="processing-section">
                <GreekScrollLoader />
                <h2>
                  {translationProgress()?.phase === 'loading_model'
                    ? 'Loading Model...'
                    : 'Translating...'}
                </h2>
                <p>Translating from {getLanguageName(sourceLang())} to {getLanguageName(targetLang())}</p>

                {/* Progress Bar */}
                <TranslationProgressBar progress={translationProgress()} />

                <div class="processing-details">
                  <span>File: {fileName()}</span>
                  <span>Quality: {getQualityLabel()}</span>
                </div>
              </section>
            </Show>

            {/* Results */}
            <Show when={result()} keyed>
              {(res) => (
                <section class="results-section">
                  {/* Metadata */}
                  <div class="result-meta">
                    <div class="meta-item">
                      <span class="meta-label">From</span>
                      <span class="meta-value">{getLanguageName(res.source_language)}</span>
                    </div>
                    <div class="meta-item">
                      <span class="meta-label">To</span>
                      <span class="meta-value">{getLanguageName(res.target_language)}</span>
                    </div>
                    <div class="meta-item">
                      <span class="meta-label">Processing</span>
                      <span class="meta-value">{res.inference_time.toFixed(1)}s</span>
                    </div>
                    <div class="meta-item">
                      <span class="meta-label">Segments</span>
                      <span class="meta-value">{res.segments_translated}</span>
                    </div>
                  </div>

                  {/* Translation comparison view */}
                  <div class="translation-comparison">
                    {/* Original text */}
                    <div class="text-box original-box">
                      <div class="text-box-header">
                        <h3>Original ({getLanguageName(res.source_language)})</h3>
                        <button class="icon-btn" onClick={handleCopyOriginal} title="Copy original">
                          <svg viewBox="0 0 24 24" width="16" height="16">
                            <rect x="9" y="9" width="13" height="13" rx="2" ry="2"/>
                            <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/>
                          </svg>
                        </button>
                      </div>
                      <div class="text-box-content">
                        {res.original_text}
                      </div>
                    </div>

                    {/* Translated text */}
                    <div class="text-box translated-box">
                      <div class="text-box-header">
                        <h3>Translated ({getLanguageName(res.target_language)})</h3>
                        <button class="icon-btn" onClick={handleCopyTranslation} title="Copy translation">
                          <svg viewBox="0 0 24 24" width="16" height="16">
                            <rect x="9" y="9" width="13" height="13" rx="2" ry="2"/>
                            <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/>
                          </svg>
                        </button>
                      </div>
                      <div class="text-box-content">
                        {res.translated_text}
                      </div>
                    </div>
                  </div>

                  {/* Export actions */}
                  <div class="export-actions">
                    <button class="action-btn primary" onClick={handleExportTranslation}>
                      <svg viewBox="0 0 24 24" width="16" height="16">
                        <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
                        <polyline points="7 10 12 15 17 10"/>
                        <line x1="12" y1="15" x2="12" y2="3"/>
                      </svg>
                      <span>Download {res.is_subtitle ? '.srt' : '.txt'}</span>
                    </button>
                  </div>

                  {/* New translation button */}
                  <button class="new-file-btn" onClick={handleNewFile}>
                    <svg viewBox="0 0 24 24" width="18" height="18">
                      <line x1="12" y1="5" x2="12" y2="19"/>
                      <line x1="5" y1="12" x2="19" y2="12"/>
                    </svg>
                    New Translation
                  </button>
                </section>
              )}
            </Show>
          </Show>
        </main>

        <div class="scroll-rod"></div>
      </div>

      <ConfirmDialog
        open={showCancelDialog()}
        title="Stop Translation?"
        message="A translation is currently in progress. Stopping will discard any partial results."
        confirmLabel="Stop & Go Back"
        cancelLabel="Keep Working"
        onConfirm={handleConfirmCancel}
        onCancel={() => setShowCancelDialog(false)}
      />
    </>
  );
};

export default Translation;
