import { Component, createSignal, For, Show, onCleanup, onMount, createEffect } from "solid-js";
import { useNavigate } from "@solidjs/router";
import { invoke } from "@tauri-apps/api/core";
import { listen, UnlistenFn } from "@tauri-apps/api/event";
import { save } from "@tauri-apps/plugin-dialog";
import { useTheme } from "../utils/theme";
import { formatTime } from "../utils/timeFormat";
import FileUploader from "../components/FileUploader";
import GreekScrollLoader from "../components/GreekScrollLoader";
import TranscriptionProgressBar from "../components/TranscriptionProgressBar";
import InfoIcon from "../components/InfoIcon";
import type {
  WhisperModel,
  TranscriptionTask,
  TranscriptResult,
  TranscriptionProgress,
  ModelOption,
  LanguageOption,
  SystemCapabilities,
  ModelValidation
} from "../types/transcription";
import "./Transcription.css";

const MODEL_OPTIONS: ModelOption[] = [
  { value: 'tiny', label: 'Tiny', description: 'Fastest, least accurate (~1GB VRAM)' },
  { value: 'tiny.en', label: 'Tiny (English)', description: 'English-only, faster' },
  { value: 'base', label: 'Base', description: 'Good balance of speed and accuracy (~1GB VRAM)' },
  { value: 'base.en', label: 'Base (English)', description: 'English-only, recommended' },
  { value: 'small', label: 'Small', description: 'Better accuracy (~2GB VRAM)' },
  { value: 'small.en', label: 'Small (English)', description: 'English-only, better accuracy' },
  { value: 'medium', label: 'Medium', description: 'High accuracy (~5GB VRAM)' },
  { value: 'medium.en', label: 'Medium (English)', description: 'English-only, high accuracy' },
  { value: 'large', label: 'Large', description: 'Highest accuracy (~10GB VRAM)' },
  { value: 'large-v2', label: 'Large v2', description: 'Improved large model' },
  { value: 'large-v3', label: 'Large v3', description: 'Latest large model' },
];

const LANGUAGE_OPTIONS: LanguageOption[] = [
  { value: null, label: 'Auto-detect' },
  { value: 'en', label: 'English' },
  { value: 'es', label: 'Spanish' },
  { value: 'fr', label: 'French' },
  { value: 'de', label: 'German' },
  { value: 'it', label: 'Italian' },
  { value: 'pt', label: 'Portuguese' },
  { value: 'ru', label: 'Russian' },
  { value: 'zh', label: 'Chinese' },
  { value: 'ja', label: 'Japanese' },
  { value: 'ko', label: 'Korean' },
  { value: 'ar', label: 'Arabic' },
  { value: 'el', label: 'Greek' },
  { value: 'he', label: 'Hebrew' },
  { value: 'hi', label: 'Hindi' },
  { value: 'nl', label: 'Dutch' },
  { value: 'pl', label: 'Polish' },
  { value: 'tr', label: 'Turkish' },
  { value: 'vi', label: 'Vietnamese' },
  { value: 'th', label: 'Thai' },
];

const Transcription: Component = () => {
  const navigate = useNavigate();
  const { toggleTheme } = useTheme();

  // File state
  const [filePath, setFilePath] = createSignal<string>("");
  const [fileName, setFileName] = createSignal<string>("");

  // Settings
  const [selectedModel, setSelectedModel] = createSignal<WhisperModel>('tiny');
  const [selectedTask, setSelectedTask] = createSignal<TranscriptionTask>('transcribe');
  const [selectedLanguage, setSelectedLanguage] = createSignal<string | null>(null);
  const [includeTimestamps, setIncludeTimestamps] = createSignal(true);

  // Processing state
  const [isTranscribing, setIsTranscribing] = createSignal(false);
  const [error, setError] = createSignal<string | null>(null);
  const [result, setResult] = createSignal<TranscriptResult | null>(null);
  const [transcriptionProgress, setTranscriptionProgress] = createSignal<TranscriptionProgress | null>(null);

  // System capability detection
  const [systemCapabilities, setSystemCapabilities] = createSignal<SystemCapabilities | null>(null);
  const [modelValidation, setModelValidation] = createSignal<ModelValidation | null>(null);

  // Track unlisten function for cleanup
  let progressUnlisten: UnlistenFn | null = null;

  // Cleanup on component unmount
  onCleanup(() => {
    if (progressUnlisten) {
      progressUnlisten();
      progressUnlisten = null;
    }
  });

  // Fetch system capabilities on mount
  onMount(async () => {
    try {
      const caps = await invoke<SystemCapabilities>("get_system_capabilities");
      setSystemCapabilities(caps);
    } catch (err) {
      console.warn("Failed to get system capabilities:", err);
    }
  });

  // Validate model selection whenever model changes
  createEffect(() => {
    const model = selectedModel();
    const caps = systemCapabilities();

    if (caps) {
      validateModel(model);
    }
  });

  const validateModel = async (model: WhisperModel) => {
    try {
      const validation = await invoke<ModelValidation>(
        "validate_model_selection",
        { model, forceCpu: false }
      );
      setModelValidation(validation);
    } catch (err) {
      console.error("Validation failed:", err);
    }
  };

  // Segments expanded state
  const [segmentsExpanded, setSegmentsExpanded] = createSignal(false);

  // Handle file selection
  const handleFileSelected = (path: string) => {
    setFilePath(path);
    setFileName(path.split("/").pop() || path.split("\\").pop() || "Unknown");
    setResult(null);
    setError(null);
  };

  // Format timestamp for display (MM:SS.ms)
  const formatTimestamp = (seconds: number | null): string => {
    if (seconds === null) return "--:--";
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    const ms = Math.floor((seconds % 1) * 100);
    return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}.${ms.toString().padStart(2, '0')}`;
  };

  // Format timestamp for SRT format (HH:MM:SS,mmm)
  const formatSrtTimestamp = (seconds: number | null): string => {
    if (seconds === null) return "00:00:00,000";
    const hours = Math.floor(seconds / 3600);
    const mins = Math.floor((seconds % 3600) / 60);
    const secs = Math.floor(seconds % 60);
    const ms = Math.floor((seconds % 1) * 1000);
    return `${hours.toString().padStart(2, '0')}:${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')},${ms.toString().padStart(3, '0')}`;
  };

  // Generate plain text content
  const getPlainTextContent = (): string => {
    const transcript = result();
    if (!transcript) return "";
    return transcript.text;
  };

  // Generate SRT content
  const getSrtContent = (): string => {
    const transcript = result();
    if (!transcript || transcript.segments.length === 0) return "";

    return transcript.segments
      .map((seg, index) => {
        const startTime = formatSrtTimestamp(seg.start);
        const endTime = formatSrtTimestamp(seg.end);
        return `${index + 1}\n${startTime} --> ${endTime}\n${seg.text.trim()}\n`;
      })
      .join('\n');
  };

  // Check if error is OOM-related
  const isOOMError = (err: string): boolean => {
    return err.toLowerCase().includes('out of memory') ||
           err.toLowerCase().includes('oom') ||
           err.includes('OutOfMemory');
  };

  // Start transcription
  const handleTranscribe = async () => {
    if (!filePath()) return;

    setIsTranscribing(true);
    setError(null);
    setTranscriptionProgress(null);

    // Set up progress event listener
    try {
      progressUnlisten = await listen<TranscriptionProgress>('transcription-progress', (event) => {
        setTranscriptionProgress(event.payload);
      });
    } catch (err) {
      console.warn('Failed to set up progress listener:', err);
    }

    try {
      const transcriptResult = await invoke<TranscriptResult>("transcribe_audio_file", {
        filePath: filePath(),
        model: selectedModel(),
        task: selectedTask(),
        language: selectedLanguage(),
        timestamps: includeTimestamps(),
      });

      setResult(transcriptResult);
    } catch (err) {
      setError(String(err));
    } finally {
      // Clean up progress listener
      if (progressUnlisten) {
        progressUnlisten();
        progressUnlisten = null;
      }
      setIsTranscribing(false);
      setTranscriptionProgress(null);
    }
  };

  // Copy plain text to clipboard
  const handleCopyPlainText = async () => {
    const text = getPlainTextContent();
    if (!text) return;

    try {
      await navigator.clipboard.writeText(text);
    } catch (err) {
      console.error("Failed to copy to clipboard:", err);
    }
  };

  // Copy SRT to clipboard
  const handleCopySrt = async () => {
    const srt = getSrtContent();
    if (!srt) return;

    try {
      await navigator.clipboard.writeText(srt);
    } catch (err) {
      console.error("Failed to copy SRT to clipboard:", err);
    }
  };

  // Export transcript as plain text file
  const handleExportPlainText = async () => {
    const transcript = result();
    if (!transcript) return;

    try {
      const outputPath = await save({
        filters: [{
          name: "Text Files",
          extensions: ["txt"],
        }],
        defaultPath: `${fileName().replace(/\.[^/.]+$/, "")}_transcript.txt`,
      });

      if (!outputPath) return;

      const content = transcript.text;

      await invoke("write_text_file", {
        path: outputPath,
        content: content,
      });
    } catch (err) {
      console.error("Failed to export transcript:", err);
    }
  };

  // Export transcript as SRT file
  const handleExportSrt = async () => {
    const transcript = result();
    if (!transcript || transcript.segments.length === 0) return;

    try {
      const outputPath = await save({
        filters: [{
          name: "SRT Subtitle Files",
          extensions: ["srt"],
        }],
        defaultPath: `${fileName().replace(/\.[^/.]+$/, "")}.srt`,
      });

      if (!outputPath) return;

      const content = getSrtContent();

      await invoke("write_text_file", {
        path: outputPath,
        content: content,
      });
    } catch (err) {
      console.error("Failed to export SRT:", err);
    }
  };

  // Reset for new file
  const handleNewFile = () => {
    setFilePath("");
    setFileName("");
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
            <button class="back-button" onClick={() => navigate("/")}>
              <svg viewBox="0 0 24 24" width="20" height="20">
                <path d="M19 12H5M12 19l-7-7 7-7"/>
              </svg>
              <span>Home</span>
            </button>
            <h1>Transcription</h1>
          </header>

          {/* File Upload Section - shown when no file */}
          <Show when={!filePath()}>
            <section class="upload-section">
              <FileUploader onFileSelected={handleFileSelected} />
            </section>
          </Show>

          {/* Main Content - shown after file selected */}
          <Show when={filePath()}>
            {/* File info bar */}
            <div class="file-bar">
              <div class="file-info">
                <svg viewBox="0 0 24 24" width="20" height="20">
                  <path d="M9 18V5l12-2v13"/>
                  <circle cx="6" cy="18" r="3"/>
                  <circle cx="18" cy="16" r="3"/>
                </svg>
                <span class="file-name">{fileName()}</span>
              </div>
              <button class="change-file-btn" onClick={handleNewFile} disabled={isTranscribing()}>
                Change
              </button>
            </div>

            {/* Error display */}
            <Show when={error()}>
              <div class="error-banner">
                <div class="error-content">
                  <span>{error()}</span>
                  <Show when={isOOMError(error()!)}>
                    <div class="error-suggestion">
                      <svg viewBox="0 0 24 24" width="16" height="16">
                        <circle cx="12" cy="12" r="10"/>
                        <line x1="12" y1="8" x2="12" y2="12"/>
                        <line x1="12" y1="16" x2="12.01" y2="16"/>
                      </svg>
                      <span>Try selecting 'tiny' or 'base' model for your system</span>
                    </div>
                  </Show>
                </div>
                <button onClick={() => setError(null)}>Dismiss</button>
              </div>
            </Show>

            {/* Settings panel - hidden during transcription and when showing results */}
            <Show when={!isTranscribing() && !result()}>
              <section class="settings-panel">
                {/* Model Selection */}
                <div class="setting-group">
                  <label for="model-select" class="label-with-info">
                    Model
                    <InfoIcon
                      content="Smaller models (tiny, base) are faster but less accurate. Larger models provide better accuracy but need more VRAM. Models ending in '.en' are English-only and faster."
                      position="right"
                    />
                  </label>
                  <div class="select-wrapper">
                    <select
                      id="model-select"
                      value={selectedModel()}
                      onChange={(e) => setSelectedModel(e.currentTarget.value as WhisperModel)}
                    >
                      <For each={MODEL_OPTIONS}>
                        {(option) => (
                          <option value={option.value}>
                            {option.label}
                          </option>
                        )}
                      </For>
                    </select>
                    <svg class="select-arrow" viewBox="0 0 24 24">
                      <path d="M6 9l6 6 6-6"/>
                    </svg>
                  </div>
                  <span class="setting-hint">
                    {MODEL_OPTIONS.find(m => m.value === selectedModel())?.description}
                  </span>
                  <Show when={modelValidation() && modelValidation()!.status === 'warning'}>
                    <div class="model-warning">
                      <svg viewBox="0 0 24 24" width="14" height="14">
                        <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm1 15h-2v-2h2v2zm0-4h-2V7h2v6z" fill="currentColor"/>
                      </svg>
                      <For each={modelValidation()!.messages}>
                        {(message) => <span class="warning-text">{message}</span>}
                      </For>
                    </div>
                  </Show>
                </div>

                {/* Task Selection */}
                <div class="setting-group">
                  <label class="label-with-info">
                    Task
                    <InfoIcon
                      content="Transcribe converts speech to text in the original language. Translate converts speech to English text, regardless of the source language."
                      position="right"
                    />
                  </label>
                  <div class="task-toggle">
                    <button
                      class={`task-btn ${selectedTask() === 'transcribe' ? 'active' : ''}`}
                      onClick={() => setSelectedTask('transcribe')}
                    >
                      Transcribe
                    </button>
                    <button
                      class={`task-btn ${selectedTask() === 'translate' ? 'active' : ''}`}
                      onClick={() => setSelectedTask('translate')}
                    >
                      Translate to English
                    </button>
                  </div>
                </div>

                {/* Language Selection */}
                <div class="setting-group">
                  <label for="language-select" class="label-with-info">
                    Source Language
                    <InfoIcon
                      content="Auto-detect identifies the language automatically (recommended). Manually selecting a language can improve accuracy if you're certain of the source."
                      position="right"
                    />
                  </label>
                  <div class="select-wrapper">
                    <select
                      id="language-select"
                      value={selectedLanguage() || ''}
                      onChange={(e) => setSelectedLanguage(e.currentTarget.value || null)}
                    >
                      <For each={LANGUAGE_OPTIONS}>
                        {(option) => (
                          <option value={option.value || ''}>
                            {option.label}
                          </option>
                        )}
                      </For>
                    </select>
                    <svg class="select-arrow" viewBox="0 0 24 24">
                      <path d="M6 9l6 6 6-6"/>
                    </svg>
                  </div>
                </div>

                {/* Timestamps Toggle */}
                <div class="setting-group inline">
                  <label class="toggle-label">
                    <span class="toggle-switch">
                      <input
                        type="checkbox"
                        checked={includeTimestamps()}
                        onChange={(e) => setIncludeTimestamps(e.currentTarget.checked)}
                      />
                      <span class="toggle-slider"></span>
                    </span>
                    <span class="label-with-info">
                      Include timestamps
                      <InfoIcon
                        content="Generates time-coded segments. Enables SRT subtitle export for video subtitles and precise navigation."
                        position="right"
                      />
                    </span>
                  </label>
                </div>

                {/* Start button */}
                <button
                  class="start-btn"
                  onClick={handleTranscribe}
                  disabled={modelValidation()?.status === 'error'}
                  title={modelValidation()?.status === 'error' ? 'Cannot run: insufficient system resources' : ''}
                >
                  <svg viewBox="0 0 24 24" width="20" height="20">
                    <polygon points="5 3 19 12 5 21 5 3"/>
                  </svg>
                  Begin Transcription
                </button>
              </section>
            </Show>

            {/* Processing state - inline on page */}
            <Show when={isTranscribing()}>
              <section class="processing-section">
                <GreekScrollLoader />
                <h2>
                  {transcriptionProgress()?.phase === 'loading_model'
                    ? 'Loading Model...'
                    : 'Transcribing...'}
                </h2>
                <p>Processing your audio with the {selectedModel()} model</p>

                {/* Progress Bar */}
                <TranscriptionProgressBar progress={transcriptionProgress()} />

                <div class="processing-details">
                  <span>File: {fileName()}</span>
                  <span>Task: {selectedTask() === 'transcribe' ? 'Transcription' : 'Translation'}</span>
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
                      <span class="meta-label">Language</span>
                      <span class="meta-value">{res.language || 'Unknown'}</span>
                    </div>
                    <div class="meta-item">
                      <span class="meta-label">Duration</span>
                      <span class="meta-value">{formatTime(res.duration, false)}</span>
                    </div>
                    <div class="meta-item">
                      <span class="meta-label">Processing</span>
                      <span class="meta-value">{res.inference_time.toFixed(1)}s</span>
                    </div>
                    <div class="meta-item">
                      <span class="meta-label">Model</span>
                      <span class="meta-value">{res.model}</span>
                    </div>
                  </div>

                  {/* Transcript text */}
                  <div class="transcript-box">
                    <div class="transcript-header">
                      <h3>Transcript</h3>
                    </div>
                    <div class="transcript-content">
                      {res.text}
                    </div>
                    <div class="transcript-actions">
                      <div class="action-group">
                        <span class="action-label">Copy:</span>
                        <button class="action-btn" onClick={handleCopyPlainText} title="Copy plain text">
                          <svg viewBox="0 0 24 24" width="16" height="16">
                            <rect x="9" y="9" width="13" height="13" rx="2" ry="2"/>
                            <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/>
                          </svg>
                          <span>Text</span>
                        </button>
                        <Show when={includeTimestamps() && res.segments.length > 0}>
                          <button class="action-btn" onClick={handleCopySrt} title="Copy SRT format">
                            <svg viewBox="0 0 24 24" width="16" height="16">
                              <rect x="9" y="9" width="13" height="13" rx="2" ry="2"/>
                              <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/>
                            </svg>
                            <span>SRT</span>
                          </button>
                        </Show>
                      </div>
                      <div class="action-group">
                        <span class="action-label">Download:</span>
                        <button class="action-btn" onClick={handleExportPlainText} title="Download as .txt file">
                          <svg viewBox="0 0 24 24" width="16" height="16">
                            <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
                            <polyline points="7 10 12 15 17 10"/>
                            <line x1="12" y1="15" x2="12" y2="3"/>
                          </svg>
                          <span>.txt</span>
                        </button>
                        <Show when={includeTimestamps() && res.segments.length > 0}>
                          <button class="action-btn" onClick={handleExportSrt} title="Download as .srt file">
                            <svg viewBox="0 0 24 24" width="16" height="16">
                              <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
                              <polyline points="7 10 12 15 17 10"/>
                              <line x1="12" y1="15" x2="12" y2="3"/>
                            </svg>
                            <span>.srt</span>
                          </button>
                        </Show>
                      </div>
                    </div>
                  </div>

                  {/* Segments with timestamps */}
                  <Show when={includeTimestamps() && res.segments.length > 0}>
                    <div class="segments-box">
                      <button
                        class="segments-header"
                        onClick={() => setSegmentsExpanded(!segmentsExpanded())}
                      >
                        <span>Timestamps ({res.segments.length} segments)</span>
                        <svg class={`expand-icon ${segmentsExpanded() ? 'expanded' : ''}`} viewBox="0 0 24 24" width="18" height="18">
                          <path d="M6 9l6 6 6-6"/>
                        </svg>
                      </button>

                      <Show when={segmentsExpanded()}>
                        <div class="segments-content">
                          <For each={res.segments}>
                            {(segment) => (
                              <div class="segment-row">
                                <span class="segment-time">
                                  {formatTimestamp(segment.start)}
                                </span>
                                <span class="segment-text">{segment.text}</span>
                              </div>
                            )}
                          </For>
                        </div>
                      </Show>
                    </div>
                  </Show>

                  {/* New transcription button */}
                  <button class="new-file-btn" onClick={handleNewFile}>
                    <svg viewBox="0 0 24 24" width="18" height="18">
                      <line x1="12" y1="5" x2="12" y2="19"/>
                      <line x1="5" y1="12" x2="19" y2="12"/>
                    </svg>
                    New Transcription
                  </button>
                </section>
              )}
            </Show>
          </Show>
        </main>

        <div class="scroll-rod"></div>
      </div>
    </>
  );
};

export default Transcription;
