import { Component, createSignal, Show } from "solid-js";
import { useNavigate } from "@solidjs/router";
import { invoke } from "@tauri-apps/api/core";
import { save } from "@tauri-apps/plugin-dialog";
import { useTheme } from "../utils/theme";
import FileUploader from "../components/FileUploader";
import WaveformEditor from "../components/WaveformEditor";
import type { AudioFileState, TrimSelection, WaveformPeaks } from "../types/audio";
import "./AudioEditor.css";

const AudioEditor: Component = () => {
  const navigate = useNavigate();
  const { theme, toggleTheme } = useTheme();

  // Audio file state
  const [audioFile, setAudioFile] = createSignal<AudioFileState>({
    filePath: "",
    fileName: "",
    peaks: null,
    isLoading: false,
    error: null,
  });

  // Trim selection
  const [trimSelection, setTrimSelection] = createSignal<TrimSelection>({
    start: 0,
    end: 0,
  });

  // Trimming state
  const [isTrimming, setIsTrimming] = createSignal(false);
  const [trimError, setTrimError] = createSignal<string | null>(null);

  // Handle file selection (from uploader)
  const handleFileSelected = async (filePath: string) => {
    setAudioFile({
      filePath,
      fileName: filePath.split("/").pop() || filePath.split("\\").pop() || "Unknown",
      peaks: null,
      isLoading: true,
      error: null,
    });

    try {
      // Extract waveform peaks using existing Tauri command
      const peaks = await invoke<WaveformPeaks>("get_waveform_peaks", {
        filePath,
        numPeaks: 2000,
      });

      setAudioFile({
        ...audioFile(),
        peaks,
        isLoading: false,
      });

      // Initialize trim selection to full duration
      setTrimSelection({
        start: 0,
        end: peaks.duration_seconds,
      });

    } catch (err) {
      setAudioFile({
        ...audioFile(),
        isLoading: false,
        error: String(err),
      });
    }
  };

  // Handle trim operation
  const handleTrim = async () => {
    const file = audioFile();
    if (!file.filePath || !file.peaks) return;

    setIsTrimming(true);
    setTrimError(null);

    try {
      // Show save dialog
      const outputPath = await save({
        filters: [{
          name: "WAV Audio",
          extensions: ["wav"],
        }],
        defaultPath: `trimmed_${file.fileName.replace(/\.[^/.]+$/, "")}.wav`,
      });

      if (!outputPath) {
        setIsTrimming(false);
        return; // User cancelled
      }

      // Call trim command
      const selection = trimSelection();
      await invoke("trim_audio_file", {
        inputPath: file.filePath,
        outputPath,
        startSeconds: selection.start,
        endSeconds: selection.end,
      });

      setIsTrimming(false);
      // Success feedback could go here (toast notification, etc.)

    } catch (err) {
      setIsTrimming(false);
      setTrimError(String(err));
    }
  };

  // Handle trim selection change from waveform
  const handleSelectionChange = (start: number, end: number) => {
    setTrimSelection({ start, end });
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
          {/* Header with back button */}
          <header class="editor-header">
            <button class="back-button" onClick={() => navigate("/")}>
              ← Back to Home
            </button>
            <h1 class="editor-title">Audio Editor</h1>
            <p class="editor-subtitle">Trim and prepare your sermon recordings</p>
          </header>

          <div class="divider">
            <span class="divider-line"></span>
            <span class="divider-symbol">✤</span>
            <span class="divider-line"></span>
          </div>

          {/* File uploader section */}
          <Show when={!audioFile().filePath}>
            <FileUploader onFileSelected={handleFileSelected} />
          </Show>

          {/* Loading state */}
          <Show when={audioFile().isLoading}>
            <div class="loading-overlay">
              <div class="loading-content">
                <div class="loading-spinner"></div>
                <p>Analyzing audio file...</p>
                <p class="loading-info">Extracting waveform peaks</p>
              </div>
            </div>
          </Show>

          {/* Error state */}
          <Show when={audioFile().error}>
            <div class="error-state">
              <p>Error loading audio: {audioFile().error}</p>
              <button onClick={() => setAudioFile({
                filePath: "",
                fileName: "",
                peaks: null,
                isLoading: false,
                error: null,
              })}>
                Try Again
              </button>
            </div>
          </Show>

          {/* Trimming state */}
          <Show when={isTrimming()}>
            <div class="trimming-overlay">
              <div class="trimming-content">
                <div class="loading-spinner"></div>
                <p>Processing audio...</p>
                <p class="trim-info">Trimming and encoding to WAV format</p>
              </div>
            </div>
          </Show>

          {/* Waveform editor */}
          <Show when={audioFile().peaks}>
            <div class="editor-workspace">
              <div class="file-info">
                <h3>{audioFile().fileName}</h3>
                <p>
                  Duration: {audioFile().peaks?.duration_seconds.toFixed(2)}s |
                  Sample Rate: {audioFile().peaks?.sample_rate}Hz |
                  Channels: {audioFile().peaks?.channels}
                </p>
              </div>

              <WaveformEditor
                peaks={audioFile().peaks!}
                selection={trimSelection()}
                onSelectionChange={handleSelectionChange}
              />

              <div class="trim-controls">
                <div class="time-inputs">
                  <label>
                    Start:
                    <input
                      type="number"
                      step="0.1"
                      min="0"
                      max={audioFile().peaks?.duration_seconds || 0}
                      value={trimSelection().start}
                      onInput={(e) => setTrimSelection({
                        ...trimSelection(),
                        start: parseFloat(e.currentTarget.value),
                      })}
                    />
                    seconds
                  </label>

                  <label>
                    End:
                    <input
                      type="number"
                      step="0.1"
                      min="0"
                      max={audioFile().peaks?.duration_seconds || 0}
                      value={trimSelection().end}
                      onInput={(e) => setTrimSelection({
                        ...trimSelection(),
                        end: parseFloat(e.currentTarget.value),
                      })}
                    />
                    seconds
                  </label>

                  <p class="trim-duration">
                    Trim Duration: {(trimSelection().end - trimSelection().start).toFixed(2)}s
                  </p>
                </div>

                <button
                  class="trim-button"
                  onClick={handleTrim}
                  disabled={isTrimming()}
                >
                  {isTrimming() ? "Trimming..." : "Trim & Save"}
                </button>

                <Show when={trimError()}>
                  <p class="trim-error">{trimError()}</p>
                </Show>
              </div>

              <button
                class="load-new-file"
                onClick={() => setAudioFile({
                  filePath: "",
                  fileName: "",
                  peaks: null,
                  isLoading: false,
                  error: null,
                })}
              >
                Load Different File
              </button>
            </div>
          </Show>
        </main>

        <div class="scroll-rod"></div>
      </div>
    </>
  );
};

export default AudioEditor;
