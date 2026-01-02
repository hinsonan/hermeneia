import { Component, createSignal, onCleanup, Show } from "solid-js";
import { useNavigate } from "@solidjs/router";
import { invoke } from "@tauri-apps/api/core";
import { save } from "@tauri-apps/plugin-dialog";
import { useTheme } from "../utils/theme";
import FileUploader from "../components/FileUploader";
import WaveformEditor from "../components/WaveformEditor";
import GreekScrollLoader from "../components/GreekScrollLoader";
import type { AudioFileState, TrimSelection, WaveformPeaks } from "../types/audio";
import "./AudioEditor.css";

// Playback state returned from Rust backend
interface PlaybackInfo {
  is_playing: boolean;
  current_time: number;
  duration: number;
}

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

  // Playback state (synced from Rust backend)
  const [isPlaying, setIsPlaying] = createSignal(false);
  const [currentTime, setCurrentTime] = createSignal(0);

  // Polling interval for playback state
  let pollInterval: number | undefined;

  // Trimming state
  const [isTrimming, setIsTrimming] = createSignal(false);
  const [trimError, setTrimError] = createSignal<string | null>(null);

  // Start polling playback state from Rust backend
  const startPolling = () => {
    if (pollInterval) return;

    pollInterval = window.setInterval(async () => {
      try {
        const state = await invoke<PlaybackInfo>("get_playback_state");
        setIsPlaying(state.is_playing);
        setCurrentTime(state.current_time);
      } catch (err) {
        console.error("Failed to get playback state:", err);
      }
    }, 50); // Poll every 50ms for smooth playhead updates
  };

  const stopPolling = () => {
    if (pollInterval) {
      clearInterval(pollInterval);
      pollInterval = undefined;
    }
  };

  // Handle file selection (from uploader)
  const handleFileSelected = async (filePath: string) => {
    console.log("🔵 handleFileSelected called with:", filePath);
    // Stop any existing playback
    try {
      await invoke("stop_audio");
    } catch (_) {
      // Ignore errors if nothing was playing
    }

    setIsPlaying(false);
    setCurrentTime(0);

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

      // Initialize playback in paused state so seeking works immediately
      await invoke("play_audio", { filePath });
      await invoke("pause_audio");

      // Start polling for playback state updates
      startPolling();

    } catch (err) {
      setAudioFile({
        ...audioFile(),
        isLoading: false,
        error: String(err),
      });
    }
  };

  // Playback controls using Rust backend
  const togglePlayPause = async () => {
    console.log("🟢 togglePlayPause called, isPlaying:", isPlaying());
    const file = audioFile();
    if (!file.filePath) return;

    try {
      if (isPlaying()) {
        console.log("  -> Pausing");
        await invoke("pause_audio");
      } else {
        // If file is loaded, just resume; otherwise start fresh
        const state = await invoke<PlaybackInfo>("get_playback_state");
        console.log("  -> Current state:", state);
        if (state.duration === 0) {
          // No file loaded - shouldn't happen but handle it
          console.log("  -> Calling play_audio");
          await invoke("play_audio", { filePath: file.filePath });
        } else {
          // File is loaded, just resume playback
          console.log("  -> Calling resume_audio");
          await invoke("resume_audio");
        }
      }
    } catch (err) {
      console.error("Playback error:", err);
    }
  };

  const handleSeek = async (time: number) => {
    try {
      await invoke("seek_audio", { timeSeconds: time });
      setCurrentTime(time);
    } catch (err) {
      console.error("Seek error:", err);
    }
  };

  const handleStop = async () => {
    console.log("🛑 handleStop called");
    try {
      await invoke("stop_audio");
      setIsPlaying(false);
      setCurrentTime(0);
    } catch (err) {
      console.error("Stop error:", err);
    }
  };

  // Format time for display (MM:SS)
  const formatTime = (seconds: number): string => {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, "0")}`;
  };

  // Handle trim operation
  const handleTrim = async () => {
    const file = audioFile();
    if (!file.filePath || !file.peaks) return;

    // Stop playback before trimming
    await handleStop();

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

    } catch (err) {
      setIsTrimming(false);
      setTrimError(String(err));
    }
  };

  // Handle trim selection change from waveform
  const handleSelectionChange = (start: number, end: number) => {
    setTrimSelection({ start, end });
  };

  // Cleanup on unmount
  onCleanup(async () => {
    console.log("🔴 onCleanup running!");
    stopPolling();
    try {
      await invoke("stop_audio");
    } catch (_) {
      // Ignore cleanup errors
    }
  });

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
                <GreekScrollLoader />
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
                <GreekScrollLoader />
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
                currentTime={currentTime()}
                onSeek={handleSeek}
              />

              {/* Playback controls */}
              <div class="playback-controls">
                <button
                  class="play-button"
                  onClick={togglePlayPause}
                  aria-label={isPlaying() ? "Pause" : "Play"}
                >
                  <Show when={isPlaying()} fallback={
                    <svg viewBox="0 0 24 24" class="play-icon">
                      <polygon points="5 3 19 12 5 21 5 3"/>
                    </svg>
                  }>
                    <svg viewBox="0 0 24 24" class="pause-icon">
                      <rect x="6" y="4" width="4" height="16"/>
                      <rect x="14" y="4" width="4" height="16"/>
                    </svg>
                  </Show>
                </button>

                <div class="time-display">
                  <span class="current-time">{formatTime(currentTime())}</span>
                  <span class="time-separator">/</span>
                  <span class="total-time">{formatTime(audioFile().peaks?.duration_seconds || 0)}</span>
                </div>

                <button
                  class="stop-button"
                  onClick={handleStop}
                  aria-label="Stop and reset"
                >
                  <svg viewBox="0 0 24 24" class="stop-icon">
                    <rect x="6" y="6" width="12" height="12"/>
                  </svg>
                </button>
              </div>

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
                onClick={async () => {
                  // Stop playback before loading new file
                  await handleStop();
                  stopPolling();
                  setAudioFile({
                    filePath: "",
                    fileName: "",
                    peaks: null,
                    isLoading: false,
                    error: null,
                  });
                }}
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
