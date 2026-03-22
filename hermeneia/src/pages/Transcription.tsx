import { Component, createSignal, For, Show, onCleanup, onMount, createEffect, createMemo } from "solid-js";
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
import ConfirmDialog from "../components/ConfirmDialog";
import DownloadProgressBar from "../components/DownloadProgressBar";
import type {
  WhisperModel,
  TranscriptionTask,
  TranscriptResult,
  TranscriptionProgress,
  ModelOption,
  LanguageOption,
  SystemCapabilities,
  ModelValidation,
  AnnotationProgress,
  AnnotatedResult,
  SpeakerDevice,
  SpeakerModelKey,
  SpeakerModelRequirement,
} from "../types/transcription";
import type { DownloadProgress } from "../types/models";
import "./Transcription.css";

type JobProgress = TranscriptionProgress | AnnotationProgress;

const MODEL_OPTIONS: ModelOption[] = [
  { value: "tiny", label: "Tiny", description: "Fastest, least accurate (~1GB VRAM)" },
  { value: "tiny.en", label: "Tiny (English)", description: "English-only, faster" },
  { value: "base", label: "Base", description: "Good balance of speed and accuracy (~1GB VRAM)" },
  { value: "base.en", label: "Base (English)", description: "English-only, recommended" },
  { value: "small", label: "Small", description: "Better accuracy (~2GB VRAM)" },
  { value: "small.en", label: "Small (English)", description: "English-only, better accuracy" },
  { value: "medium", label: "Medium", description: "High accuracy (~5GB VRAM)" },
  { value: "medium.en", label: "Medium (English)", description: "English-only, high accuracy" },
  { value: "large", label: "Large", description: "Highest accuracy (~10GB VRAM)" },
  { value: "large-v2", label: "Large v2", description: "Improved large model" },
  { value: "large-v3", label: "Large v3", description: "Latest large model" },
];

const LANGUAGE_OPTIONS: LanguageOption[] = [
  { value: null, label: "Auto-detect" },
  { value: "en", label: "English" },
  { value: "es", label: "Spanish" },
  { value: "fr", label: "French" },
  { value: "de", label: "German" },
  { value: "it", label: "Italian" },
  { value: "pt", label: "Portuguese" },
  { value: "ru", label: "Russian" },
  { value: "zh", label: "Chinese" },
  { value: "ja", label: "Japanese" },
  { value: "ko", label: "Korean" },
  { value: "ar", label: "Arabic" },
  { value: "el", label: "Greek" },
  { value: "he", label: "Hebrew" },
  { value: "hi", label: "Hindi" },
  { value: "nl", label: "Dutch" },
  { value: "pl", label: "Polish" },
  { value: "tr", label: "Turkish" },
  { value: "vi", label: "Vietnamese" },
  { value: "th", label: "Thai" },
];

const Transcription: Component = () => {
  const navigate = useNavigate();
  const { toggleTheme } = useTheme();

  // File state
  const [filePath, setFilePath] = createSignal<string>("");
  const [fileName, setFileName] = createSignal<string>("");

  // Settings
  const [mode, setMode] = createSignal<"transcribe" | "annotate">("transcribe");
  const [selectedModel, setSelectedModel] = createSignal<WhisperModel>("tiny");
  const [selectedTask, setSelectedTask] = createSignal<TranscriptionTask>("transcribe");
  const [selectedLanguage, setSelectedLanguage] = createSignal<string | null>(null);
  const [includeTimestamps, setIncludeTimestamps] = createSignal(true);
  const [selectedSpeakerModel, setSelectedSpeakerModel] = createSignal<SpeakerModelKey>("english");
  const [selectedSpeakerDevice, setSelectedSpeakerDevice] = createSignal<SpeakerDevice>("cpu");
  const [numSpeakers, setNumSpeakers] = createSignal<number | null>(null);
  const [diarizeThreshold, setDiarizeThreshold] = createSignal<number>(0.5);

  // Processing state
  const [isTranscribing, setIsTranscribing] = createSignal(false);
  const [error, setError] = createSignal<string | null>(null);
  const [result, setResult] = createSignal<TranscriptResult | null>(null);
  const [annotatedResult, setAnnotatedResult] = createSignal<AnnotatedResult | null>(null);
  const [speakerNames, setSpeakerNames] = createSignal<Record<string, string>>({});
  const [transcriptionProgress, setTranscriptionProgress] = createSignal<JobProgress | null>(null);

  // Cancel dialog
  const [showCancelDialog, setShowCancelDialog] = createSignal(false);

  // Model download state
  const [isDownloading, setIsDownloading] = createSignal(false);
  const [modelDownloadProgress, setModelDownloadProgress] = createSignal<DownloadProgress | null>(null);
  const [isPreparingSpeaker, setIsPreparingSpeaker] = createSignal(false);

  // System capability detection
  const [systemCapabilities, setSystemCapabilities] = createSignal<SystemCapabilities | null>(null);
  const [modelValidation, setModelValidation] = createSignal<ModelValidation | null>(null);
  const [speakerModelRequirements, setSpeakerModelRequirements] = createSignal<SpeakerModelRequirement[]>([]);

  // Track unlisten functions for cleanup
  let progressUnlisten: UnlistenFn | null = null;
  let downloadUnlisten: UnlistenFn | null = null;

  const isAnnotateMode = createMemo(() => mode() === "annotate");

  // Check if selected model is English-only
  const isEnglishOnlyModel = createMemo(() => {
    return selectedModel().endsWith(".en");
  });

  const availableSpeakerDevices = createMemo(() => {
    const deviceType = systemCapabilities()?.gpu_info?.device_type;
    if (deviceType === "NvidiaCuda") return ["cuda", "cpu"] as SpeakerDevice[];
    if (deviceType === "AppleMetal") return ["coreml", "cpu"] as SpeakerDevice[];
    return ["cpu"] as SpeakerDevice[];
  });

  const activeSpeakerRequirement = createMemo(() =>
    speakerModelRequirements().find((m) => m.key === selectedSpeakerModel())
  );

  const annotatedSegmentsWithNames = createMemo(() => {
    const res = annotatedResult();
    if (!res) return [];
    return res.segments.map((seg) => ({
      ...seg,
      speaker_name: speakerNames()[String(seg.speaker)] || seg.speaker_name || `Speaker ${seg.speaker}`,
    }));
  });

  const speakerIds = createMemo(() => {
    const ids = Array.from(new Set(annotatedSegmentsWithNames().map((s) => s.speaker)));
    return ids.sort((a, b) => a - b);
  });

  // Filter language options based on model selection
  const availableLanguages = createMemo(() => {
    if (isEnglishOnlyModel()) {
      return LANGUAGE_OPTIONS.filter((lang) => lang.value === "en");
    }
    return LANGUAGE_OPTIONS;
  });

  // Auto-select English when English-only model is selected
  createEffect(() => {
    if (isEnglishOnlyModel()) {
      setSelectedLanguage("en");
    }
  });

  // Force timestamps for annotate mode
  createEffect(() => {
    if (isAnnotateMode()) {
      setIncludeTimestamps(true);
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

  // Fetch system capabilities + speaker requirements on mount
  onMount(async () => {
    try {
      const caps = await invoke<SystemCapabilities>("get_system_capabilities");
      setSystemCapabilities(caps);
      const deviceType = caps.gpu_info?.device_type;
      if (deviceType === "NvidiaCuda") {
        setSelectedSpeakerDevice("cuda");
      } else if (deviceType === "AppleMetal") {
        setSelectedSpeakerDevice("coreml");
      } else {
        setSelectedSpeakerDevice("cpu");
      }
    } catch (err) {
      console.warn("Failed to get system capabilities:", err);
    }

    try {
      const requirements = await invoke<SpeakerModelRequirement[]>("list_speaker_model_requirements");
      setSpeakerModelRequirements(requirements);
    } catch (err) {
      console.warn("Failed to load speaker model requirements:", err);
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
      const validation = await invoke<ModelValidation>("validate_model_selection", {
        model,
        forceCpu: false,
      });
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
    setAnnotatedResult(null);
    setError(null);
  };

  // Format timestamp for display (MM:SS.ms)
  const formatTimestamp = (seconds: number | null): string => {
    if (seconds === null) return "--:--";
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    const ms = Math.floor((seconds % 1) * 100);
    return `${mins.toString().padStart(2, "0")}:${secs.toString().padStart(2, "0")}.${ms
      .toString()
      .padStart(2, "0")}`;
  };

  // Format timestamp for SRT format (HH:MM:SS,mmm)
  const formatSrtTimestamp = (seconds: number | null): string => {
    if (seconds === null) return "00:00:00,000";
    const hours = Math.floor(seconds / 3600);
    const mins = Math.floor((seconds % 3600) / 60);
    const secs = Math.floor(seconds % 60);
    const ms = Math.floor((seconds % 1) * 1000);
    return `${hours.toString().padStart(2, "0")}:${mins
      .toString()
      .padStart(2, "0")}:${secs.toString().padStart(2, "0")},${ms.toString().padStart(3, "0")}`;
  };

  // Generate plain text content
  const getPlainTextContent = (): string => {
    const transcript = result();
    if (!transcript) return "";
    return transcript.text;
  };

  // Generate SRT content (transcription or annotate mode)
  const getSrtContent = (): string => {
    if (isAnnotateMode()) {
      const segments = annotatedSegmentsWithNames();
      if (segments.length === 0) return "";

      return segments
        .map((seg, index) => {
          const startTime = formatSrtTimestamp(seg.start);
          const endTime = formatSrtTimestamp(seg.end);
          return `${index + 1}\n${startTime} --> ${endTime}\n[${seg.speaker_name}] ${seg.text.trim()}\n`;
        })
        .join("\n");
    }

    const transcript = result();
    if (!transcript || transcript.segments.length === 0) return "";

    return transcript.segments
      .map((seg, index) => {
        const startTime = formatSrtTimestamp(seg.start);
        const endTime = formatSrtTimestamp(seg.end);
        return `${index + 1}\n${startTime} --> ${endTime}\n${seg.text.trim()}\n`;
      })
      .join("\n");
  };

  // Check if error is OOM-related
  const isOOMError = (err: string): boolean => {
    return (
      err.toLowerCase().includes("out of memory") ||
      err.toLowerCase().includes("oom") ||
      err.includes("OutOfMemory")
    );
  };

  // Map whisper model key to HuggingFace model ID for cache checks
  const whisperModelId = (model: string): string => `openai/whisper-${model}`;

  // Download a model before transcription if not cached
  const ensureModelDownloaded = async (): Promise<boolean> => {
    const hfId = whisperModelId(selectedModel());
    const cached = await invoke<boolean>("is_model_cached", { modelId: hfId });
    if (cached) return true;

    setIsDownloading(true);
    setModelDownloadProgress(null);

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
      const modelLabel = MODEL_OPTIONS.find((m) => m.value === selectedModel())?.label || selectedModel();
      await invoke("download_model", {
        modelId: hfId,
        modelName: `Whisper ${modelLabel}`,
      });
      return true;
    } catch (err) {
      const errStr = String(err);
      if (errStr.includes("cancelled") || errStr.includes("Download cancelled")) {
        return false;
      }
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

  const ensureSpeakerModelDownloaded = async (): Promise<boolean> => {
    const requirement = activeSpeakerRequirement();
    if (requirement?.is_cached) {
      return true;
    }

    setIsPreparingSpeaker(true);
    try {
      await invoke("ensure_speaker_model_downloaded", {
        model: selectedSpeakerModel(),
      });
      const requirements = await invoke<SpeakerModelRequirement[]>("list_speaker_model_requirements");
      setSpeakerModelRequirements(requirements);
      return true;
    } catch (err) {
      setError(`Speaker model download failed: ${String(err)}`);
      return false;
    } finally {
      setIsPreparingSpeaker(false);
    }
  };

  // Start transcription/annotation
  const handleTranscribe = async () => {
    if (!filePath()) return;

    setError(null);

    const modelReady = await ensureModelDownloaded();
    if (!modelReady) return;

    if (isAnnotateMode()) {
      const speakerReady = await ensureSpeakerModelDownloaded();
      if (!speakerReady) return;
    }

    setIsTranscribing(true);
    if (isAnnotateMode()) {
      setTranscriptionProgress({
        phase: "starting",
        current: null,
        total: null,
        message: "Starting annotation...",
        indeterminate: true,
      } as AnnotationProgress);
    } else {
      setTranscriptionProgress(null);
    }

    try {
      const eventName = isAnnotateMode() ? "annotation-progress" : "transcription-progress";

      progressUnlisten = await listen(eventName, (event) => {
        const payload = event.payload as AnnotationProgress | TranscriptionProgress;
        if (isAnnotateMode()) {
          setTranscriptionProgress(payload as AnnotationProgress);
        } else {
          setTranscriptionProgress(payload as TranscriptionProgress);
        }
      });
    } catch (err) {
      console.warn("Failed to set up progress listener:", err);
    }

    try {
      if (isAnnotateMode()) {
        const annotated = await invoke<AnnotatedResult>("annotate_audio_file", {
          filePath: filePath(),
          transcribeModel: selectedModel(),
          speakerModel: selectedSpeakerModel(),
          task: selectedTask(),
          language: selectedLanguage(),
          timestamps: true,
          numSpeakers: numSpeakers(),
          threshold: diarizeThreshold(),
          device: selectedSpeakerDevice(),
          speakerNames: speakerNames(),
        });

        setResult(null);
        setAnnotatedResult(annotated);
        setSpeakerNames(annotated.speaker_names || {});
      } else {
        const transcriptResult = await invoke<TranscriptResult>("transcribe_audio_file", {
          filePath: filePath(),
          model: selectedModel(),
          task: selectedTask(),
          language: selectedLanguage(),
          timestamps: includeTimestamps(),
        });

        setAnnotatedResult(null);
        setResult(transcriptResult);
      }
    } catch (err) {
      const errStr = String(err);
      if (errStr.includes("Operation cancelled")) {
        navigate("/");
        return;
      }
      setError(errStr);
    } finally {
      if (progressUnlisten) {
        progressUnlisten();
        progressUnlisten = null;
      }
      setIsTranscribing(false);
      setTranscriptionProgress(null);
    }
  };

  // Generate plain text for annotated result: "[MM:SS] Speaker: text"
  const getAnnotatedPlainTextContent = (): string => {
    return annotatedSegmentsWithNames()
      .map((seg) => {
        const startMin = Math.floor(seg.start / 60).toString().padStart(2, "0");
        const startSec = Math.floor(seg.start % 60).toString().padStart(2, "0");
        return `[${startMin}:${startSec}] ${seg.speaker_name}: ${seg.text}`;
      })
      .join("\n");
  };

  const handleCopyAnnotatedText = async () => {
    const text = getAnnotatedPlainTextContent();
    if (!text) return;
    try {
      await navigator.clipboard.writeText(text);
    } catch (err) {
      console.error("Failed to copy annotated text:", err);
    }
  };

  const handleExportAnnotatedText = async () => {
    if (annotatedSegmentsWithNames().length === 0) return;
    try {
      const outputPath = await save({
        filters: [{ name: "Text Files", extensions: ["txt"] }],
        defaultPath: `${fileName().replace(/\.[^/.]+$/, "")}_annotated.txt`,
      });
      if (!outputPath) return;
      await invoke("write_text_file", {
        path: outputPath,
        content: getAnnotatedPlainTextContent(),
      });
    } catch (err) {
      console.error("Failed to export annotated text:", err);
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
        filters: [
          {
            name: "Text Files",
            extensions: ["txt"],
          },
        ],
        defaultPath: `${fileName().replace(/\.[^/.]+$/, "")}_transcript.txt`,
      });

      if (!outputPath) return;

      await invoke("write_text_file", {
        path: outputPath,
        content: transcript.text,
      });
    } catch (err) {
      console.error("Failed to export transcript:", err);
    }
  };

  // Export transcript as SRT file
  const handleExportSrt = async () => {
    const hasSegments = isAnnotateMode()
      ? annotatedSegmentsWithNames().length > 0
      : !!result() && result()!.segments.length > 0;

    if (!hasSegments) return;

    try {
      const outputPath = await save({
        filters: [
          {
            name: "SRT Subtitle Files",
            extensions: ["srt"],
          },
        ],
        defaultPath: `${fileName().replace(/\.[^/.]+$/, "")}${isAnnotateMode() ? "_annotated" : ""}.srt`,
      });

      if (!outputPath) return;

      const content = getSrtContent();

      await invoke("write_text_file", {
        path: outputPath,
        content,
      });
    } catch (err) {
      console.error("Failed to export SRT:", err);
    }
  };

  const updateSpeakerName = (speakerId: number, value: string) => {
    setSpeakerNames((prev) => ({ ...prev, [String(speakerId)]: value }));
  };

  // Handle back button - show confirm dialog if transcribing
  const handleBack = () => {
    if (isTranscribing()) {
      setShowCancelDialog(true);
    } else {
      navigate("/");
    }
  };

  // Confirm cancellation and navigate home immediately
  const handleConfirmCancel = () => {
    setShowCancelDialog(false);
    invoke("cancel_inference").catch(() => {});
    navigate("/");
  };

  // Reset for new file
  const handleNewFile = () => {
    setFilePath("");
    setFileName("");
    setResult(null);
    setAnnotatedResult(null);
    setSpeakerNames({});
    setError(null);
    setNumSpeakers(null);
    setDiarizeThreshold(0.5);
  };

  return (
    <>
      {/* Theme Toggle */}
      <button class="theme-toggle" onClick={toggleTheme} aria-label="Toggle dark mode">
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
          <header class="page-header">
            <button class="back-button" onClick={handleBack}>
              <svg viewBox="0 0 24 24" width="20" height="20">
                <path d="M19 12H5M12 19l-7-7 7-7" />
              </svg>
              <span>Home</span>
            </button>
            <h1>{isAnnotateMode() ? "Transcription + Annotation" : "Transcription"}</h1>
          </header>

          <Show when={!filePath()}>
            <section class="upload-section">
              <FileUploader onFileSelected={handleFileSelected} />
            </section>
          </Show>

          <Show when={filePath()}>
            <div class="file-bar">
              <div class="file-info">
                <svg viewBox="0 0 24 24" width="20" height="20">
                  <path d="M9 18V5l12-2v13" />
                  <circle cx="6" cy="18" r="3" />
                  <circle cx="18" cy="16" r="3" />
                </svg>
                <span class="file-name">{fileName()}</span>
              </div>
              <button class="change-file-btn" onClick={handleNewFile} disabled={isTranscribing()}>
                Change
              </button>
            </div>

            <Show when={error()}>
              <div class="error-banner">
                <div class="error-content">
                  <span>{error()}</span>
                  <Show when={isOOMError(error()!)}>
                    <div class="error-suggestion">
                      <svg viewBox="0 0 24 24" width="16" height="16">
                        <circle cx="12" cy="12" r="10" />
                        <line x1="12" y1="8" x2="12" y2="12" />
                        <line x1="12" y1="16" x2="12.01" y2="16" />
                      </svg>
                      <span>Try selecting 'tiny' or 'base' model for your system</span>
                    </div>
                  </Show>
                </div>
                <button onClick={() => setError(null)}>Dismiss</button>
              </div>
            </Show>

            <Show when={!isTranscribing() && !isDownloading() && !isPreparingSpeaker() && !result() && !annotatedResult()}>
              <section class="settings-panel">
                <div class="setting-group">
                  <label class="label-with-info">
                    Mode
                    <InfoIcon
                      content="Transcription returns plain transcript output. Annotate runs speaker diarization + transcription and forces speaker-labeled SRT output."
                      position="right"
                    />
                  </label>
                  <div class="task-toggle">
                    <button
                      class={`task-btn ${!isAnnotateMode() ? "active" : ""}`}
                      onClick={() => setMode("transcribe")}
                    >
                      Transcribe
                    </button>
                    <button
                      class={`task-btn ${isAnnotateMode() ? "active" : ""}`}
                      onClick={() => setMode("annotate")}
                    >
                      Annotate
                    </button>
                  </div>
                </div>

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
                      <For each={MODEL_OPTIONS}>{(option) => <option value={option.value}>{option.label}</option>}</For>
                    </select>
                    <svg class="select-arrow" viewBox="0 0 24 24">
                      <path d="M6 9l6 6 6-6" />
                    </svg>
                  </div>
                  <span class="setting-hint">{MODEL_OPTIONS.find((m) => m.value === selectedModel())?.description}</span>
                  <Show when={modelValidation() && modelValidation()!.status === "warning"}>
                    <div class="model-warning">
                      <svg viewBox="0 0 24 24" width="14" height="14">
                        <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm1 15h-2v-2h2v2zm0-4h-2V7h2v6z" fill="currentColor" />
                      </svg>
                      <For each={modelValidation()!.messages}>{(message) => <span class="warning-text">{message}</span>}</For>
                    </div>
                  </Show>
                </div>

                <div class="setting-group">
                  <label class="label-with-info">
                    Task
                    <InfoIcon
                      content="Transcribe keeps original language text. Translate converts speech to English text."
                      position="right"
                    />
                  </label>
                  <div class="task-toggle">
                    <button
                      class={`task-btn ${selectedTask() === "transcribe" ? "active" : ""}`}
                      onClick={() => setSelectedTask("transcribe")}
                    >
                      Transcribe
                    </button>
                    <button
                      class={`task-btn ${selectedTask() === "translate" ? "active" : ""}`}
                      onClick={() => setSelectedTask("translate")}
                    >
                      Translate to English
                    </button>
                  </div>
                </div>

                <div class="setting-group">
                  <label for="language-select" class="label-with-info">
                    Source Language
                    <InfoIcon
                      content={
                        isEnglishOnlyModel()
                          ? "English-only models can only transcribe English audio."
                          : "Auto-detect identifies the language automatically (recommended). Manually selecting a language can improve accuracy if you're certain of the source."
                      }
                      position="right"
                    />
                  </label>
                  <div class="select-wrapper">
                    <select
                      id="language-select"
                      value={selectedLanguage() || ""}
                      onChange={(e) => setSelectedLanguage(e.currentTarget.value || null)}
                      disabled={isEnglishOnlyModel()}
                    >
                      <For each={availableLanguages()}>
                        {(option) => <option value={option.value || ""}>{option.label}</option>}
                      </For>
                    </select>
                    <svg class="select-arrow" viewBox="0 0 24 24">
                      <path d="M6 9l6 6 6-6" />
                    </svg>
                  </div>
                </div>

                <Show when={isAnnotateMode()}>
                  <>
                    <div class="setting-group">
                      <label for="speaker-model-select">Speaker Model</label>
                      <div class="select-wrapper">
                        <select
                          id="speaker-model-select"
                          value={selectedSpeakerModel()}
                          onChange={(e) => setSelectedSpeakerModel(e.currentTarget.value as SpeakerModelKey)}
                        >
                          <For each={speakerModelRequirements()}>
                            {(m) => <option value={m.key}>{m.key.charAt(0).toUpperCase() + m.key.slice(1)} ({m.approx_size_mb.toFixed(1)} MB)</option>}
                          </For>
                        </select>
                        <svg class="select-arrow" viewBox="0 0 24 24">
                          <path d="M6 9l6 6 6-6" />
                        </svg>
                      </div>
                    </div>

                    <div class="setting-group">
                      <label for="speaker-device-select">Speaker Device</label>
                      <div class="select-wrapper">
                        <select
                          id="speaker-device-select"
                          value={selectedSpeakerDevice()}
                          onChange={(e) => setSelectedSpeakerDevice(e.currentTarget.value as SpeakerDevice)}
                        >
                          <For each={availableSpeakerDevices()}>{(d) => <option value={d}>{d.toUpperCase()}</option>}</For>
                        </select>
                        <svg class="select-arrow" viewBox="0 0 24 24">
                          <path d="M6 9l6 6 6-6" />
                        </svg>
                      </div>
                      <span class="setting-hint">
                        Annotate mode forces SRT output with speaker labels (ID or edited names).
                      </span>
                    </div>

                    <div class="setting-group">
                      <label for="num-speakers-input">Expected Speakers</label>
                      <input
                        id="num-speakers-input"
                        type="number"
                        min="1"
                        max="20"
                        placeholder="Auto-detect"
                        value={numSpeakers() ?? ""}
                        onInput={(e) => {
                          const v = e.currentTarget.value;
                          setNumSpeakers(v === "" ? null : parseInt(v, 10));
                        }}
                        class="number-input"
                      />
                    </div>

                    <div class="setting-group">
                      <label for="diarize-threshold-input">
                        Clustering Threshold
                        <span class="threshold-value"> ({diarizeThreshold().toFixed(2)})</span>
                      </label>
                      <input
                        id="diarize-threshold-input"
                        type="range"
                        min="0.1"
                        max="0.9"
                        step="0.05"
                        value={diarizeThreshold()}
                        onInput={(e) => setDiarizeThreshold(parseFloat(e.currentTarget.value))}
                        class="range-slider"
                      />
                      <span class="setting-hint">Lower = more speakers detected</span>
                    </div>
                  </>
                </Show>

                <div class="setting-group inline">
                  <label class="toggle-label">
                    <span class="toggle-switch">
                      <input
                        type="checkbox"
                        checked={includeTimestamps()}
                        onChange={(e) => setIncludeTimestamps(e.currentTarget.checked)}
                        disabled={isAnnotateMode()}
                      />
                      <span class="toggle-slider"></span>
                    </span>
                    <span class="label-with-info">
                      Include timestamps
                      <InfoIcon
                        content={
                          isAnnotateMode()
                            ? "Annotate mode always requires timestamps and SRT output."
                            : "Generates time-coded segments. Enables SRT subtitle export for video subtitles and precise navigation."
                        }
                        position="right"
                      />
                    </span>
                  </label>
                </div>

                <button
                  class="start-btn"
                  onClick={handleTranscribe}
                  disabled={modelValidation()?.status === "error"}
                  title={modelValidation()?.status === "error" ? "Cannot run: insufficient system resources" : ""}
                >
                  <svg viewBox="0 0 24 24" width="20" height="20">
                    <polygon points="5 3 19 12 5 21 5 3" />
                  </svg>
                  {isAnnotateMode() ? "Begin Annotation" : "Begin Transcription"}
                </button>
              </section>
            </Show>

            <Show when={isDownloading()}>
              <section class="processing-section">
                <h2>Downloading Model...</h2>
                <p>Downloading the {selectedModel()} model before processing</p>
                <DownloadProgressBar
                  progress={modelDownloadProgress()}
                  onCancel={() => invoke("cancel_download").catch(() => {})}
                />
              </section>
            </Show>

            <Show when={isPreparingSpeaker()}>
              <section class="processing-section">
                <GreekScrollLoader />
                <h2>Preparing Speaker Model...</h2>
                <p>Downloading and caching speaker diarization model bundle</p>
              </section>
            </Show>

            <Show when={isTranscribing()}>
              <section class="processing-section">
                <GreekScrollLoader />
                <h2>
                  {transcriptionProgress()?.phase === "loading_model"
                    || transcriptionProgress()?.phase === "starting"
                    || transcriptionProgress()?.phase === "decoding_audio"
                    || transcriptionProgress()?.phase === "preparing_audio"
                    || transcriptionProgress()?.phase === "loading_speaker_model"
                    || transcriptionProgress()?.phase === "loading_transcription_model"
                    ? "Loading / Preparing..."
                    : isAnnotateMode()
                    ? transcriptionProgress()?.message?.startsWith("Diarizing")
                      ? "Diarizing..."
                      : transcriptionProgress()?.message?.startsWith("Transcribing")
                      ? "Transcribing..."
                      : "Annotating..."
                    : "Transcribing..."}
                </h2>
                <p>
                  Processing your audio with the {selectedModel()} model
                  {isAnnotateMode() ? " + speaker diarization" : ""}
                </p>

                <TranscriptionProgressBar progress={transcriptionProgress()} />

                <div class="processing-details">
                  <span>File: {fileName()}</span>
                  <span>Task: {isAnnotateMode() ? "Annotation" : selectedTask() === "transcribe" ? "Transcription" : "Translation"}</span>
                </div>
              </section>
            </Show>

            <Show when={result()} keyed>
              {(res) => (
                <section class="results-section">
                  <div class="result-meta">
                    <div class="meta-item">
                      <span class="meta-label">Language</span>
                      <span class="meta-value">{res.language || "Unknown"}</span>
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

                  <div class="transcript-box">
                    <div class="transcript-header">
                      <h3>Transcript</h3>
                    </div>
                    <div class="transcript-content">{res.text}</div>
                    <div class="transcript-actions">
                      <div class="action-group">
                        <span class="action-label">Copy:</span>
                        <button class="action-btn" onClick={handleCopyPlainText} title="Copy plain text">
                          <svg viewBox="0 0 24 24" width="16" height="16">
                            <rect x="9" y="9" width="13" height="13" rx="2" ry="2" />
                            <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1" />
                          </svg>
                          <span>Text</span>
                        </button>
                        <Show when={includeTimestamps() && res.segments.length > 0}>
                          <button class="action-btn" onClick={handleCopySrt} title="Copy SRT format">
                            <svg viewBox="0 0 24 24" width="16" height="16">
                              <rect x="9" y="9" width="13" height="13" rx="2" ry="2" />
                              <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1" />
                            </svg>
                            <span>SRT</span>
                          </button>
                        </Show>
                      </div>
                      <div class="action-group">
                        <span class="action-label">Download:</span>
                        <button class="action-btn" onClick={handleExportPlainText} title="Download as .txt file">
                          <svg viewBox="0 0 24 24" width="16" height="16">
                            <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
                            <polyline points="7 10 12 15 17 10" />
                            <line x1="12" y1="15" x2="12" y2="3" />
                          </svg>
                          <span>.txt</span>
                        </button>
                        <Show when={includeTimestamps() && res.segments.length > 0}>
                          <button class="action-btn" onClick={handleExportSrt} title="Download as .srt file">
                            <svg viewBox="0 0 24 24" width="16" height="16">
                              <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
                              <polyline points="7 10 12 15 17 10" />
                              <line x1="12" y1="15" x2="12" y2="3" />
                            </svg>
                            <span>.srt</span>
                          </button>
                        </Show>
                      </div>
                    </div>
                  </div>

                  <Show when={includeTimestamps() && res.segments.length > 0}>
                    <div class="segments-box">
                      <button class="segments-header" onClick={() => setSegmentsExpanded(!segmentsExpanded())}>
                        <span>Timestamps ({res.segments.length} segments)</span>
                        <svg
                          class={`expand-icon ${segmentsExpanded() ? "expanded" : ""}`}
                          viewBox="0 0 24 24"
                          width="18"
                          height="18"
                        >
                          <path d="M6 9l6 6 6-6" />
                        </svg>
                      </button>

                      <Show when={segmentsExpanded()}>
                        <div class="segments-content">
                          <For each={res.segments}>
                            {(segment) => (
                              <div class="segment-row">
                                <span class="segment-time">{formatTimestamp(segment.start)}</span>
                                <span class="segment-text">{segment.text}</span>
                              </div>
                            )}
                          </For>
                        </div>
                      </Show>
                    </div>
                  </Show>

                  <button class="new-file-btn" onClick={handleNewFile}>
                    <svg viewBox="0 0 24 24" width="18" height="18">
                      <line x1="12" y1="5" x2="12" y2="19" />
                      <line x1="5" y1="12" x2="19" y2="12" />
                    </svg>
                    New Transcription
                  </button>
                </section>
              )}
            </Show>

            <Show when={annotatedResult()} keyed>
              {(res) => (
                <section class="results-section">
                  <div class="result-meta">
                    <div class="meta-item">
                      <span class="meta-label">Language</span>
                      <span class="meta-value">{res.language || "Unknown"}</span>
                    </div>
                    <div class="meta-item">
                      <span class="meta-label">Speakers</span>
                      <span class="meta-value">{res.num_speakers}</span>
                    </div>
                    <div class="meta-item">
                      <span class="meta-label">Duration</span>
                      <span class="meta-value">{formatTime(res.audio_duration, false)}</span>
                    </div>
                    <div class="meta-item">
                      <span class="meta-label">Processing</span>
                      <span class="meta-value">{res.total_inference_time.toFixed(1)}s</span>
                    </div>
                  </div>

                  <div class="speaker-editor-box">
                    <h3>Speaker Names</h3>
                    <For each={speakerIds()}>
                      {(id) => (
                        <label class="speaker-name-row">
                          <span>Speaker {id}</span>
                          <input
                            value={speakerNames()[String(id)] || `Speaker ${id}`}
                            onInput={(e) => updateSpeakerName(id, e.currentTarget.value)}
                          />
                        </label>
                      )}
                    </For>
                  </div>

                  <Show
                    when={annotatedSegmentsWithNames().length > 0}
                    fallback={<p class="no-segments-msg">No annotated segments to display.</p>}
                  >
                    <div class="transcript-box">
                      <div class="transcript-header">
                        <h3>Annotated Preview</h3>
                      </div>
                      <div class="segments-content speaker-segments-content">
                        <For each={annotatedSegmentsWithNames()}>
                          {(segment) => (
                            <div class="segment-row">
                              <span class="segment-time">{formatTimestamp(segment.start)}</span>
                              <span class="segment-speaker">[{segment.speaker_name}]</span>
                              <span class="segment-text">{segment.text}</span>
                            </div>
                          )}
                        </For>
                      </div>
                      <div class="transcript-actions">
                        <div class="action-group">
                          <span class="action-label">Copy:</span>
                          <button class="action-btn" onClick={handleCopyAnnotatedText} title="Copy annotated text">
                            <svg viewBox="0 0 24 24" width="16" height="16">
                              <rect x="9" y="9" width="13" height="13" rx="2" ry="2" />
                              <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1" />
                            </svg>
                            <span>Text</span>
                          </button>
                          <button class="action-btn" onClick={handleCopySrt} title="Copy speaker-labeled SRT">
                            <svg viewBox="0 0 24 24" width="16" height="16">
                              <rect x="9" y="9" width="13" height="13" rx="2" ry="2" />
                              <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1" />
                            </svg>
                            <span>SRT</span>
                          </button>
                        </div>
                        <div class="action-group">
                          <span class="action-label">Download:</span>
                          <button class="action-btn" onClick={handleExportAnnotatedText} title="Download annotated .txt file">
                            <svg viewBox="0 0 24 24" width="16" height="16">
                              <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
                              <polyline points="7 10 12 15 17 10" />
                              <line x1="12" y1="15" x2="12" y2="3" />
                            </svg>
                            <span>.txt</span>
                          </button>
                          <button class="action-btn" onClick={handleExportSrt} title="Download speaker-labeled .srt file">
                            <svg viewBox="0 0 24 24" width="16" height="16">
                              <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
                              <polyline points="7 10 12 15 17 10" />
                              <line x1="12" y1="15" x2="12" y2="3" />
                            </svg>
                            <span>.srt</span>
                          </button>
                        </div>
                      </div>
                    </div>
                  </Show>

                  <button class="new-file-btn" onClick={handleNewFile}>
                    <svg viewBox="0 0 24 24" width="18" height="18">
                      <line x1="12" y1="5" x2="12" y2="19" />
                      <line x1="5" y1="12" x2="19" y2="12" />
                    </svg>
                    New Annotation
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
        title="Stop Processing?"
        message="An operation is currently in progress. Stopping will discard any partial results."
        confirmLabel="Stop & Go Back"
        cancelLabel="Keep Working"
        onConfirm={handleConfirmCancel}
        onCancel={() => setShowCancelDialog(false)}
      />
    </>
  );
};

export default Transcription;
