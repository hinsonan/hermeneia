import { Component, For, Show, createEffect, createMemo, createSignal, onCleanup, onMount } from "solid-js";
import { useNavigate } from "@solidjs/router";
import { invoke } from "@tauri-apps/api/core";
import { listen, UnlistenFn } from "@tauri-apps/api/event";
import { open, save } from "@tauri-apps/plugin-dialog";
import { useTheme } from "../utils/theme";
import { formatTime } from "../utils/timeFormat";
import FileUploader from "../components/FileUploader";
import GreekScrollLoader from "../components/GreekScrollLoader";
import TranscriptionProgressBar from "../components/TranscriptionProgressBar";
import InfoIcon from "../components/InfoIcon";
import ConfirmDialog from "../components/ConfirmDialog";
import type {
  AnnotatedResult,
  AnnotationProgress,
  LanguageOption,
  ModelOption,
  ModelValidation,
  SpeakerDevice,
  SpeakerModelKey,
  SpeakerModelRequirement,
  SystemCapabilities,
  TranscriptResult,
  TranscriptionProgress,
  TranscriptionTask,
  WhisperModel,
} from "../types/transcription";
import "./Transcription.css";

type JobProgress = TranscriptionProgress | AnnotationProgress;
type JobStatus = "queued" | "running" | "completed" | "failed" | "cancelled";
type JobMode = "transcribe" | "annotate";

interface JobSettings {
  mode: JobMode;
  model: WhisperModel;
  task: TranscriptionTask;
  language: string | null;
  includeTimestamps: boolean;
  speakerModel: SpeakerModelKey;
  speakerDevice: SpeakerDevice;
  numSpeakers: number | null;
  diarizeThreshold: number;
}

interface QueueJob {
  id: string;
  batchId: string;
  createdAt: number;
  filePath: string;
  fileName: string;
  status: JobStatus;
  settings: JobSettings;
  progress: JobProgress | null;
  result: TranscriptResult | null;
  annotatedResult: AnnotatedResult | null;
  speakerNames: Record<string, string>;
  error: string | null;
}

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

const getBaseName = (path: string): string => path.split("/").pop() || path.split("\\").pop() || path;

const makeId = (): string => {
  if (typeof crypto !== "undefined" && typeof crypto.randomUUID === "function") {
    return crypto.randomUUID();
  }
  return `${Date.now()}-${Math.random().toString(16).slice(2)}`;
};

const Transcription: Component = () => {
  const navigate = useNavigate();
  const { toggleTheme } = useTheme();

  const [mode, setMode] = createSignal<JobMode>("transcribe");
  const [selectedModel, setSelectedModel] = createSignal<WhisperModel>("tiny");
  const [selectedTask, setSelectedTask] = createSignal<TranscriptionTask>("transcribe");
  const [selectedLanguage, setSelectedLanguage] = createSignal<string | null>(null);
  const [includeTimestamps, setIncludeTimestamps] = createSignal(true);
  const [selectedSpeakerModel, setSelectedSpeakerModel] = createSignal<SpeakerModelKey>("english");
  const [selectedSpeakerDevice, setSelectedSpeakerDevice] = createSignal<SpeakerDevice>("cpu");
  const [numSpeakers, setNumSpeakers] = createSignal<number | null>(null);
  const [diarizeThreshold, setDiarizeThreshold] = createSignal(0.5);
  const [maxConcurrency, setMaxConcurrency] = createSignal(2);

  const [queueJobs, setQueueJobs] = createSignal<QueueJob[]>([]);
  const [selectedJobId, setSelectedJobId] = createSignal<string | null>(null);
  const [schedulerTick, setSchedulerTick] = createSignal(0);
  const [queueError, setQueueError] = createSignal<string | null>(null);
  const [showCancelDialog, setShowCancelDialog] = createSignal(false);
  const [showDefaults, setShowDefaults] = createSignal(false);
  const [inspectorTab, setInspectorTab] = createSignal<"output" | "segments" | "speakers" | "export">("output");

  const [systemCapabilities, setSystemCapabilities] = createSignal<SystemCapabilities | null>(null);
  const [modelValidation, setModelValidation] = createSignal<ModelValidation | null>(null);
  const [speakerModelRequirements, setSpeakerModelRequirements] = createSignal<SpeakerModelRequirement[]>([]);

  let transcriptionProgressUnlisten: UnlistenFn | null = null;
  let annotationProgressUnlisten: UnlistenFn | null = null;
  let schedulerRunning = false;

  const isAnnotateMode = createMemo(() => mode() === "annotate");
  const isEnglishOnlyModel = createMemo(() => selectedModel().endsWith(".en"));

  const availableSpeakerDevices = createMemo(() => {
    const deviceType = systemCapabilities()?.gpu_info?.device_type;
    if (deviceType === "NvidiaCuda") return ["cuda", "cpu"] as SpeakerDevice[];
    if (deviceType === "AppleMetal") return ["coreml", "cpu"] as SpeakerDevice[];
    return ["cpu"] as SpeakerDevice[];
  });

  const availableLanguages = createMemo(() => {
    if (isEnglishOnlyModel()) {
      return LANGUAGE_OPTIONS.filter((lang) => lang.value === "en");
    }
    return LANGUAGE_OPTIONS;
  });

  const selectedJob = createMemo(() => queueJobs().find((job) => job.id === selectedJobId()) || null);
  const jobTabs = createMemo(() => queueJobs().slice().sort((a, b) => a.createdAt - b.createdAt));
  const runningJobs = createMemo(() => queueJobs().filter((job) => job.status === "running"));
  const runningCount = createMemo(() => runningJobs().length);
  const queuedCount = createMemo(() => queueJobs().filter((job) => job.status === "queued").length);
  const completedCount = createMemo(() => queueJobs().filter((job) => job.status === "completed").length);
  const failedCount = createMemo(() => queueJobs().filter((job) => job.status === "failed").length);
  const cancelledCount = createMemo(() => queueJobs().filter((job) => job.status === "cancelled").length);
  const totalJobsCount = createMemo(() => queueJobs().length);

  const statusLabel = (status: JobStatus): string => {
    if (status === "queued") return "Queued";
    if (status === "running") return "Running";
    if (status === "completed") return "Completed";
    if (status === "failed") return "Failed";
    return "Cancelled";
  };

  const queueSummary = createMemo(() => [
    { key: "total", label: "Total", value: totalJobsCount() },
    { key: "queued", label: "Queued", value: queuedCount() },
    { key: "running", label: "Running", value: runningCount() },
    { key: "completed", label: "Completed", value: completedCount() },
    { key: "failed", label: "Failed", value: failedCount() },
    { key: "cancelled", label: "Cancelled", value: cancelledCount() },
  ]);

  const getAnnotatedSegmentsWithNames = (job: QueueJob): AnnotatedResult["segments"] => {
    if (!job.annotatedResult) return [];
    return job.annotatedResult.segments.map((seg) => ({
      ...seg,
      speaker_name: job.speakerNames[String(seg.speaker)] || seg.speaker_name || `Speaker ${seg.speaker}`,
    }));
  };

  const getSpeakerIdsForJob = (job: QueueJob): number[] => {
    const ids = Array.from(new Set(getAnnotatedSegmentsWithNames(job).map((s) => s.speaker)));
    return ids.sort((a, b) => a - b);
  };

  const selectedAnnotatedSegmentsWithNames = createMemo(() => {
    const job = selectedJob();
    if (!job) return [];
    return getAnnotatedSegmentsWithNames(job);
  });

  const canShowSegmentsTab = createMemo(() => {
    const job = selectedJob();
    if (!job || job.status !== "completed") return false;
    if (job.settings.mode === "annotate") return getAnnotatedSegmentsWithNames(job).length > 0;
    return Boolean(job.settings.includeTimestamps && job.result?.segments.length);
  });

  createEffect(() => {
    if (isEnglishOnlyModel()) {
      setSelectedLanguage("en");
    }
  });

  createEffect(() => {
    if (isAnnotateMode()) {
      setIncludeTimestamps(true);
    }
  });

  createEffect(() => {
    const caps = systemCapabilities();
    const model = selectedModel();
    if (caps) {
      void validateModel(model);
    }
  });

  createEffect(() => {
    maxConcurrency();
    schedulerTick();
    void scheduleQueue();
  });

  createEffect(() => {
    const jobs = queueJobs();
    const idSet = new Set(jobs.map((job) => job.id));
    const activeSelected = selectedJobId();
    if (activeSelected && !idSet.has(activeSelected)) {
      setSelectedJobId(jobs[0]?.id || null);
    }
  });

  createEffect(() => {
    const job = selectedJob();
    const tab = inspectorTab();

    if (!job || job.status !== "completed") {
      if (tab !== "output") {
        setInspectorTab("output");
      }
      return;
    }

    if (tab === "segments" && !canShowSegmentsTab()) {
      setInspectorTab("output");
      return;
    }

    if (tab === "speakers" && job.settings.mode !== "annotate") {
      setInspectorTab("output");
    }
  });

  onMount(async () => {
    transcriptionProgressUnlisten = await listen<TranscriptionProgress>("transcription-progress", (event) => {
      const payload = event.payload;
      if (!payload?.job_id) {
        return;
      }
      updateJob(payload.job_id, (job) => ({ ...job, progress: payload }));
    });

    annotationProgressUnlisten = await listen<AnnotationProgress>("annotation-progress", (event) => {
      const payload = event.payload;
      if (!payload?.job_id) {
        return;
      }
      updateJob(payload.job_id, (job) => ({ ...job, progress: payload }));
    });

    try {
      const caps = await invoke<SystemCapabilities>("get_system_capabilities");
      setSystemCapabilities(caps);
      const deviceType = caps.gpu_info?.device_type;
      if (deviceType === "NvidiaCuda") {
        setSelectedSpeakerDevice("cuda");
      } else if (deviceType === "AppleMetal") {
        setSelectedSpeakerDevice("coreml");
      }
    } catch (err) {
      console.warn("Failed to get system capabilities:", err);
    }

    try {
      const requirements = await invoke<SpeakerModelRequirement[]>("list_speaker_model_requirements");
      setSpeakerModelRequirements(requirements);
      if (requirements.length > 0 && !requirements.find((r) => r.key === selectedSpeakerModel())) {
        setSelectedSpeakerModel(requirements[0].key);
      }
    } catch (err) {
      console.warn("Failed to load speaker model requirements:", err);
    }
  });

  onCleanup(() => {
    if (transcriptionProgressUnlisten) transcriptionProgressUnlisten();
    if (annotationProgressUnlisten) annotationProgressUnlisten();
  });

  const validateModel = async (model: WhisperModel) => {
    try {
      const validation = await invoke<ModelValidation>("validate_model_selection", {
        model,
        forceCpu: false,
      });
      setModelValidation(validation);
    } catch (err) {
      console.warn("Validation failed:", err);
    }
  };

  const buildSettings = (): JobSettings => ({
    mode: mode(),
    model: selectedModel(),
    task: selectedTask(),
    language: selectedLanguage(),
    includeTimestamps: includeTimestamps(),
    speakerModel: selectedSpeakerModel(),
    speakerDevice: selectedSpeakerDevice(),
    numSpeakers: numSpeakers(),
    diarizeThreshold: diarizeThreshold(),
  });

  const updateJob = (jobId: string, updater: (job: QueueJob) => QueueJob) => {
    let changed = false;
    setQueueJobs((prev) =>
      prev.map((job) => {
        if (job.id !== jobId) return job;
        changed = true;
        return updater(job);
      })
    );
    if (changed) {
      setSchedulerTick((value) => value + 1);
    }
  };

  const enqueueFiles = (paths: string[]) => {
    const normalized = paths.filter(Boolean);
    if (normalized.length === 0) return;

    const batchId = makeId();
    const settings = buildSettings();
    const createdAtBase = Date.now();
    const newJobs = normalized.map((path, index) => ({
      id: makeId(),
      batchId,
      createdAt: createdAtBase + index,
      filePath: path,
      fileName: getBaseName(path),
      status: "queued" as JobStatus,
      settings,
      progress: null,
      result: null,
      annotatedResult: null,
      speakerNames: {},
      error: null,
    }));

    setQueueJobs((prev) => [...prev, ...newJobs]);
    if (!selectedJobId()) {
      setSelectedJobId(newJobs[0].id);
    }
    setQueueError(null);
    setSchedulerTick((value) => value + 1);
  };

  const scheduleQueue = async () => {
    if (schedulerRunning) return;
    schedulerRunning = true;
    try {
      while (true) {
        const jobs = queueJobs();
        const openSlots = maxConcurrency() - jobs.filter((job) => job.status === "running").length;
        if (openSlots <= 0) break;

        const nextJobs = jobs.filter((job) => job.status === "queued").slice(0, openSlots);
        if (nextJobs.length === 0) break;

        nextJobs.forEach((job) => {
          void runJob(job.id);
        });
      }
    } finally {
      schedulerRunning = false;
    }
  };

  const runJob = async (jobId: string) => {
    const job = queueJobs().find((entry) => entry.id === jobId);
    if (!job || job.status !== "queued") return;

    const initialMessage = job.settings.mode === "annotate" ? "Starting annotation..." : "Starting transcription...";

    updateJob(jobId, (current) => ({
      ...current,
      status: "running",
      error: null,
      result: null,
      annotatedResult: null,
      progress: {
        job_id: current.id,
        phase: job.settings.mode === "annotate" ? "starting" : "loading_model",
        current: null,
        total: null,
        message: initialMessage,
        ...(job.settings.mode === "annotate" ? { indeterminate: true } : {}),
      } as JobProgress,
    }));

    try {
      if (job.settings.mode === "annotate") {
        const annotated = await invoke<AnnotatedResult>("annotate_audio_file", {
          filePath: job.filePath,
          transcribeModel: job.settings.model,
          speakerModel: job.settings.speakerModel,
          task: job.settings.task,
          language: job.settings.language,
          timestamps: true,
          numSpeakers: job.settings.numSpeakers,
          threshold: job.settings.diarizeThreshold,
          device: job.settings.speakerDevice,
          speakerNames: job.speakerNames,
          jobId: job.id,
          batchId: job.batchId,
        });

        updateJob(jobId, (current) => ({
          ...current,
          status: "completed",
          annotatedResult: annotated,
          speakerNames: annotated.speaker_names || current.speakerNames,
          progress: {
            job_id: current.id,
            phase: "completed",
            current: 1,
            total: 1,
            message: "Annotation complete",
            indeterminate: false,
          },
        }));
      } else {
        const transcript = await invoke<TranscriptResult>("transcribe_audio_file", {
          filePath: job.filePath,
          model: job.settings.model,
          task: job.settings.task,
          language: job.settings.language,
          timestamps: job.settings.includeTimestamps,
          jobId: job.id,
          batchId: job.batchId,
        });

        updateJob(jobId, (current) => ({
          ...current,
          status: "completed",
          result: transcript,
          progress: {
            job_id: current.id,
            phase: "completed",
            current: 1,
            total: 1,
            message: "Transcription complete",
          },
        }));
      }
    } catch (err) {
      const errStr = String(err);
      if (errStr.includes("cancelled") || errStr.includes("Operation cancelled")) {
        updateJob(jobId, (current) => ({
          ...current,
          status: "cancelled",
          error: null,
          progress: null,
        }));
      } else {
        updateJob(jobId, (current) => ({
          ...current,
          status: "failed",
          error: errStr,
        }));
      }
    } finally {
      setSchedulerTick((value) => value + 1);
    }
  };

  const cancelJob = async (jobId: string) => {
    updateJob(jobId, (job) => ({
      ...job,
      status: job.status === "completed" ? "completed" : "cancelled",
      progress: job.status === "running" ? null : job.progress,
    }));

    try {
      await invoke("cancel_job", { jobId });
    } catch {
      await invoke("cancel_inference").catch(() => {});
    }
  };

  const cancelRunningJobs = async () => {
    const running = queueJobs().filter((job) => job.status === "running");
    await Promise.allSettled(running.map((job) => cancelJob(job.id)));
  };

  const retryFailedJobs = () => {
    setQueueJobs((prev) =>
      prev.map((job) => {
        if (job.status !== "failed") return job;
        return {
          ...job,
          status: "queued",
          error: null,
          progress: null,
          result: null,
          annotatedResult: null,
        };
      })
    );
    setSchedulerTick((value) => value + 1);
  };

  const clearCompleted = () => {
    const next = queueJobs().filter((job) => job.status !== "completed");
    setQueueJobs(next);
    if (!next.find((job) => job.id === selectedJobId())) {
      setSelectedJobId(next[0]?.id || null);
    }
    setSchedulerTick((value) => value + 1);
  };

  const removeJob = (jobId: string) => {
    const target = queueJobs().find((job) => job.id === jobId);
    if (!target || target.status === "running") return;
    const next = queueJobs().filter((job) => job.id !== jobId);
    setQueueJobs(next);
    if (selectedJobId() === jobId) {
      setSelectedJobId(next[0]?.id || null);
    }
    setSchedulerTick((value) => value + 1);
  };

  const setJobToQueued = (jobId: string) => {
    updateJob(jobId, (current) => ({
      ...current,
      status: "queued",
      error: null,
      progress: null,
      result: null,
      annotatedResult: null,
    }));
    setSchedulerTick((value) => value + 1);
  };

  const formatTimestamp = (seconds: number | null): string => {
    if (seconds === null) return "--:--";
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    const ms = Math.floor((seconds % 1) * 100);
    return `${mins.toString().padStart(2, "0")}:${secs.toString().padStart(2, "0")}.${ms.toString().padStart(2, "0")}`;
  };

  const formatSrtTimestamp = (seconds: number | null): string => {
    if (seconds === null) return "00:00:00,000";
    const hours = Math.floor(seconds / 3600);
    const mins = Math.floor((seconds % 3600) / 60);
    const secs = Math.floor(seconds % 60);
    const ms = Math.floor((seconds % 1) * 1000);
    return `${hours.toString().padStart(2, "0")}:${mins.toString().padStart(2, "0")}:${secs.toString().padStart(2, "0")},${ms.toString().padStart(3, "0")}`;
  };

  const getProgressPercent = (progress: JobProgress | null): number | null => {
    if (!progress || progress.current === null || progress.total === null || progress.total === 0) {
      return null;
    }
    return Math.min(100, Math.round((progress.current / progress.total) * 100));
  };

  const getPlainTextContent = (job: QueueJob): string => {
    if (job.settings.mode === "annotate") {
      return getAnnotatedSegmentsWithNames(job)
        .map((seg) => {
          const startMin = Math.floor(seg.start / 60).toString().padStart(2, "0");
          const startSec = Math.floor(seg.start % 60).toString().padStart(2, "0");
          return `[${startMin}:${startSec}] ${seg.speaker_name}: ${seg.text}`;
        })
        .join("\n");
    }
    return job.result?.text || "";
  };

  const getSrtContent = (job: QueueJob): string => {
    if (job.settings.mode === "annotate") {
      const segments = getAnnotatedSegmentsWithNames(job);
      if (segments.length === 0) return "";
      return segments
        .map((seg, index) => {
          const startTime = formatSrtTimestamp(seg.start);
          const endTime = formatSrtTimestamp(seg.end);
          return `${index + 1}\n${startTime} --> ${endTime}\n[${seg.speaker_name}] ${seg.text.trim()}\n`;
        })
        .join("\n");
    }

    if (!job.result?.segments.length) return "";
    return job.result.segments
      .map((seg, index) => {
        const startTime = formatSrtTimestamp(seg.start);
        const endTime = formatSrtTimestamp(seg.end);
        return `${index + 1}\n${startTime} --> ${endTime}\n${seg.text.trim()}\n`;
      })
      .join("\n");
  };

  const copyToClipboard = async (text: string) => {
    if (!text) return;
    try {
      await navigator.clipboard.writeText(text);
    } catch (err) {
      console.error("Failed to copy to clipboard:", err);
    }
  };

  const exportToFile = async (defaultPath: string, extensions: string[], content: string) => {
    if (!content) return;
    try {
      const outputPath = await save({
        filters: [{ name: "Export", extensions }],
        defaultPath,
      });
      if (!outputPath) return;
      await invoke("write_text_file", { path: outputPath, content });
    } catch (err) {
      console.error("Failed to export file:", err);
    }
  };

  const updateSelectedSpeakerName = (speakerId: number, value: string) => {
    const job = selectedJob();
    if (!job) return;
    updateJob(job.id, (current) => ({
      ...current,
      speakerNames: {
        ...current.speakerNames,
        [String(speakerId)]: value,
      },
    }));
  };

  const openAddFilesDialog = async () => {
    try {
      const selected = await open({
        multiple: true,
        filters: [{
          name: "Audio Files",
          extensions: ["mp3", "wav", "flac", "m4a", "ogg"],
        }],
      });

      if (!selected) return;
      if (Array.isArray(selected)) {
        enqueueFiles(selected);
        return;
      }

      enqueueFiles([selected]);
    } catch (err) {
      console.error("Failed to add files:", err);
      setQueueError("Could not open file picker. Please try again.");
    }
  };

  const exportCompletedJobsBundle = async () => {
    const completed = queueJobs().filter((job) => job.status === "completed");
    if (completed.length === 0) return;

    const lines: string[] = [];
    completed.forEach((job, index) => {
      const divider = "=".repeat(80);
      lines.push(divider);
      lines.push(`${index + 1}. ${job.fileName}`);
      lines.push(`Mode: ${job.settings.mode === "annotate" ? "Annotate" : "Transcribe"}`);
      lines.push(`Model: ${job.settings.model}`);
      lines.push("");
      lines.push(getPlainTextContent(job).trim());
      lines.push("");
    });

    const bundleContent = lines.join("\n");
    const defaultPath = `hermeneia_batch_${new Date().toISOString().slice(0, 10)}.txt`;
    await exportToFile(defaultPath, ["txt"], bundleContent);
  };

  const handleBack = () => {
    if (runningCount() > 0) {
      setShowCancelDialog(true);
      return;
    }
    navigate("/");
  };

  const handleConfirmBack = async () => {
    setShowCancelDialog(false);
    await cancelRunningJobs();
    navigate("/");
  };

  return (
    <>
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

      <div class="scroll-container transcription-scroll-container">
        <div class="scroll-rod"></div>

        <main class="parchment transcription-workbench parchment-batch-layout">
          <header class="page-header">
            <button class="back-button" onClick={handleBack}>
              <svg viewBox="0 0 24 24" width="20" height="20">
                <path d="M19 12H5M12 19l-7-7 7-7" />
              </svg>
              <span>Home</span>
            </button>

            <div class="page-heading-group">
              <h1>Transcription</h1>
              <p class="page-subtitle">Single-page flow with batch tabs. Keep context, switch jobs instantly.</p>
            </div>
          </header>

          <section class="batch-command-bar job-tabs-panel">
            <div class="job-tabs-header">
              <div class="command-title-block">
                <span class="command-kicker">Batch Controls</span>
                <h2>Job Tabs</h2>
              </div>

              <div class="command-field narrow compact-inline">
                <label for="quick-concurrency">Workers</label>
                <div class="select-wrapper compact">
                  <select
                    id="quick-concurrency"
                    value={String(maxConcurrency())}
                    onChange={(e) => setMaxConcurrency(Math.max(1, Math.min(4, parseInt(e.currentTarget.value, 10) || 1)))}
                  >
                    <For each={[1, 2, 3, 4]}>{(value) => <option value={value}>{value}</option>}</For>
                  </select>
                  <svg class="select-arrow" viewBox="0 0 24 24">
                    <path d="M6 9l6 6 6-6" />
                  </svg>
                </div>
              </div>

              <div class="command-actions-cluster">
                <button class="primary-command-btn" onClick={() => void openAddFilesDialog()}>Add Files</button>
                <button class="change-file-btn" onClick={() => setShowDefaults(!showDefaults())}>{showDefaults() ? "Hide Defaults" : "Show Defaults"}</button>
                <button class="change-file-btn" onClick={() => void cancelRunningJobs()} disabled={runningCount() === 0}>Cancel Running</button>
                <button class="change-file-btn" onClick={retryFailedJobs} disabled={failedCount() === 0}>Retry Failed</button>
                <button class="change-file-btn" onClick={clearCompleted} disabled={completedCount() === 0}>Clear Completed</button>
                <button class="change-file-btn" onClick={() => void exportCompletedJobsBundle()} disabled={completedCount() === 0}>Export Completed</button>
              </div>
            </div>

            <div class="queue-summary queue-summary-refined">
              <For each={queueSummary()}>
                {(chip) => (
                  <span class={`summary-chip ${chip.key}`}>
                    <span class="summary-label">{chip.label}</span>
                    <span class="summary-value">{chip.value}</span>
                  </span>
                )}
              </For>
            </div>

            <Show when={jobTabs().length > 0} fallback={<p class="queue-empty">Add audio files to create your first job tab.</p>}>
              <div class="job-tabs-strip">
                <For each={jobTabs()}>
                  {(job) => (
                    <div class={`job-tab ${selectedJobId() === job.id ? "active" : ""} ${job.status}`}>
                      <button class="job-tab-main" onClick={() => setSelectedJobId(job.id)}>
                        <span class="job-tab-name" title={job.fileName}>{job.fileName}</span>
                        <span class={`status-badge ${job.status}`}>{statusLabel(job.status)}</span>
                        <Show when={job.status === "running"}>
                          <span class="job-tab-progress">
                            {getProgressPercent(job.progress) === null ? "..." : `${getProgressPercent(job.progress)}%`}
                          </span>
                        </Show>
                      </button>

                      <button
                        class="job-tab-remove"
                        onClick={() => removeJob(job.id)}
                        disabled={job.status === "running"}
                        aria-label={`Remove ${job.fileName}`}
                        title={job.status === "running" ? "Cancel this job before removing" : "Remove from queue"}
                      >
                        ×
                      </button>
                    </div>
                  )}
                </For>
              </div>
            </Show>
          </section>

          <section class="upload-section compact-upload-section">
            <div class="quick-add-card">
              <div class="quick-add-copy">
                <span class="section-eyebrow">Ingress</span>
                <h2>Add More Audio</h2>
                <p>Drop recordings anytime; they appear as tabs and process in queue order.</p>
              </div>
              <div class="quick-add-uploader-shell">
                <FileUploader multiple onFilesSelected={enqueueFiles} />
              </div>
            </div>
          </section>

          <Show when={queueError()}>
            <div class="error-banner">
              <div class="error-content">
                <span>{queueError()}</span>
              </div>
              <button onClick={() => setQueueError(null)}>Dismiss</button>
            </div>
          </Show>

          <Show when={showDefaults()}>
            <section class="settings-panel queue-settings defaults-panel">
              <div class="defaults-header">
                <div>
                  <span class="section-eyebrow">Defaults</span>
                  <h2>Settings For Newly Added Jobs</h2>
                </div>
                <span class="setting-hint">Existing queued items keep the settings they were created with.</span>
              </div>

              <div class="defaults-grid refined-defaults-grid">
                <div class="setting-group feature-setting-group feature-setting-group-wide">
                  <label class="label-with-info">
                    Mode
                    <InfoIcon
                      content="Transcription returns transcript text output. Annotate runs speaker diarization + transcription and outputs speaker-labeled views."
                      position="right"
                    />
                  </label>
                  <div class="task-toggle">
                    <button class={`task-btn ${!isAnnotateMode() ? "active" : ""}`} onClick={() => setMode("transcribe")}>Transcribe</button>
                    <button class={`task-btn ${isAnnotateMode() ? "active" : ""}`} onClick={() => setMode("annotate")}>Annotate</button>
                  </div>
                </div>

                <div class="setting-group feature-setting-group">
                  <label for="model-select" class="label-with-info">
                    Model
                    <InfoIcon
                      content="Smaller models are faster with lower memory use. Larger models are more accurate but require more resources."
                      position="right"
                    />
                  </label>
                  <div class="select-wrapper">
                    <select id="model-select" value={selectedModel()} onChange={(e) => setSelectedModel(e.currentTarget.value as WhisperModel)}>
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

                <div class="setting-group feature-setting-group">
                  <label class="label-with-info">Task</label>
                  <div class="task-toggle">
                    <button class={`task-btn ${selectedTask() === "transcribe" ? "active" : ""}`} onClick={() => setSelectedTask("transcribe")}>Transcribe</button>
                    <button class={`task-btn ${selectedTask() === "translate" ? "active" : ""}`} onClick={() => setSelectedTask("translate")}>Translate to English</button>
                  </div>
                </div>

                <div class="setting-group feature-setting-group">
                  <label for="language-select">Source Language</label>
                  <div class="select-wrapper">
                    <select
                      id="language-select"
                      value={selectedLanguage() || ""}
                      onChange={(e) => setSelectedLanguage(e.currentTarget.value || null)}
                      disabled={isEnglishOnlyModel()}
                    >
                      <For each={availableLanguages()}>{(option) => <option value={option.value || ""}>{option.label}</option>}</For>
                    </select>
                    <svg class="select-arrow" viewBox="0 0 24 24">
                      <path d="M6 9l6 6 6-6" />
                    </svg>
                  </div>
                </div>

                <div class="setting-group inline feature-setting-group toggle-setting-group">
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
                    <span>Include timestamps</span>
                  </label>
                </div>

                <Show when={isAnnotateMode()}>
                  <>
                    <div class="setting-group feature-setting-group">
                      <label for="speaker-model-select">Speaker Model</label>
                      <div class="select-wrapper">
                        <select
                          id="speaker-model-select"
                          value={selectedSpeakerModel()}
                          onChange={(e) => setSelectedSpeakerModel(e.currentTarget.value as SpeakerModelKey)}
                        >
                          <For each={speakerModelRequirements()}>
                            {(m) => (
                              <option value={m.key}>
                                {m.display_name} ({m.approx_size_mb.toFixed(1)} MB)
                              </option>
                            )}
                          </For>
                        </select>
                        <svg class="select-arrow" viewBox="0 0 24 24">
                          <path d="M6 9l6 6 6-6" />
                        </svg>
                      </div>
                    </div>

                    <div class="setting-group feature-setting-group">
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
                    </div>

                    <div class="setting-group feature-setting-group">
                      <label for="num-speakers-input">Expected Speakers</label>
                      <input
                        id="num-speakers-input"
                        type="number"
                        min="1"
                        max="20"
                        placeholder="Auto-detect"
                        value={numSpeakers() ?? ""}
                        onInput={(e) => {
                          const value = e.currentTarget.value;
                          setNumSpeakers(value === "" ? null : parseInt(value, 10));
                        }}
                        class="number-input"
                      />
                    </div>

                    <div class="setting-group feature-setting-group">
                      <label for="diarize-threshold-input">
                        Clustering Threshold <span class="threshold-value">({diarizeThreshold().toFixed(2)})</span>
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
                    </div>
                  </>
                </Show>
              </div>
            </section>
          </Show>

          <Show
            when={selectedJob()}
            fallback={
              <section class="queue-card inspector-pane single-job-panel">
                <div class="inspector-empty-state">
                  <span class="section-eyebrow">Ready</span>
                  <h2>Select Or Add A Job</h2>
                  <p>Add audio files above. Each one appears as a tab so you can switch context without losing the single-job workflow.</p>
                </div>
              </section>
            }
          >
            {(job) => (
              <section class="queue-card inspector-pane single-job-panel">
                <div class="single-job-header">
                  <div>
                    <span class="section-eyebrow">Current Job</span>
                    <h2 class="inspector-title">{job().fileName}</h2>
                    <div class="queue-meta refined-inspector-meta">
                      <span>{job().settings.mode === "annotate" ? "Annotate" : "Transcribe"}</span>
                      <span>{job().settings.model}</span>
                      <Show when={job().settings.language}>
                        <span>{job().settings.language}</span>
                      </Show>
                      <span class={`status-badge ${job().status}`}>{statusLabel(job().status)}</span>
                    </div>
                  </div>

                  <div class="single-job-actions">
                    <Show when={job().status === "running"}>
                      <button class="change-file-btn" onClick={() => void cancelJob(job().id)}>Cancel</button>
                    </Show>
                    <Show when={job().status === "failed" || job().status === "cancelled"}>
                      <button class="change-file-btn" onClick={() => setJobToQueued(job().id)}>Retry</button>
                    </Show>
                    <button class="change-file-btn" onClick={() => removeJob(job().id)} disabled={job().status === "running"}>Remove</button>
                  </div>
                </div>

                <Show when={job().status === "running"}>
                  <section class="processing-section transcription-processing">
                    <GreekScrollLoader />
                    <h2>{job().settings.mode === "annotate" ? "Annotating..." : "Transcribing..."}</h2>
                    <p>{job().progress?.message || "Preparing audio"}</p>
                    <TranscriptionProgressBar progress={job().progress} />
                    <div class="processing-details">
                      <span>Mode: {job().settings.mode === "annotate" ? "Annotate" : "Transcribe"}</span>
                      <span>Model: {job().settings.model}</span>
                    </div>
                  </section>
                </Show>

                <Show when={job().status === "queued"}>
                  <div class="queue-idle-state">
                    <h3>Queued</h3>
                    <p>This job is waiting for an available worker. Keep this tab open or switch to another job.</p>
                  </div>
                </Show>

                <Show when={job().status === "failed" && job().error}>
                  <div class="row-error inspector-error">{job().error}</div>
                </Show>

                <Show when={job().status === "cancelled"}>
                  <div class="queue-idle-state">
                    <h3>Cancelled</h3>
                    <p>This job was cancelled before completion. Use Retry to queue it again.</p>
                  </div>
                </Show>

                <Show when={job().status === "completed"}>
                  <div class="result-meta refined-result-meta">
                    <Show
                      when={job().settings.mode === "transcribe" && job().result}
                      fallback={
                        <>
                          <div class="meta-item">
                            <span class="meta-label">Language</span>
                            <span class="meta-value">{job().annotatedResult?.language || "Unknown"}</span>
                          </div>
                          <div class="meta-item">
                            <span class="meta-label">Speakers</span>
                            <span class="meta-value">{job().annotatedResult?.num_speakers ?? 0}</span>
                          </div>
                          <div class="meta-item">
                            <span class="meta-label">Duration</span>
                            <span class="meta-value">{formatTime(job().annotatedResult?.audio_duration || 0, false)}</span>
                          </div>
                          <div class="meta-item">
                            <span class="meta-label">Processing</span>
                            <span class="meta-value">{(job().annotatedResult?.total_inference_time || 0).toFixed(1)}s</span>
                          </div>
                        </>
                      }
                    >
                      {(res) => (
                        <>
                          <div class="meta-item">
                            <span class="meta-label">Language</span>
                            <span class="meta-value">{res().language || "Unknown"}</span>
                          </div>
                          <div class="meta-item">
                            <span class="meta-label">Duration</span>
                            <span class="meta-value">{formatTime(res().duration, false)}</span>
                          </div>
                          <div class="meta-item">
                            <span class="meta-label">Processing</span>
                            <span class="meta-value">{res().inference_time.toFixed(1)}s</span>
                          </div>
                          <div class="meta-item">
                            <span class="meta-label">Model</span>
                            <span class="meta-value">{res().model}</span>
                          </div>
                        </>
                      )}
                    </Show>
                  </div>

                  <div class="inspector-tabs refined-inspector-tabs">
                    <button class={`task-btn ${inspectorTab() === "output" ? "active" : ""}`} onClick={() => setInspectorTab("output")}>Output</button>
                    <Show when={canShowSegmentsTab()}>
                      <button class={`task-btn ${inspectorTab() === "segments" ? "active" : ""}`} onClick={() => setInspectorTab("segments")}>Segments</button>
                    </Show>
                    <Show when={job().settings.mode === "annotate"}>
                      <button class={`task-btn ${inspectorTab() === "speakers" ? "active" : ""}`} onClick={() => setInspectorTab("speakers")}>Speakers</button>
                    </Show>
                    <button class={`task-btn ${inspectorTab() === "export" ? "active" : ""}`} onClick={() => setInspectorTab("export")}>Export</button>
                  </div>

                  <Show when={inspectorTab() === "output"}>
                    <Show
                      when={job().settings.mode === "transcribe" && job().result}
                      fallback={
                        <div class="transcript-box refined-transcript-box">
                          <div class="transcript-header">
                            <h3>Annotated Preview</h3>
                          </div>
                          <div class="segments-content speaker-segments-content refined-segments-content">
                            <For each={selectedAnnotatedSegmentsWithNames()}>
                              {(segment) => (
                                <div class="segment-row refined-segment-row">
                                  <span class="segment-time">{formatTimestamp(segment.start)}</span>
                                  <span class="segment-speaker">[{segment.speaker_name}]</span>
                                  <span class="segment-text">{segment.text}</span>
                                </div>
                              )}
                            </For>
                          </div>
                        </div>
                      }
                    >
                      {(res) => (
                        <div class="transcript-box refined-transcript-box">
                          <div class="transcript-header">
                            <h3>Transcript</h3>
                          </div>
                          <div class="transcript-content refined-transcript-content">{res().text}</div>
                        </div>
                      )}
                    </Show>
                  </Show>

                  <Show when={inspectorTab() === "segments" && canShowSegmentsTab()}>
                    <div class="segments-box refined-segments-box">
                      <div class="transcript-header">
                        <h3>
                          {job().settings.mode === "annotate"
                            ? `Annotated Segments (${selectedAnnotatedSegmentsWithNames().length})`
                            : `Timestamps (${job().result?.segments.length || 0} segments)`}
                        </h3>
                      </div>
                      <div class="segments-content refined-segments-content">
                        <Show
                          when={job().settings.mode === "annotate"}
                          fallback={
                            <For each={job().result?.segments || []}>
                              {(segment) => (
                                <div class="segment-row refined-segment-row">
                                  <span class="segment-time">{formatTimestamp(segment.start)}</span>
                                  <span class="segment-text">{segment.text}</span>
                                </div>
                              )}
                            </For>
                          }
                        >
                          <For each={selectedAnnotatedSegmentsWithNames()}>
                            {(segment) => (
                              <div class="segment-row refined-segment-row">
                                <span class="segment-time">{formatTimestamp(segment.start)}</span>
                                <span class="segment-speaker">[{segment.speaker_name}]</span>
                                <span class="segment-text">{segment.text}</span>
                              </div>
                            )}
                          </For>
                        </Show>
                      </div>
                    </div>
                  </Show>

                  <Show when={inspectorTab() === "speakers" && job().settings.mode === "annotate"}>
                    <div class="speaker-editor-box refined-speaker-editor-box">
                      <h3>Speaker Names</h3>
                      <p class="speaker-editor-note">Rename diarization labels before exporting or copying the annotated transcript.</p>
                      <For each={getSpeakerIdsForJob(job())}>
                        {(id) => (
                          <label class="speaker-name-row">
                            <span>Speaker {id}</span>
                            <input
                              value={job().speakerNames[String(id)] || `Speaker ${id}`}
                              onInput={(e) => updateSelectedSpeakerName(id, e.currentTarget.value)}
                            />
                          </label>
                        )}
                      </For>
                    </div>
                  </Show>

                  <Show when={inspectorTab() === "export"}>
                    <div class="export-grid">
                      <div class="transcript-box refined-transcript-box">
                        <div class="transcript-header">
                          <h3>Export This Job</h3>
                        </div>
                        <div class="transcript-actions">
                          <div class="action-group">
                            <span class="action-label">Copy</span>
                            <button class="action-btn" onClick={() => void copyToClipboard(getPlainTextContent(job()))}>Text</button>
                            <button class="action-btn" onClick={() => void copyToClipboard(getSrtContent(job()))}>SRT</button>
                          </div>
                          <div class="action-group">
                            <span class="action-label">Download</span>
                            <button
                              class="action-btn"
                              onClick={() =>
                                void exportToFile(
                                  `${job().fileName.replace(/\.[^/.]+$/, "")}.txt`,
                                  ["txt"],
                                  getPlainTextContent(job())
                                )
                              }
                            >
                              .txt
                            </button>
                            <button
                              class="action-btn"
                              onClick={() =>
                                void exportToFile(
                                  `${job().fileName.replace(/\.[^/.]+$/, "")}.srt`,
                                  ["srt"],
                                  getSrtContent(job())
                                )
                              }
                            >
                              .srt
                            </button>
                          </div>
                        </div>
                      </div>

                      <div class="transcript-box refined-transcript-box batch-export-box">
                        <div class="transcript-header">
                          <h3>Export Batch</h3>
                        </div>
                        <div class="batch-export-body">
                          <div class="batch-export-row">
                            <span>Completed queue</span>
                            <button class="action-btn" onClick={() => void exportCompletedJobsBundle()} disabled={completedCount() === 0}>
                              Export All ({completedCount()})
                            </button>
                          </div>
                        </div>
                      </div>
                    </div>
                  </Show>
                </Show>
              </section>
            )}
          </Show>
        </main>

        <div class="scroll-rod"></div>
      </div>

      <ConfirmDialog
        open={showCancelDialog()}
        title="Leave Queue?"
        message="There are active jobs. Going back will cancel running jobs."
        confirmLabel="Cancel Jobs & Go Back"
        cancelLabel="Stay Here"
        onConfirm={() => void handleConfirmBack()}
        onCancel={() => setShowCancelDialog(false)}
      />
    </>
  );
};

export default Transcription;
