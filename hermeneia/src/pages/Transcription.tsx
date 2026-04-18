import { Component, For, Show, createEffect, createMemo, createSignal } from "solid-js";
import { useNavigate } from "@solidjs/router";
import { invoke } from "@tauri-apps/api/core";
import { open, save } from "@tauri-apps/plugin-dialog";
import { useTheme } from "../utils/theme";
import { formatTime } from "../utils/timeFormat";
import GreekScrollLoader from "../components/GreekScrollLoader";
import TranscriptionProgressBar from "../components/TranscriptionProgressBar";
import InfoIcon from "../components/InfoIcon";
import ConfirmDialog from "../components/ConfirmDialog";
import FileUploader from "../components/FileUploader";
import type {
  AnnotatedResult,
  LanguageOption,
  ModelOption,
  SpeakerDevice,
  SpeakerModelKey,
  TranscriptionTask,
  WhisperModel,
} from "../types/transcription";
import {
  JobProgress,
  JobStatus,
  QueueJob,
  cancelAllRunning,
  cancelAllAndClear,
  cancelJob,
  cancellingCount,
  cancelledCount,
  clearCompleted,
  completedCount,
  dismissQueueError,
  enqueueFiles,
  failedCount,
  queueState,
  queuedCount,
  removeJob,
  retryFailedJobs,
  retryJob,
  runningCount,
  selectedJob,
  setDefault,
  setInspectorTab,
  setMaxConcurrency,
  setSelectedJob,
  sortedJobs,
  updateSpeakerName,
} from "../stores/jobQueue";
import "./Transcription.css";

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

const statusLabel = (status: JobStatus): string => {
  if (status === "queued") return "Queued";
  if (status === "running") return "Running";
  if (status === "cancelling") return "Cancelling";
  if (status === "completed") return "Done";
  if (status === "failed") return "Failed";
  return "Cancelled";
};

interface QueueStatusIconProps {
  status: JobStatus;
}

const QueueStatusIcon: Component<QueueStatusIconProps> = (props) => {
  const icon = () => {
    if (props.status === "running") {
      return (
        <svg viewBox="0 0 24 24" aria-hidden="true">
          <path d="M12 3l6 6-6 12L6 9z" />
          <path d="M12 7v10" />
          <path d="M10 13h4" />
        </svg>
      );
    }

    if (props.status === "cancelling") {
      return (
        <svg viewBox="0 0 24 24" aria-hidden="true">
          <circle cx="12" cy="12" r="8" />
          <path d="M9 9h6v6H9z" />
        </svg>
      );
    }

    if (props.status === "completed") {
      return (
        <svg viewBox="0 0 24 24" aria-hidden="true">
          <circle cx="12" cy="12" r="8" />
          <path d="M8.5 12.2l2.2 2.4 4.8-4.9" />
        </svg>
      );
    }

    if (props.status === "failed") {
      return (
        <svg viewBox="0 0 24 24" aria-hidden="true">
          <circle cx="12" cy="12" r="8" />
          <path d="M9.4 9.4l5.2 5.2" />
          <path d="M14.6 9.4l-5.2 5.2" />
        </svg>
      );
    }

    if (props.status === "cancelled") {
      return (
        <svg viewBox="0 0 24 24" aria-hidden="true">
          <circle cx="12" cy="12" r="8" />
          <path d="M8.8 15.2l6.4-6.4" />
        </svg>
      );
    }

    return (
      <svg viewBox="0 0 24 24" aria-hidden="true">
        <path d="M12 4.5L19.5 12 12 19.5 4.5 12z" />
      </svg>
    );
  };

  return (
    <span class={`tx-status-icon ${props.status}`} aria-label={statusLabel(props.status)}>
      <Show when={props.status === "running"}>
        <span class="tx-status-icon-halo" aria-hidden="true" />
      </Show>
      {icon()}
    </span>
  );
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

const QUEUE_EXPANDED_STORAGE_KEY = "hermeneia-transcription-queue-expanded";

const loadQueueExpandedPreference = (): boolean => {
  if (typeof window === "undefined") return true;
  try {
    const value = window.localStorage.getItem(QUEUE_EXPANDED_STORAGE_KEY);
    if (value === null) return true;
    return value === "1";
  } catch {
    return true;
  }
};

const sanitizeExportBaseName = (fileName: string): string => {
  const base = fileName.replace(/\.[^/.]+$/, "").trim();
  const sanitized = base.replace(/[<>:"/\\|?*\u0000-\u001f]/g, "_").replace(/\s+/g, " ").trim();
  return sanitized || "transcript";
};

const Transcription: Component = () => {
  const navigate = useNavigate();
  const { toggleTheme } = useTheme();

  const [showClearDialog, setShowClearDialog] = createSignal(false);
  const [queueExpanded, setQueueExpanded] = createSignal(loadQueueExpandedPreference());

  const defaults = () => queueState.defaults;
  const isAnnotateMode = createMemo(() => defaults().mode === "annotate");
  const isEnglishOnlyModel = createMemo(() => defaults().model.endsWith(".en"));

  const availableSpeakerDevices = createMemo(() => {
    const deviceType = queueState.systemCapabilities?.gpu_info?.device_type;
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

  const totalJobs = createMemo(() => queueState.jobs.length);
  const activeRunningCount = createMemo(() => runningCount() + cancellingCount());

  createEffect(() => {
    if (typeof window === "undefined") return;
    try {
      window.localStorage.setItem(QUEUE_EXPANDED_STORAGE_KEY, queueExpanded() ? "1" : "0");
    } catch {
      // best-effort preference persistence
    }
  });

  createEffect(() => {
    if (totalJobs() > 0 && !queueState.selectedJobId) {
      setQueueExpanded(true);
    }
  });

  const progressSegments = createMemo(() => {
    const total = totalJobs();
    if (total === 0) return null;
    return {
      completed: (completedCount() / total) * 100,
      running: (runningCount() / total) * 100,
      failed: (failedCount() / total) * 100,
      cancelled: (cancelledCount() / total) * 100,
    };
  });

  const handleLanguageChange = (value: string) => {
    setDefault("language", value || null);
  };

  const handleModelChange = (value: WhisperModel) => {
    setDefault("model", value);
    if (value.endsWith(".en")) {
      setDefault("language", "en");
    }
  };

  const openAddFilesDialog = async () => {
    try {
      const selected = await open({
        multiple: true,
        filters: [{ name: "Audio Files", extensions: ["mp3", "wav", "flac", "m4a", "ogg"] }],
      });
      if (!selected) return;
      setQueueExpanded(true);
      enqueueFiles(Array.isArray(selected) ? selected : [selected]);
    } catch (err) {
      console.error("Failed to add files:", err);
    }
  };

  const exportCompletedBundle = async () => {
    const completed = queueState.jobs.filter((job) => job.status === "completed");
    if (completed.length === 0) return;

    const nameCounts = new Map<string, number>();
    const entries: { path: string; content: string }[] = [];

    completed.forEach((job) => {
      const baseName = sanitizeExportBaseName(job.fileName);
      const seen = nameCounts.get(baseName) || 0;
      nameCounts.set(baseName, seen + 1);
      const uniqueBaseName = seen === 0 ? baseName : `${baseName}_${seen + 1}`;

      entries.push({
        path: `${uniqueBaseName}.txt`,
        content: getPlainTextContent(job),
      });
      entries.push({
        path: `${uniqueBaseName}.srt`,
        content: getSrtContent(job),
      });
    });

    try {
      const defaultPath = `hermeneia_batch_${new Date().toISOString().slice(0, 10)}.zip`;
      const outputPath = await save({
        filters: [{ name: "Zip Archive", extensions: ["zip"] }],
        defaultPath,
      });
      if (!outputPath) return;

      await invoke("write_zip_archive", { path: outputPath, entries });
    } catch (err) {
      console.error("Failed to export zip archive:", err);
    }
  };

  const handleBack = () => {
    navigate("/");
  };

  const requestClearAll = () => {
    if (totalJobs() === 0) return;
    setShowClearDialog(true);
  };

  const confirmClearAll = async () => {
    setShowClearDialog(false);
    await cancelAllAndClear();
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

      <div class="scroll-container tx-scroll-container">
        <div class="scroll-rod"></div>
        <main class="parchment tx-parchment">
        <div class="tx-shell">
        <header class="tx-header">
          <button class="tx-back" onClick={handleBack}>
            <svg viewBox="0 0 24 24" width="18" height="18">
              <path d="M19 12H5M12 19l-7-7 7-7" />
            </svg>
            <span>Home</span>
          </button>

          <div class="tx-title">
            <h1>Transcription</h1>
            <p>Queue batches, switch freely — jobs keep running in the background.</p>
          </div>

          <div class="tx-header-actions">
            <div class="tx-worker-select">
              <label for="worker-select">Jobs at once</label>
              <div class="tx-select">
                <select
                  id="worker-select"
                  value={String(queueState.maxConcurrency)}
                  onChange={(e) => setMaxConcurrency(parseInt(e.currentTarget.value, 10))}
                >
                  <For each={[1, 2, 3, 4].filter((value) => value <= queueState.maxConcurrencyLimit)}>{(value) => <option value={value}>{value}</option>}</For>
                </select>
                <svg viewBox="0 0 24 24"><path d="M6 9l6 6 6-6" /></svg>
              </div>
            </div>

            <button class="tx-btn tx-btn-primary" onClick={() => void openAddFilesDialog()}>
              <svg viewBox="0 0 24 24" width="16" height="16">
                <line x1="12" y1="5" x2="12" y2="19" />
                <line x1="5" y1="12" x2="19" y2="12" />
              </svg>
              <span>Add Files</span>
            </button>
          </div>
        </header>

        <Show when={totalJobs() > 0}>
          <div class="tx-batch-status">
            <div class="tx-batch-counts">
              <span class="tx-count-pill tx-count-total">{totalJobs()} total</span>
              <Show when={runningCount() > 0}>
                <span class="tx-count-pill tx-count-running">{runningCount()} running</span>
              </Show>
              <Show when={cancellingCount() > 0}>
                <span class="tx-count-pill tx-count-cancelled">{cancellingCount()} cancelling</span>
              </Show>
              <Show when={queuedCount() > 0}>
                <span class="tx-count-pill tx-count-queued">{queuedCount()} queued</span>
              </Show>
              <Show when={completedCount() > 0}>
                <span class="tx-count-pill tx-count-completed">{completedCount()} done</span>
              </Show>
              <Show when={failedCount() > 0}>
                <span class="tx-count-pill tx-count-failed">{failedCount()} failed</span>
              </Show>
              <Show when={cancelledCount() > 0}>
                <span class="tx-count-pill tx-count-cancelled">{cancelledCount()} cancelled</span>
              </Show>
            </div>

            <Show when={progressSegments()}>
              {(seg) => (
                <div class="tx-batch-bar" role="progressbar" aria-valuenow={completedCount()} aria-valuemax={totalJobs()}>
                  <div class="tx-batch-bar-seg tx-seg-completed" style={{ width: `${seg().completed}%` }} />
                  <div class="tx-batch-bar-seg tx-seg-running" style={{ width: `${(seg().running + (cancellingCount() / totalJobs()) * 100)}%` }} />
                  <div class="tx-batch-bar-seg tx-seg-failed" style={{ width: `${seg().failed}%` }} />
                  <div class="tx-batch-bar-seg tx-seg-cancelled" style={{ width: `${seg().cancelled}%` }} />
                </div>
              )}
            </Show>

            <div class="tx-batch-actions">
              <button class="tx-link-btn" onClick={() => void cancelAllRunning()} disabled={activeRunningCount() === 0}>
                Cancel running
              </button>
              <button class="tx-link-btn" onClick={retryFailedJobs} disabled={failedCount() === 0}>
                Retry failed
              </button>
              <button class="tx-link-btn" onClick={clearCompleted} disabled={completedCount() === 0}>
                Clear completed
              </button>
              <button class="tx-link-btn" onClick={() => void exportCompletedBundle()} disabled={completedCount() === 0}>
                Export all ({completedCount()})
              </button>
              <button class="tx-link-btn tx-link-btn-danger" onClick={requestClearAll} disabled={totalJobs() === 0}>
                Clear all
              </button>
            </div>
          </div>
        </Show>

        <Show when={queueState.queueError}>
          <div class="tx-banner tx-banner-error">
            <span>{queueState.queueError}</span>
            <button onClick={dismissQueueError}>Dismiss</button>
          </div>
        </Show>

        <div class="tx-main-layout">
          <section class="tx-workbench">
            <Show
              when={totalJobs() > 0}
              fallback={
                <section class="tx-uploader-panel">
                  <FileUploader multiple onFilesSelected={enqueueFiles} />
                </section>
              }
            >
              <>
                <section class="tx-queue-panel">
                  <header class="tx-queue-head">
                    <div class="tx-queue-head-title-wrap">
                      <h2>Queue</h2>
                      <span class="tx-queue-head-total">{totalJobs()} {totalJobs() === 1 ? "job" : "jobs"}</span>
                    </div>
                    <button
                      class="tx-queue-toggle"
                      aria-expanded={queueExpanded()}
                      aria-controls="tx-queue-list"
                      onClick={() => setQueueExpanded(!queueExpanded())}
                    >
                      <span>{queueExpanded() ? "Hide queue" : "Show queue"}</span>
                      <svg viewBox="0 0 24 24" width="14" height="14" classList={{ expanded: queueExpanded() }}>
                        <path d="M6 9l6 6 6-6" />
                      </svg>
                    </button>
                  </header>

                  <Show when={queueExpanded()}>
                    <ul id="tx-queue-list" class="tx-queue-list" role="list">
                      <For each={sortedJobs()}>
                        {(job) => (
                          <li>
                            <button
                              class={`tx-queue-row ${queueState.selectedJobId === job.id ? "selected" : ""}`}
                              onClick={() => setSelectedJob(job.id)}
                            >
                              <div class="tx-queue-row-head">
                                <QueueStatusIcon status={job.status} />
                                <span class="tx-queue-row-name" title={job.fileName}>{job.fileName}</span>
                                <span class={`tx-queue-row-state ${job.status}`}>{statusLabel(job.status)}</span>
                              </div>

                              <div class="tx-queue-row-meta">
                                <span class="tx-chip">{job.settings.mode === "annotate" ? "Annotate" : "Transcribe"}</span>
                                <span class="tx-chip tx-chip-muted">{job.settings.model}</span>
                                <Show when={job.status === "running" || job.status === "cancelling"}>
                                  <span class="tx-queue-row-progress">
                                    {getProgressPercent(job.progress) === null
                                      ? (job.progress?.message || "Preparing...")
                                      : `${getProgressPercent(job.progress)}%`}
                                  </span>
                                </Show>
                              </div>
                            </button>
                          </li>
                        )}
                      </For>
                    </ul>
                  </Show>
                </section>

                <section class="tx-inspector tx-inspector-main">
                  <Show
                    when={selectedJob()}
                    fallback={
                      <div class="tx-inspector-empty">
                        <h2>Select a job</h2>
                        <p>Choose a queue item to open its progress and transcript details.</p>
                      </div>
                    }
                  >
                    {(job) => <JobInspector job={job()} />}
                  </Show>
                </section>
              </>
            </Show>
          </section>

          <SettingsPanel
            isAnnotateMode={isAnnotateMode()}
            isEnglishOnlyModel={isEnglishOnlyModel()}
            availableSpeakerDevices={availableSpeakerDevices()}
            availableLanguages={availableLanguages()}
            onModelChange={handleModelChange}
            onLanguageChange={handleLanguageChange}
          />
        </div>
        </div>
        </main>
        <div class="scroll-rod"></div>
      </div>

      <ConfirmDialog
        open={showClearDialog()}
        title="Clear queue?"
        message="This removes every job. Running jobs will be cancelled and any unsaved transcripts will be lost."
        confirmLabel="Clear All"
        cancelLabel="Keep Jobs"
        onConfirm={() => void confirmClearAll()}
        onCancel={() => setShowClearDialog(false)}
      />
    </>
  );
};

interface JobInspectorProps {
  job: QueueJob;
}

const JobInspector: Component<JobInspectorProps> = (props) => {
  const job = () => props.job;

  const annotatedSegments = createMemo(() => getAnnotatedSegmentsWithNames(job()));
  const speakerIds = createMemo(() => getSpeakerIdsForJob(job()));
  const srtContent = createMemo(() => getSrtContent(job()));
  const textContent = createMemo(() => getPlainTextContent(job()));
  const exportBaseName = createMemo(() => sanitizeExportBaseName(job().fileName));

  const activeTab = () => {
    const j = job();
    if (j.status !== "completed") return "srt" as const;
    return j.inspectorTab;
  };

  const renderSpeakerEditor = () => (
    <aside class="tx-speaker-rail">
      <h4>Speakers</h4>
      <p class="tx-speaker-hint">Rename before exporting.</p>
      <For each={speakerIds()}>
        {(id) => (
          <label class="tx-speaker-row">
            <span>Speaker {id}</span>
            <input
              value={job().speakerNames[String(id)] || `Speaker ${id}`}
              onInput={(e) => updateSpeakerName(job().id, id, e.currentTarget.value)}
            />
          </label>
        )}
      </For>
    </aside>
  );

  return (
    <>
      <header class="tx-inspector-head">
        <div class="tx-inspector-title">
          <span class="tx-inspector-eyebrow">Selected Job</span>
          <h2 title={job().fileName}>{job().fileName}</h2>
          <div class="tx-inspector-meta">
            <span class="tx-chip">{job().settings.mode === "annotate" ? "Annotate" : "Transcribe"}</span>
            <span class="tx-chip tx-chip-muted">{job().settings.model}</span>
            <Show when={job().settings.language}>
              <span class="tx-chip tx-chip-muted">{job().settings.language}</span>
            </Show>
            <span class={`tx-status-badge ${job().status}`}>{statusLabel(job().status)}</span>
          </div>
        </div>

        <div class="tx-inspector-actions">
          <Show when={job().status === "running"}>
            <button class="tx-btn" onClick={() => void cancelJob(job().id)}>Cancel</button>
          </Show>
          <Show when={job().status === "cancelling"}>
            <button class="tx-btn" disabled>Cancelling...</button>
          </Show>
          <Show when={job().status === "failed" || job().status === "cancelled"}>
            <button class="tx-btn" onClick={() => retryJob(job().id)}>Retry</button>
          </Show>
          <button class="tx-btn" onClick={() => removeJob(job().id)} disabled={job().status === "running" || job().status === "cancelling"}>Remove</button>
        </div>
      </header>

      <Show when={job().status === "queued"}>
        <section class="tx-status-panel tx-status-queued">
          <h3>Queued</h3>
          <p>Waiting for an available worker. You can navigate away — this job will keep its place.</p>
        </section>
      </Show>

      <Show when={job().status === "running"}>
        <section class="tx-status-panel tx-status-running">
          <GreekScrollLoader />
          <h3>{job().settings.mode === "annotate" ? "Annotating..." : "Transcribing..."}</h3>
          <p>{job().progress?.message || "Preparing audio"}</p>
          <div class="tx-progress-wrap">
            <TranscriptionProgressBar progress={job().progress} />
          </div>
        </section>
      </Show>

      <Show when={job().status === "cancelling"}>
        <section class="tx-status-panel tx-status-cancelled">
          <h3>Cancelling...</h3>
          <p>Waiting for backend workers to stop cleanly.</p>
        </section>
      </Show>

      <Show when={job().status === "failed" && job().error}>
        <section class="tx-status-panel tx-status-failed">
          <h3>Failed</h3>
          <pre class="tx-error-text">{job().error}</pre>
        </section>
      </Show>

      <Show when={job().status === "cancelled"}>
        <section class="tx-status-panel tx-status-cancelled">
          <h3>Cancelled</h3>
          <p>This job was cancelled before completion. Use Retry to queue it again.</p>
        </section>
      </Show>

      <Show when={job().status === "completed"}>
        <div class="tx-meta-grid">
          <Show
            when={job().settings.mode === "transcribe" && job().result}
            fallback={
              <>
                <MetaItem label="Language" value={job().annotatedResult?.language || "Unknown"} />
                <MetaItem label="Speakers" value={String(job().annotatedResult?.num_speakers ?? 0)} />
                <MetaItem label="Duration" value={formatTime(job().annotatedResult?.audio_duration || 0, false)} />
                <MetaItem label="Processing" value={`${(job().annotatedResult?.total_inference_time || 0).toFixed(1)}s`} />
              </>
            }
          >
            {(res) => (
              <>
                <MetaItem label="Language" value={res().language || "Unknown"} />
                <MetaItem label="Duration" value={formatTime(res().duration, false)} />
                <MetaItem label="Processing" value={`${res().inference_time.toFixed(1)}s`} />
                <MetaItem label="Model" value={res().model} />
              </>
            )}
          </Show>
        </div>

        <nav class="tx-tabs" aria-label="Result formats">
          <div class="tx-tab-item">
            <button class={`tx-tab ${activeTab() === "srt" ? "active" : ""}`} onClick={() => setInspectorTab(job().id, "srt")}>SRT</button>
            <button
              class="tx-tab-download"
              onClick={() => void exportToFile(`${exportBaseName()}.srt`, ["srt"], srtContent())}
              disabled={!srtContent().trim()}
              title="Download .srt"
              aria-label="Download SRT"
            >
              <svg viewBox="0 0 24 24" width="12" height="12" aria-hidden="true">
                <path d="M12 3v12" />
                <path d="M7 10l5 5 5-5" />
                <path d="M5 20h14" />
              </svg>
              <span>.srt</span>
            </button>
          </div>
          <div class="tx-tab-item">
            <button class={`tx-tab ${activeTab() === "text" ? "active" : ""}`} onClick={() => setInspectorTab(job().id, "text")}>Text</button>
            <button
              class="tx-tab-download"
              onClick={() => void exportToFile(`${exportBaseName()}.txt`, ["txt"], textContent())}
              disabled={!textContent().trim()}
              title="Download .txt"
              aria-label="Download text"
            >
              <svg viewBox="0 0 24 24" width="12" height="12" aria-hidden="true">
                <path d="M12 3v12" />
                <path d="M7 10l5 5 5-5" />
                <path d="M5 20h14" />
              </svg>
              <span>.txt</span>
            </button>
          </div>
        </nav>

        <Show when={activeTab() === "srt"}>
          <Show
            when={job().settings.mode === "annotate"}
            fallback={
              <div class="tx-panel">
                <div class="tx-panel-head">
                  <h3>SRT</h3>
                </div>
                <Show
                  when={srtContent().trim()}
                  fallback={<p class="tx-panel-empty">No SRT output is available for this job.</p>}
                >
                  <pre class="tx-transcript tx-srt-output">{srtContent()}</pre>
                </Show>
              </div>
            }
          >
            <div class="tx-panel tx-annotated-layout">
              <div class="tx-annotated-main">
                <div class="tx-panel-head">
                  <h3>SRT</h3>
                </div>
                <Show
                  when={srtContent().trim()}
                  fallback={<p class="tx-panel-empty">No SRT output is available for this job.</p>}
                >
                  <pre class="tx-transcript tx-srt-output">{srtContent()}</pre>
                </Show>
              </div>
              {renderSpeakerEditor()}
            </div>
          </Show>
        </Show>

        <Show when={activeTab() === "text"}>
          <Show
            when={job().settings.mode === "transcribe" && job().result}
            fallback={
              <div class="tx-panel tx-annotated-layout">
                <div class="tx-annotated-main">
                  <div class="tx-panel-head">
                    <h3>Annotated Transcript</h3>
                  </div>
                  <div class="tx-segments">
                    <For each={annotatedSegments()}>
                      {(segment) => (
                        <div class="tx-segment">
                          <span class="tx-segment-time">{formatTimestamp(segment.start)}</span>
                          <span class="tx-segment-speaker">{segment.speaker_name}</span>
                          <span class="tx-segment-text">{segment.text}</span>
                        </div>
                      )}
                    </For>
                  </div>
                </div>
                {renderSpeakerEditor()}
              </div>
            }
          >
            {(res) => (
              <div class="tx-panel">
                <div class="tx-panel-head">
                  <h3>Text</h3>
                </div>
                <div class="tx-transcript">{res().text || textContent()}</div>
              </div>
            )}
          </Show>
        </Show>
      </Show>
    </>
  );
};

const MetaItem: Component<{ label: string; value: string }> = (props) => (
  <div class="tx-meta-item">
    <span class="tx-meta-label">{props.label}</span>
    <span class="tx-meta-value">{props.value}</span>
  </div>
);

interface SettingsPanelProps {
  isAnnotateMode: boolean;
  isEnglishOnlyModel: boolean;
  availableSpeakerDevices: SpeakerDevice[];
  availableLanguages: LanguageOption[];
  onModelChange: (value: WhisperModel) => void;
  onLanguageChange: (value: string) => void;
}

const SettingsPanel: Component<SettingsPanelProps> = (props) => {
  const defaults = () => queueState.defaults;

  return (
    <section class="tx-settings-panel" role="complementary" aria-label="Default job settings">
      <header class="tx-settings-head">
        <div>
          <span class="tx-settings-eyebrow">Defaults for new jobs</span>
          <h2>Settings</h2>
        </div>
      </header>

      <div class="tx-settings-body">
        <div class="tx-field">
          <label class="tx-field-label">
            Mode
            <InfoIcon
              content="Transcription returns plain text. Annotate adds speaker diarization for multi-speaker audio."
              position="right"
            />
          </label>
          <div class="tx-toggle">
            <button class={`tx-toggle-btn ${!props.isAnnotateMode ? "active" : ""}`} onClick={() => setDefault("mode", "transcribe")}>Transcribe</button>
            <button class={`tx-toggle-btn ${props.isAnnotateMode ? "active" : ""}`} onClick={() => setDefault("mode", "annotate")}>Annotate</button>
          </div>
        </div>

        <div class="tx-field">
          <label class="tx-field-label" for="model-select">
            Model
            <InfoIcon
              content="Smaller models run faster with less memory. Larger models are more accurate."
              position="right"
            />
          </label>
          <div class="tx-select">
            <select id="model-select" value={defaults().model} onChange={(e) => props.onModelChange(e.currentTarget.value as WhisperModel)}>
              <For each={MODEL_OPTIONS}>{(o) => <option value={o.value}>{o.label}</option>}</For>
            </select>
            <svg viewBox="0 0 24 24"><path d="M6 9l6 6 6-6" /></svg>
          </div>
          <span class="tx-field-hint">{MODEL_OPTIONS.find((m) => m.value === defaults().model)?.description}</span>
          <Show when={queueState.modelValidation && queueState.modelValidation.status === "warning"}>
            <div class="tx-field-warning">
              <svg viewBox="0 0 24 24" width="14" height="14">
                <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm1 15h-2v-2h2v2zm0-4h-2V7h2v6z" fill="currentColor" />
              </svg>
              <For each={queueState.modelValidation!.messages}>{(message) => <span>{message}</span>}</For>
            </div>
          </Show>
        </div>

        <div class="tx-field">
          <label class="tx-field-label">Task</label>
          <div class="tx-toggle">
            <button class={`tx-toggle-btn ${defaults().task === "transcribe" ? "active" : ""}`} onClick={() => setDefault("task", "transcribe")}>Transcribe</button>
            <button class={`tx-toggle-btn ${defaults().task === "translate" ? "active" : ""}`} onClick={() => setDefault("task", "translate" as TranscriptionTask)}>Translate → English</button>
          </div>
        </div>

        <div class="tx-field">
          <label class="tx-field-label" for="language-select">Source Language</label>
          <div class="tx-select">
            <select
              id="language-select"
              value={defaults().language || ""}
              onChange={(e) => props.onLanguageChange(e.currentTarget.value)}
              disabled={props.isEnglishOnlyModel}
            >
              <For each={props.availableLanguages}>{(o) => <option value={o.value || ""}>{o.label}</option>}</For>
            </select>
            <svg viewBox="0 0 24 24"><path d="M6 9l6 6 6-6" /></svg>
          </div>
        </div>

        <Show when={props.isAnnotateMode}>
          <div class="tx-settings-group">
            <span class="tx-settings-group-label">Diarization</span>

            <div class="tx-field">
              <label class="tx-field-label" for="speaker-model-select">Speaker Model</label>
              <div class="tx-select">
                <select
                  id="speaker-model-select"
                  value={defaults().speakerModel}
                  onChange={(e) => setDefault("speakerModel", e.currentTarget.value as SpeakerModelKey)}
                >
                  <For each={queueState.speakerModelRequirements}>
                    {(m) => <option value={m.key}>{m.display_name} ({m.approx_size_mb.toFixed(1)} MB)</option>}
                  </For>
                </select>
                <svg viewBox="0 0 24 24"><path d="M6 9l6 6 6-6" /></svg>
              </div>
            </div>

            <div class="tx-field">
              <label class="tx-field-label" for="speaker-device-select">Device</label>
              <div class="tx-select">
                <select
                  id="speaker-device-select"
                  value={defaults().speakerDevice}
                  onChange={(e) => setDefault("speakerDevice", e.currentTarget.value as SpeakerDevice)}
                >
                  <For each={props.availableSpeakerDevices}>{(d) => <option value={d}>{d.toUpperCase()}</option>}</For>
                </select>
                <svg viewBox="0 0 24 24"><path d="M6 9l6 6 6-6" /></svg>
              </div>
            </div>

            <div class="tx-field">
              <label class="tx-field-label" for="num-speakers-input">Expected speakers</label>
              <input
                id="num-speakers-input"
                type="number"
                min="1"
                max="20"
                placeholder="Auto-detect"
                value={defaults().numSpeakers ?? ""}
                onInput={(e) => {
                  const value = e.currentTarget.value;
                  setDefault("numSpeakers", value === "" ? null : parseInt(value, 10));
                }}
                class="tx-input"
              />
            </div>

            <div class="tx-field">
              <label class="tx-field-label" for="diarize-threshold-input">
                Clustering threshold <span class="tx-field-value">{defaults().diarizeThreshold.toFixed(2)}</span>
              </label>
              <input
                id="diarize-threshold-input"
                type="range"
                min="0.1"
                max="0.9"
                step="0.05"
                value={defaults().diarizeThreshold}
                onInput={(e) => setDefault("diarizeThreshold", parseFloat(e.currentTarget.value))}
                class="tx-range"
              />
            </div>
          </div>
        </Show>
      </div>

      <footer class="tx-settings-foot">
        <p class="tx-settings-foot-hint">
          Existing queued jobs keep the settings they were enqueued with.
        </p>
      </footer>
    </section>
  );
};

export default Transcription;
