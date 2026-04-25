import { Component, For, Show, createEffect, createMemo, createSignal, onMount } from "solid-js";
import { useNavigate } from "@solidjs/router";
import { invoke } from "@tauri-apps/api/core";
import { open, save } from "@tauri-apps/plugin-dialog";
import { useTheme } from "../utils/theme";
import TextFileUploader from "../components/TextFileUploader";
import GreekScrollLoader from "../components/GreekScrollLoader";
import TranslationProgressBar from "../components/TranslationProgressBar";
import InfoIcon from "../components/InfoIcon";
import ConfirmDialog from "../components/ConfirmDialog";
import DownloadProgressBar from "../components/DownloadProgressBar";
import type {
  TranslationJobSettings,
  TranslationStrategy,
  TranslationJobStatus,
  TranslationQueueJob,
} from "../types/translation";
import { MADLAD_LANGUAGES, getLanguageName } from "../types/translation";
import {
  cancelTranslationJob,
  clearAllTranslationJobs,
  clearCompletedTranslationJobs,
  dismissTranslationQueueError,
  enqueueTranslationFiles,
  initTranslationJobQueue,
  removeTranslationJob,
  retryAllFailedTranslationJobs,
  retryFailedTranslationJob,
  selectedTranslationJob,
  startQueuedTranslationJobs,
  setSelectedTranslationJob,
  setTranslationDefault,
  setTranslationMaxConcurrency,
  sortedTranslationJobs,
  translationActiveCount,
  translationCancelledCount,
  translationCompletedCount,
  translationFailedCount,
  translationQueuedCount,
  translationQueueState,
} from "../stores/translationJobQueue";
import "./Transcription.css";
import "./Translation.css";

const QUEUE_EXPANDED_STORAGE_KEY = "hermeneia-translation-queue-expanded";
const FAST_CHIP_LABEL = "Fast";
const SLOW_CHIP_LABEL = "Slower (larger general model)";

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

const statusLabel = (status: TranslationJobStatus): string => {
  if (status === "queued") return "Queued";
  if (status === "waiting_resources") return "Waiting";
  if (status === "downloading_model") return "Downloading";
  if (status === "loading_model") return "Loading";
  if (status === "running") return "Running";
  if (status === "cancelling") return "Cancelling";
  if (status === "completed") return "Done";
  if (status === "failed") return "Failed";
  return "Cancelled";
};

const isActiveStatus = (status: TranslationJobStatus): boolean => (
  status === "waiting_resources"
  || status === "downloading_model"
  || status === "loading_model"
  || status === "running"
  || status === "cancelling"
);

const modelIsSlowerGeneral = (value: string): boolean => {
  const normalized = value.toLowerCase();
  return normalized.includes("madlad") || normalized.includes("nllb") || normalized.includes("m2m") || normalized.includes("general");
};

const pairKey = (sourceLang: string, targetLang: string): string => `${sourceLang}::${targetLang}`;

const getProgressPercent = (job: TranslationQueueJob): number | null => {
  const progress = job.progress;
  if (!progress || progress.current === null || progress.total === null || progress.total === 0) {
    if (job.downloadProgress?.bytes_total) {
      return Math.min(100, Math.round((job.downloadProgress.bytes_downloaded / job.downloadProgress.bytes_total) * 100));
    }
    return null;
  }
  return Math.min(100, Math.round((progress.current / progress.total) * 100));
};

const getProgressLabel = (job: TranslationQueueJob): string => {
  const percent = getProgressPercent(job);
  if (percent !== null && isActiveStatus(job.status) && job.status !== "loading_model" && job.status !== "waiting_resources") {
    return `${percent}%`;
  }
  if (job.status === "loading_model") return "Loading model";
  if (job.status === "waiting_resources") return "Waiting for resources";
  if (job.status === "cancelling") return "Cancelling";
  if (job.progress?.message) return job.progress.message;
  return statusLabel(job.status);
};

const strategyChipLabel = (
  strategy: TranslationStrategy,
  pairSupported: boolean | undefined,
  engineTier: "fast" | "universal" | undefined,
  modelUsed: string | undefined
): string => {
  if (modelUsed) {
    return modelIsSlowerGeneral(modelUsed) ? SLOW_CHIP_LABEL : FAST_CHIP_LABEL;
  }
  if (engineTier) {
    return engineTier === "universal" ? SLOW_CHIP_LABEL : FAST_CHIP_LABEL;
  }
  if (strategy === "fast_only") return FAST_CHIP_LABEL;
  if (strategy === "universal") return SLOW_CHIP_LABEL;
  if (pairSupported === false) return SLOW_CHIP_LABEL;
  return FAST_CHIP_LABEL;
};

const sanitizeExportBaseName = (fileName: string): string => {
  const base = fileName.replace(/\.[^/.]+$/, "").trim();
  const sanitized = base.replace(/[<>:"/\\|?*\u0000-\u001f]/g, "_").replace(/\s+/g, " ").trim();
  return sanitized || "translation";
};

const QueueStatusIcon: Component<{ status: TranslationJobStatus }> = (props) => {
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
    if (props.status === "downloading_model" || props.status === "loading_model" || props.status === "waiting_resources") {
      return (
        <svg viewBox="0 0 24 24" aria-hidden="true">
          <circle cx="12" cy="12" r="8" />
          <path d="M12 8v4l2.5 2.5" />
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
      <Show when={props.status === "running" || props.status === "downloading_model" || props.status === "loading_model"}>
        <span class="tx-status-icon-halo" aria-hidden="true" />
      </Show>
      {icon()}
    </span>
  );
};

const exportJobResult = async (job: TranslationQueueJob): Promise<void> => {
  if (job.status !== "completed" || !job.result) return;

  const extension = job.result.is_subtitle ? "srt" : "txt";
  const filterName = job.result.is_subtitle ? "SRT Subtitle Files" : "Text Files";
  const baseName = sanitizeExportBaseName(job.fileName);
  const outputPath = await save({
    filters: [{ name: filterName, extensions: [extension] }],
    defaultPath: `${baseName}_${job.settings.targetLang}.${extension}`,
  });
  if (!outputPath) return;

  await invoke("write_text_file", {
    path: outputPath,
    content: job.result.translated_text,
  });
};

const Translation: Component = () => {
  const navigate = useNavigate();
  const { toggleTheme } = useTheme();

  const [showClearDialog, setShowClearDialog] = createSignal(false);
  const [queueExpanded, setQueueExpanded] = createSignal(loadQueueExpandedPreference());
  const [marianSupported, setMarianSupported] = createSignal<boolean>(true);
  const [pairSupportMap, setPairSupportMap] = createSignal<Record<string, boolean>>({});
  const pendingPairChecks = new Set<string>();

  const defaults = () => translationQueueState.defaults;
  const totalJobs = createMemo(() => translationQueueState.jobs.length);

  const modeStatusMessage = createMemo(() => {
    if (defaults().strategy === "fast_only" && !marianSupported()) {
      return {
        tone: "warning",
        text: "Fast only is not available for this language pair. Switch to Auto (recommended) or More languages (slower) to use the larger, more general model.",
      };
    }
    if (defaults().strategy === "universal") {
      return {
        tone: "slower",
        text: "More languages (slower) uses a larger, more general model. It supports more language pairs, but translations take longer.",
      };
    }
    if (marianSupported()) {
      return {
        tone: "fast",
        text: "This language pair can run on the fast model.",
      };
    }
    return {
      tone: "slower",
      text: "Auto mode will use a larger, more general model for this language pair, so translation will be slower.",
    };
  });

  const availableSourceLanguages = createMemo(() =>
    MADLAD_LANGUAGES.filter((lang) => lang.code !== defaults().targetLang)
  );
  const availableTargetLanguages = createMemo(() =>
    MADLAD_LANGUAGES.filter((lang) => lang.code !== defaults().sourceLang)
  );

  const progressSegments = createMemo(() => {
    const total = totalJobs();
    if (total === 0) return null;
    return {
      completed: (translationCompletedCount() / total) * 100,
      running: (translationActiveCount() / total) * 100,
      failed: (translationFailedCount() / total) * 100,
      cancelled: (translationCancelledCount() / total) * 100,
    };
  });

  const isValidDefaultsPair = createMemo(() => defaults().sourceLang !== defaults().targetLang);

  onMount(() => {
    void initTranslationJobQueue();
  });

  createEffect(() => {
    if (typeof window === "undefined") return;
    try {
      window.localStorage.setItem(QUEUE_EXPANDED_STORAGE_KEY, queueExpanded() ? "1" : "0");
    } catch {
      // best-effort persistence
    }
  });

  createEffect(() => {
    const src = defaults().sourceLang;
    const tgt = defaults().targetLang;
    if (!src || !tgt || src === tgt) {
      setMarianSupported(false);
      return;
    }

    const requestedSourceLang = src;
    const requestedTargetLang = tgt;

    void invoke<boolean>("check_marian_pair_supported", {
      sourceLang: requestedSourceLang,
      targetLang: requestedTargetLang,
    })
      .then((supported) => {
        if (defaults().sourceLang === requestedSourceLang && defaults().targetLang === requestedTargetLang) {
          setMarianSupported(supported);
        }
      })
      .catch(() => {
        if (defaults().sourceLang === requestedSourceLang && defaults().targetLang === requestedTargetLang) {
          setMarianSupported(false);
        }
      });
  });

  createEffect(() => {
    const supportMap = pairSupportMap();
    const keys = sortedTranslationJobs()
      .filter((job) => job.status !== "completed")
      .map((job) => pairKey(job.settings.sourceLang, job.settings.targetLang));

    keys.forEach((key) => {
      if (supportMap[key] !== undefined || pendingPairChecks.has(key)) return;
      const [sourceLang, targetLang] = key.split("::");
      pendingPairChecks.add(key);
      void invoke<boolean>("check_marian_pair_supported", {
        sourceLang,
        targetLang,
      })
        .then((supported) => {
          setPairSupportMap((prev) => ({ ...prev, [key]: supported }));
        })
        .catch(() => {
          setPairSupportMap((prev) => ({ ...prev, [key]: false }));
        })
        .finally(() => {
          pendingPairChecks.delete(key);
        });
    });
  });

  createEffect(() => {
    if (totalJobs() === 0) return;
    if (selectedTranslationJob()) return;
    const first = sortedTranslationJobs()[0];
    if (first) setSelectedTranslationJob(first.id);
  });

  const setDefaultSource = (value: string) => {
    setTranslationDefault("sourceLang", value);
    if (value === defaults().targetLang) {
      const replacement = MADLAD_LANGUAGES.find((lang) => lang.code !== value);
      if (replacement) setTranslationDefault("targetLang", replacement.code);
    }
  };

  const setDefaultTarget = (value: string) => {
    setTranslationDefault("targetLang", value);
    if (value === defaults().sourceLang) {
      const replacement = MADLAD_LANGUAGES.find((lang) => lang.code !== value);
      if (replacement) setTranslationDefault("sourceLang", replacement.code);
    }
  };

  const openAddFilesDialog = async () => {
    try {
      const selected = await open({
        multiple: true,
        filters: [{ name: "Text Files", extensions: ["txt", "srt"] }],
      });
      if (!selected) return;
      const paths = Array.isArray(selected) ? selected : [selected];
      enqueueTranslationFiles(paths);
      setQueueExpanded(true);
    } catch (err) {
      console.error("Failed to add files:", err);
    }
  };

  const cancelRunning = async () => {
    const active = translationQueueState.jobs.filter((job) => isActiveStatus(job.status));
    await Promise.allSettled(active.map((job) => cancelTranslationJob(job.id)));
  };

  const exportCompletedBundle = async () => {
    const completed = translationQueueState.jobs.filter((job) => job.status === "completed" && job.result);
    if (completed.length === 0) return;

    const nameCounts = new Map<string, number>();
    const entries: { path: string; content: string }[] = [];

    completed.forEach((job) => {
      const baseName = sanitizeExportBaseName(job.fileName);
      const seen = nameCounts.get(baseName) || 0;
      nameCounts.set(baseName, seen + 1);
      const uniqueBaseName = seen === 0 ? baseName : `${baseName}_${seen + 1}`;

      if (!job.result) return;
      const extension = job.result.is_subtitle ? "srt" : "txt";
      entries.push({
        path: `${uniqueBaseName}_${job.settings.targetLang}.${extension}`,
        content: job.result.translated_text,
      });
    });

    if (entries.length === 0) return;

    try {
      const defaultPath = `hermeneia_translation_batch_${new Date().toISOString().slice(0, 10)}.zip`;
      const outputPath = await save({
        filters: [{ name: "Zip Archive", extensions: ["zip"] }],
        defaultPath,
      });
      if (!outputPath) return;
      await invoke("write_zip_archive", { path: outputPath, entries });
    } catch (err) {
      console.error("Failed to export translation zip:", err);
    }
  };

  const requestClearAll = () => {
    if (totalJobs() === 0) return;
    setShowClearDialog(true);
  };

  const confirmClearAll = async () => {
    setShowClearDialog(false);
    await clearAllTranslationJobs();
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

      <div class="scroll-container tx-scroll-container trq-scroll-container">
        <div class="scroll-rod"></div>
        <main class="parchment tx-parchment trq-parchment">
          <div class="tx-shell trq-shell">
            <header class="tx-header">
              <button class="tx-back" onClick={() => navigate("/")}>
                <svg viewBox="0 0 24 24" width="18" height="18">
                  <path d="M19 12H5M12 19l-7-7 7-7" />
                </svg>
                <span>Home</span>
              </button>

              <div class="tx-title">
                <h1>Translation</h1>
              </div>

              <div class="tx-header-actions">
                <div class="tx-worker-select">
                  <label for="translation-worker-select">Jobs at once</label>
                  <div class="tx-select">
                    <select
                      id="translation-worker-select"
                      value={String(translationQueueState.maxConcurrency)}
                      onChange={(e) => setTranslationMaxConcurrency(parseInt(e.currentTarget.value, 10))}
                    >
                      <For each={[1, 2, 3, 4].filter((value) => value <= translationQueueState.maxConcurrencyLimit)}>{(value) => <option value={value}>{value}</option>}</For>
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
                  <Show when={translationActiveCount() > 0}>
                    <span class="tx-count-pill tx-count-running">{translationActiveCount()} running</span>
                  </Show>
                  <Show when={translationQueuedCount() > 0}>
                    <span class="tx-count-pill tx-count-queued">{translationQueuedCount()} queued</span>
                  </Show>
                  <Show when={translationCompletedCount() > 0}>
                    <span class="tx-count-pill tx-count-completed">{translationCompletedCount()} done</span>
                  </Show>
                  <Show when={translationFailedCount() > 0}>
                    <span class="tx-count-pill tx-count-failed">{translationFailedCount()} failed</span>
                  </Show>
                  <Show when={translationCancelledCount() > 0}>
                    <span class="tx-count-pill tx-count-cancelled">{translationCancelledCount()} cancelled</span>
                  </Show>
                </div>

                <Show when={progressSegments()}>
                  {(seg) => (
                    <div class="tx-batch-bar" role="progressbar" aria-valuenow={translationCompletedCount()} aria-valuemax={totalJobs()}>
                      <div class="tx-batch-bar-seg tx-seg-completed" style={{ width: `${seg().completed}%` }} />
                      <div class="tx-batch-bar-seg tx-seg-running" style={{ width: `${seg().running}%` }} />
                      <div class="tx-batch-bar-seg tx-seg-failed" style={{ width: `${seg().failed}%` }} />
                      <div class="tx-batch-bar-seg tx-seg-cancelled" style={{ width: `${seg().cancelled}%` }} />
                    </div>
                  )}
                </Show>

                <div class="tx-batch-actions">
                  <button class="tx-link-btn" onClick={() => void cancelRunning()} disabled={translationActiveCount() === 0}>
                    Cancel running
                  </button>
                  <button class="tx-link-btn" onClick={retryAllFailedTranslationJobs} disabled={translationFailedCount() === 0}>
                    Retry failed
                  </button>
                  <button class="tx-link-btn" onClick={clearCompletedTranslationJobs} disabled={translationCompletedCount() === 0}>
                    Clear completed
                  </button>
                  <button class="tx-link-btn" onClick={() => void exportCompletedBundle()} disabled={translationCompletedCount() === 0}>
                    Export all ({translationCompletedCount()})
                  </button>
                  <button class="tx-link-btn tx-link-btn-danger" onClick={requestClearAll} disabled={totalJobs() === 0}>
                    Clear all
                  </button>
                </div>
              </div>
            </Show>

            <Show when={translationQueueState.queueError}>
              <div class="tx-banner tx-banner-error">
                <span>{translationQueueState.queueError}</span>
                <button onClick={dismissTranslationQueueError}>Dismiss</button>
              </div>
            </Show>

            <div class="tx-main-layout trq-main-layout">
              <section class="tx-workbench">
                <Show
                  when={totalJobs() > 0}
                  fallback={
                    <section class="tx-uploader-panel">
                      <TextFileUploader onFilesSelected={enqueueTranslationFiles} />
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
                        <div class="tx-queue-head-actions">
                          <button class="tx-queue-toggle" onClick={startQueuedTranslationJobs} disabled={translationQueuedCount() === 0}>
                            <svg viewBox="0 0 24 24" width="14" height="14">
                              <path d="M8 5v14l11-7z" />
                            </svg>
                            <span>Start ({translationQueuedCount()})</span>
                          </button>
                          <button
                            class="tx-queue-toggle"
                            aria-expanded={queueExpanded()}
                            aria-controls="translation-queue-list"
                            onClick={() => setQueueExpanded(!queueExpanded())}
                          >
                            <span>{queueExpanded() ? "Hide queue" : "Show queue"}</span>
                            <svg viewBox="0 0 24 24" width="14" height="14" classList={{ expanded: queueExpanded() }}>
                              <path d="M6 9l6 6 6-6" />
                            </svg>
                          </button>
                        </div>
                      </header>

                      <Show when={queueExpanded()}>
                        <ul id="translation-queue-list" class="tx-queue-list" role="list">
                          <For each={sortedTranslationJobs()}>
                            {(job) => (
                              <li>
                                <button
                                  class={`tx-queue-row ${translationQueueState.selectedJobId === job.id ? "selected" : ""}`}
                                  onClick={() => setSelectedTranslationJob(job.id)}
                                >
                                  <div class="tx-queue-row-head">
                                    <QueueStatusIcon status={job.status} />
                                    <span class="tx-queue-row-name" title={job.fileName}>{job.fileName}</span>
                                    <span class={`tx-queue-row-state ${job.status}`}>{statusLabel(job.status)}</span>
                                  </div>

                                  <div class="tx-queue-row-meta">
                                    <span class="tx-chip">{job.settings.sourceLang} → {job.settings.targetLang}</span>
                                    <span class="tx-chip tx-chip-muted">
                                      {strategyChipLabel(
                                        job.settings.strategy,
                                        pairSupportMap()[pairKey(job.settings.sourceLang, job.settings.targetLang)],
                                        job.engine?.tier,
                                        job.result?.model_used,
                                      )}
                                    </span>
                                    <Show when={isActiveStatus(job.status)}>
                                      <span class="tx-queue-row-progress">{getProgressLabel(job)}</span>
                                    </Show>
                                  </div>
                                </button>
                              </li>
                            )}
                          </For>
                        </ul>
                      </Show>
                    </section>

                    <section class="tx-inspector tx-inspector-main trq-inspector-main">
                      <Show
                        when={selectedTranslationJob()}
                        fallback={
                          <div class="tx-inspector-empty">
                            <h2>Select a job</h2>
                            <p>Choose a queue item to inspect progress and view translation output.</p>
                          </div>
                        }
                      >
                        {(job) => (
                          <TranslationInspector
                            job={job()}
                            pairSupported={pairSupportMap()[pairKey(job().settings.sourceLang, job().settings.targetLang)]}
                          />
                        )}
                      </Show>
                    </section>
                  </>
                </Show>
              </section>

              <TranslationSettingsPanel
                defaults={defaults()}
                modeStatusMessage={modeStatusMessage()}
                isValidPair={isValidDefaultsPair()}
                availableSourceLanguages={availableSourceLanguages()}
                availableTargetLanguages={availableTargetLanguages()}
                onSetSource={setDefaultSource}
                onSetTarget={setDefaultTarget}
              />
            </div>
          </div>
        </main>
        <div class="scroll-rod"></div>
      </div>

      <ConfirmDialog
        open={showClearDialog()}
        title="Clear translation queue?"
        message="This removes every translation job. Running jobs will be cancelled and unsaved outputs will be lost."
        confirmLabel="Clear All"
        cancelLabel="Keep Jobs"
        onConfirm={() => void confirmClearAll()}
        onCancel={() => setShowClearDialog(false)}
      />
    </>
  );
};

interface TranslationInspectorProps {
  job: TranslationQueueJob;
  pairSupported: boolean | undefined;
}

const TranslationInspector: Component<TranslationInspectorProps> = (props) => {
  const job = () => props.job;

  const canCancel = createMemo(() => isActiveStatus(job().status));
  const canRetry = createMemo(() => job().status === "failed");

  const downloadExtension = createMemo(() => {
    if (!job().result) return "txt";
    return job().result.is_subtitle ? "srt" : "txt";
  });

  const speedChipLabel = createMemo(() => strategyChipLabel(
    job().settings.strategy,
    props.pairSupported,
    job().engine?.tier,
    job().result?.model_used,
  ));

  return (
    <>
      <header class="tx-inspector-head">
        <div class="tx-inspector-title">
          <span class="tx-inspector-eyebrow">Selected Job</span>
          <h2 title={job().fileName}>{job().fileName}</h2>
          <div class="tx-inspector-meta">
            <span class="tx-chip">{getLanguageName(job().settings.sourceLang)} → {getLanguageName(job().settings.targetLang)}</span>
            <span class="tx-chip tx-chip-muted">{speedChipLabel()}</span>
            <span class={`tx-status-badge ${job().status}`}>{statusLabel(job().status)}</span>
          </div>
        </div>

        <div class="tx-inspector-actions">
          <Show when={canCancel()}>
            <button class="tx-btn" onClick={() => void cancelTranslationJob(job().id)}>Cancel</button>
          </Show>
          <Show when={canRetry()}>
            <button class="tx-btn" onClick={() => retryFailedTranslationJob(job().id)}>Retry</button>
          </Show>
          <button class="tx-btn" onClick={() => void removeTranslationJob(job().id)}>
            Remove
          </button>
        </div>
      </header>

      <Show when={job().status === "queued"}>
        <section class="tx-status-panel tx-status-queued">
          <h3>Queued</h3>
          <p>Waiting for an open worker slot. This job will run automatically.</p>
        </section>
      </Show>

      <Show when={job().status === "waiting_resources"}>
        <section class="tx-status-panel tx-status-running">
          <h3>Waiting for resources</h3>
          <p>{job().progress?.message || "Preparing translation resources..."}</p>
          <div class="tx-progress-wrap">
            <TranslationProgressBar progress={job().progress} />
          </div>
        </section>
      </Show>

      <Show when={job().status === "downloading_model"}>
        <section class="tx-status-panel tx-status-running trq-download-panel">
          <h3>Downloading model</h3>
          <p>{job().progress?.message || "Fetching model artifacts..."}</p>
          <div class="tx-progress-wrap trq-download-wrap">
            <DownloadProgressBar progress={job().downloadProgress} onCancel={() => void cancelTranslationJob(job().id)} />
          </div>
        </section>
      </Show>

      <Show when={job().status === "loading_model" || job().status === "running"}>
        <section class="tx-status-panel tx-status-running">
          <GreekScrollLoader />
          <h3>{job().status === "loading_model" ? "Loading model..." : "Translating..."}</h3>
          <p>{job().progress?.message || job().engine?.message || "Processing file"}</p>
          <div class="tx-progress-wrap">
            <TranslationProgressBar progress={job().progress} />
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
          <p>This job was cancelled before completion. Add the file again to run a new translation.</p>
        </section>
      </Show>

      <Show when={job().status === "completed" && job().result}>
        {(res) => (
          <>
            <div class="tx-meta-grid">
              <MetaItem label="From" value={getLanguageName(res().source_language)} />
              <MetaItem label="To" value={getLanguageName(res().target_language)} />
              <MetaItem label="Processing" value={`${res().inference_time.toFixed(1)}s`} />
              <MetaItem label="Segments" value={String(res().segments_translated)} />
            </div>

            <div class="trq-compare-grid">
              <div class="tx-panel">
                <div class="tx-panel-head">
                  <h3>Original ({getLanguageName(res().source_language)})</h3>
                </div>
                <pre class="tx-transcript trq-text-output">{res().original_text}</pre>
              </div>

              <div class="tx-panel">
                <div class="tx-panel-head trq-translated-head">
                  <h3>Translated ({getLanguageName(res().target_language)})</h3>
                  <button
                    class="tx-tab-download trq-translated-download"
                    onClick={() => void exportJobResult(job())}
                    title={`Download .${downloadExtension()}`}
                    aria-label={`Download translated file as .${downloadExtension()}`}
                  >
                    <svg viewBox="0 0 24 24" width="12" height="12" aria-hidden="true">
                      <path d="M12 3v12" />
                      <path d="M7 10l5 5 5-5" />
                      <path d="M5 20h14" />
                    </svg>
                    <span>.{downloadExtension()}</span>
                  </button>
                </div>
                <pre class="tx-transcript trq-text-output">{res().translated_text}</pre>
              </div>
            </div>
          </>
        )}
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

interface TranslationSettingsPanelProps {
  defaults: TranslationJobSettings;
  modeStatusMessage: { tone: string; text: string };
  isValidPair: boolean;
  availableSourceLanguages: { code: string; name: string }[];
  availableTargetLanguages: { code: string; name: string }[];
  onSetSource: (value: string) => void;
  onSetTarget: (value: string) => void;
}

const TranslationSettingsPanel: Component<TranslationSettingsPanelProps> = (props) => (
  <section class="tx-settings-panel" role="complementary" aria-label="Default translation settings">
    <header class="tx-settings-head">
      <div>
        <span class="tx-settings-eyebrow">Defaults for new jobs</span>
        <h2>Settings</h2>
      </div>
    </header>

    <div class="tx-settings-body">
      <div class="tx-field">
        <label class="tx-field-label" for="translation-source-lang">
          Source Language
          <InfoIcon
            content="Language of the original text in newly queued jobs."
            position="right"
          />
        </label>
        <div class="tx-select">
          <select
            id="translation-source-lang"
            value={props.defaults.sourceLang}
            onChange={(e) => props.onSetSource(e.currentTarget.value)}
          >
            <For each={props.availableSourceLanguages}>{(lang) => <option value={lang.code}>{lang.name}</option>}</For>
          </select>
          <svg viewBox="0 0 24 24"><path d="M6 9l6 6 6-6" /></svg>
        </div>
      </div>

      <div class="tx-field">
        <label class="tx-field-label" for="translation-target-lang">
          Target Language
          <InfoIcon
            content="Language used for translated output in newly queued jobs."
            position="right"
          />
        </label>
        <div class="tx-select">
          <select
            id="translation-target-lang"
            value={props.defaults.targetLang}
            onChange={(e) => props.onSetTarget(e.currentTarget.value)}
          >
            <For each={props.availableTargetLanguages}>{(lang) => <option value={lang.code}>{lang.name}</option>}</For>
          </select>
          <svg viewBox="0 0 24 24"><path d="M6 9l6 6 6-6" /></svg>
        </div>
      </div>

      <div class="tx-field">
        <label class="tx-field-label">
          Translation mode
          <InfoIcon
            content={(
              <span class="trq-tooltip-compact">
                Auto uses fast when possible. Fast only supports fewer language pairs. More languages (slower)
                always uses the larger general model.
              </span>
            )}
            position="right"
          />
        </label>
        <div class="tx-toggle tx-toggle-three">
          <button
            class={`tx-toggle-btn ${props.defaults.strategy === "auto" ? "active" : ""}`}
            onClick={() => setTranslationDefault("strategy", "auto")}
          >
            Auto (recommended)
          </button>
          <button
            class={`tx-toggle-btn ${props.defaults.strategy === "fast_only" ? "active" : ""}`}
            onClick={() => setTranslationDefault("strategy", "fast_only")}
          >
            Fast only
          </button>
          <button
            class={`tx-toggle-btn ${props.defaults.strategy === "universal" ? "active" : ""}`}
            onClick={() => setTranslationDefault("strategy", "universal")}
          >
            More languages (slower)
          </button>
        </div>
        <div class={`trq-quality-note trq-quality-note-${props.modeStatusMessage.tone}`}>
          <span class="trq-quality-note-label">Pair status</span>
          <span class="trq-quality-note-value">{props.modeStatusMessage.text}</span>
        </div>
      </div>

      <Show when={!props.isValidPair}>
        <div class="tx-field-warning">
          <span>Source and target languages must be different.</span>
        </div>
      </Show>
    </div>

    <footer class="tx-settings-foot">
      <p class="tx-settings-foot-hint">Existing queued jobs keep the settings they were enqueued with.</p>
    </footer>
  </section>
);

export default Translation;
