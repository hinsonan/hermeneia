import { createStore, produce } from "solid-js/store";
import { createMemo } from "solid-js";
import { invoke } from "@tauri-apps/api/core";
import { listen, UnlistenFn } from "@tauri-apps/api/event";
import type {
  AnnotatedResult,
  AnnotationProgress,
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

export type JobStatus = "queued" | "running" | "completed" | "failed" | "cancelled";
export type JobMode = "transcribe" | "annotate";
export type JobProgress = TranscriptionProgress | AnnotationProgress;
export type InspectorTab = "srt" | "text";

export interface JobSettings {
  mode: JobMode;
  model: WhisperModel;
  task: TranscriptionTask;
  language: string | null;
  speakerModel: SpeakerModelKey;
  speakerDevice: SpeakerDevice;
  numSpeakers: number | null;
  diarizeThreshold: number;
}

export interface QueueJob {
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
  inspectorTab: InspectorTab;
}

interface QueueState {
  jobs: QueueJob[];
  selectedJobId: string | null;
  maxConcurrency: number;
  defaults: JobSettings;
  systemCapabilities: SystemCapabilities | null;
  modelValidation: ModelValidation | null;
  speakerModelRequirements: SpeakerModelRequirement[];
  queueError: string | null;
  listenersInitialized: boolean;
  capabilitiesLoaded: boolean;
}

const STORAGE_KEY = "hermeneia-job-queue";

const DEFAULT_SETTINGS: JobSettings = {
  mode: "transcribe",
  model: "tiny",
  task: "transcribe",
  language: null,
  speakerModel: "english",
  speakerDevice: "cpu",
  numSpeakers: null,
  diarizeThreshold: 0.5,
};

function makeId(): string {
  if (typeof crypto !== "undefined" && typeof crypto.randomUUID === "function") {
    return crypto.randomUUID();
  }
  return `${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

function getBaseName(path: string): string {
  return path.split("/").pop() || path.split("\\").pop() || path;
}

function normalizeInspectorTab(tab: unknown): InspectorTab {
  if (tab === "srt" || tab === "segments" || tab === "export") return "srt";
  if (tab === "text" || tab === "output") return "text";
  return "srt";
}

function loadPersistedState(): Partial<QueueState> {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return {};
    const parsed = JSON.parse(raw) as {
      jobs?: QueueJob[];
      selectedJobId?: string | null;
      maxConcurrency?: number;
      defaults?: JobSettings;
    };
    // Jobs that were running when the app closed are orphaned — the backend
    // job promise resolved into nothing. Mark them cancelled so the user can retry.
    const jobs = (parsed.jobs || []).map((job) =>
      job.status === "running"
        ? {
            ...job,
            status: "cancelled" as JobStatus,
            progress: null,
            error: null,
            inspectorTab: normalizeInspectorTab(job.inspectorTab),
          }
        : {
            ...job,
            inspectorTab: normalizeInspectorTab(job.inspectorTab),
          }
    );
    return {
      jobs,
      selectedJobId: parsed.selectedJobId ?? null,
      maxConcurrency: parsed.maxConcurrency ?? 2,
      defaults: parsed.defaults ?? DEFAULT_SETTINGS,
    };
  } catch {
    return {};
  }
}

const persisted = loadPersistedState();

const [state, setState] = createStore<QueueState>({
  jobs: persisted.jobs ?? [],
  selectedJobId: persisted.selectedJobId ?? null,
  maxConcurrency: persisted.maxConcurrency ?? 2,
  defaults: persisted.defaults ?? { ...DEFAULT_SETTINGS },
  systemCapabilities: null,
  modelValidation: null,
  speakerModelRequirements: [],
  queueError: null,
  listenersInitialized: false,
  capabilitiesLoaded: false,
});

let persistTimer: ReturnType<typeof setTimeout> | null = null;
function schedulePersist() {
  if (persistTimer) clearTimeout(persistTimer);
  persistTimer = setTimeout(() => {
    try {
      const snapshot = {
        jobs: state.jobs,
        selectedJobId: state.selectedJobId,
        maxConcurrency: state.maxConcurrency,
        defaults: state.defaults,
      };
      localStorage.setItem(STORAGE_KEY, JSON.stringify(snapshot));
    } catch {
      // best-effort persistence
    }
  }, 200);
}

let unlistenTranscription: UnlistenFn | null = null;
let unlistenAnnotation: UnlistenFn | null = null;

export async function initJobQueue(): Promise<void> {
  if (state.listenersInitialized) return;
  setState("listenersInitialized", true);

  unlistenTranscription = await listen<TranscriptionProgress>("transcription-progress", (event) => {
    const payload = event.payload;
    if (!payload?.job_id) return;
    updateJobProgress(payload.job_id, payload);
  });

  unlistenAnnotation = await listen<AnnotationProgress>("annotation-progress", (event) => {
    const payload = event.payload;
    if (!payload?.job_id) return;
    updateJobProgress(payload.job_id, payload);
  });

  await loadCapabilities();
}

export function teardownJobQueue() {
  if (unlistenTranscription) unlistenTranscription();
  if (unlistenAnnotation) unlistenAnnotation();
  unlistenTranscription = null;
  unlistenAnnotation = null;
  setState("listenersInitialized", false);
}

async function loadCapabilities(): Promise<void> {
  if (state.capabilitiesLoaded) return;
  setState("capabilitiesLoaded", true);

  try {
    const caps = await invoke<SystemCapabilities>("get_system_capabilities");
    setState("systemCapabilities", caps);
    const deviceType = caps.gpu_info?.device_type;
    if (deviceType === "NvidiaCuda" && state.defaults.speakerDevice === "cpu") {
      setState("defaults", "speakerDevice", "cuda");
    } else if (deviceType === "AppleMetal" && state.defaults.speakerDevice === "cpu") {
      setState("defaults", "speakerDevice", "coreml");
    }
  } catch (err) {
    console.warn("Failed to get system capabilities:", err);
  }

  try {
    const requirements = await invoke<SpeakerModelRequirement[]>("list_speaker_model_requirements");
    setState("speakerModelRequirements", requirements);
    if (requirements.length > 0 && !requirements.find((r) => r.key === state.defaults.speakerModel)) {
      setState("defaults", "speakerModel", requirements[0].key);
    }
  } catch (err) {
    console.warn("Failed to load speaker model requirements:", err);
  }

  await validateCurrentModel();
}

export async function validateCurrentModel(): Promise<void> {
  try {
    const validation = await invoke<ModelValidation>("validate_model_selection", {
      model: state.defaults.model,
      forceCpu: false,
    });
    setState("modelValidation", validation);
  } catch (err) {
    console.warn("Validation failed:", err);
  }
}

function updateJobProgress(jobId: string, progress: JobProgress) {
  setState(
    "jobs",
    (job) => job.id === jobId,
    "progress",
    progress
  );
}

function findJobIndex(jobId: string): number {
  return state.jobs.findIndex((job) => job.id === jobId);
}

export function setDefault<K extends keyof JobSettings>(key: K, value: JobSettings[K]) {
  setState("defaults", key, value);
  if (key === "model") void validateCurrentModel();
  schedulePersist();
}

export function setMaxConcurrency(n: number) {
  const clamped = Math.max(1, Math.min(10, n | 0));
  setState("maxConcurrency", clamped);
  schedulePersist();
  runScheduler();
}

export function setSelectedJob(jobId: string | null) {
  setState("selectedJobId", jobId);
  schedulePersist();
}

export function setInspectorTab(jobId: string, tab: InspectorTab) {
  const index = findJobIndex(jobId);
  if (index < 0) return;
  setState("jobs", index, "inspectorTab", tab);
  schedulePersist();
}

export function dismissQueueError() {
  setState("queueError", null);
}

export function enqueueFiles(paths: string[]): void {
  const normalized = paths.filter(Boolean);
  if (normalized.length === 0) return;

  const batchId = makeId();
  const settingsSnapshot: JobSettings = { ...state.defaults };
  const createdAtBase = Date.now();

  const newJobs: QueueJob[] = normalized.map((path, index) => ({
    id: makeId(),
    batchId,
    createdAt: createdAtBase + index,
    filePath: path,
    fileName: getBaseName(path),
    status: "queued",
    settings: settingsSnapshot,
    progress: null,
    result: null,
    annotatedResult: null,
    speakerNames: {},
    error: null,
    inspectorTab: "srt",
  }));

  setState(
    produce((draft) => {
      draft.jobs.push(...newJobs);
      if (!draft.selectedJobId) {
        draft.selectedJobId = newJobs[0].id;
      }
      draft.queueError = null;
    })
  );

  schedulePersist();
  runScheduler();
}

export async function cancelJob(jobId: string): Promise<void> {
  const index = findJobIndex(jobId);
  if (index < 0) return;
  const current = state.jobs[index];
  if (current.status === "completed") return;

  setState("jobs", index, (job) => ({
    ...job,
    status: "cancelled" as JobStatus,
    progress: null,
  }));

  schedulePersist();
  runScheduler();

  try {
    await invoke("cancel_job", { jobId });
  } catch {
    await invoke("cancel_inference").catch(() => {});
  }
}

export async function cancelAllRunning(): Promise<void> {
  const running = state.jobs.filter((job) => job.status === "running");
  await Promise.allSettled(running.map((job) => cancelJob(job.id)));
}

export function retryJob(jobId: string): void {
  const index = findJobIndex(jobId);
  if (index < 0) return;
  setState("jobs", index, (job) => ({
    ...job,
    status: "queued" as JobStatus,
    error: null,
    progress: null,
    result: null,
    annotatedResult: null,
  }));
  schedulePersist();
  runScheduler();
}

export function retryFailedJobs(): void {
  setState(
    produce((draft) => {
      draft.jobs.forEach((job) => {
        if (job.status === "failed") {
          job.status = "queued";
          job.error = null;
          job.progress = null;
          job.result = null;
          job.annotatedResult = null;
        }
      });
    })
  );
  schedulePersist();
  runScheduler();
}

export function removeJob(jobId: string): void {
  const target = state.jobs.find((job) => job.id === jobId);
  if (!target || target.status === "running") return;
  setState(
    produce((draft) => {
      draft.jobs = draft.jobs.filter((job) => job.id !== jobId);
      if (draft.selectedJobId === jobId) {
        draft.selectedJobId = draft.jobs[0]?.id ?? null;
      }
    })
  );
  schedulePersist();
}

export function clearCompleted(): void {
  setState(
    produce((draft) => {
      draft.jobs = draft.jobs.filter((job) => job.status !== "completed");
      if (!draft.jobs.find((job) => job.id === draft.selectedJobId)) {
        draft.selectedJobId = draft.jobs[0]?.id ?? null;
      }
    })
  );
  schedulePersist();
}

export function clearAll(): void {
  const running = state.jobs.filter((j) => j.status === "running");
  if (running.length > 0) return; // caller should cancel running jobs first
  setState(
    produce((draft) => {
      draft.jobs = [];
      draft.selectedJobId = null;
    })
  );
  schedulePersist();
}

export function updateSpeakerName(jobId: string, speakerId: number, value: string): void {
  const index = findJobIndex(jobId);
  if (index < 0) return;
  setState("jobs", index, "speakerNames", String(speakerId), value);
  schedulePersist();
}

let schedulerRunning = false;

function runScheduler(): void {
  if (schedulerRunning) return;
  schedulerRunning = true;
  try {
    while (true) {
      const runningCount = state.jobs.filter((job) => job.status === "running").length;
      const slots = state.maxConcurrency - runningCount;
      if (slots <= 0) break;
      const next = state.jobs.find((job) => job.status === "queued");
      if (!next) break;
      void runJob(next.id);
    }
  } finally {
    schedulerRunning = false;
  }
}

async function runJob(jobId: string): Promise<void> {
  const index = findJobIndex(jobId);
  if (index < 0) return;
  const job = state.jobs[index];
  if (job.status !== "queued") return;

  const initialMessage = job.settings.mode === "annotate" ? "Starting annotation..." : "Starting transcription...";

  setState("jobs", index, (current) => ({
    ...current,
    status: "running" as JobStatus,
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
  schedulePersist();

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
      const freshIndex = findJobIndex(jobId);
      if (freshIndex >= 0) {
        setState("jobs", freshIndex, (current): QueueJob => ({
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
          } as AnnotationProgress,
        }));
      }
    } else {
      const transcript = await invoke<TranscriptResult>("transcribe_audio_file", {
        filePath: job.filePath,
        model: job.settings.model,
        task: job.settings.task,
        language: job.settings.language,
        timestamps: true,
        jobId: job.id,
        batchId: job.batchId,
      });
      const freshIndex = findJobIndex(jobId);
      if (freshIndex >= 0) {
        setState("jobs", freshIndex, (current): QueueJob => ({
          ...current,
          status: "completed",
          result: transcript,
          progress: {
            job_id: current.id,
            phase: "completed",
            current: 1,
            total: 1,
            message: "Transcription complete",
          } as TranscriptionProgress,
        }));
      }
    }
  } catch (err) {
    const errStr = String(err);
    const freshIndex = findJobIndex(jobId);
    if (freshIndex < 0) return;
    const current = state.jobs[freshIndex];
    // User-initiated cancel already set status=cancelled; don't overwrite.
    if (current.status === "cancelled") {
      // leave as cancelled
    } else if (errStr.includes("cancelled") || errStr.includes("Operation cancelled")) {
      setState("jobs", freshIndex, (c) => ({
        ...c,
        status: "cancelled" as JobStatus,
        error: null,
        progress: null,
      }));
    } else {
      setState("jobs", freshIndex, (c) => ({
        ...c,
        status: "failed" as JobStatus,
        error: errStr,
      }));
    }
  } finally {
    schedulePersist();
    runScheduler();
  }
}

// Reactive selectors
export const queueState = state;

export const selectedJob = createMemo(() =>
  state.jobs.find((job) => job.id === state.selectedJobId) ?? null
);

export const runningCount = createMemo(() =>
  state.jobs.filter((job) => job.status === "running").length
);

export const queuedCount = createMemo(() =>
  state.jobs.filter((job) => job.status === "queued").length
);

export const completedCount = createMemo(() =>
  state.jobs.filter((job) => job.status === "completed").length
);

export const failedCount = createMemo(() =>
  state.jobs.filter((job) => job.status === "failed").length
);

export const cancelledCount = createMemo(() =>
  state.jobs.filter((job) => job.status === "cancelled").length
);

export const sortedJobs = createMemo(() =>
  state.jobs.slice().sort((a, b) => a.createdAt - b.createdAt)
);

// Kick the scheduler once on module load in case persisted state contains queued jobs.
// Safe: runJob guards against non-queued status and concurrency limit.
queueMicrotask(() => runScheduler());
