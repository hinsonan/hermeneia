import { createMemo } from "solid-js";
import { createStore, produce } from "solid-js/store";
import { invoke } from "@tauri-apps/api/core";
import { listen, UnlistenFn } from "@tauri-apps/api/event";
import type { DownloadProgress } from "../types/models";
import type {
  TranslationEngineMetadata,
  TranslationEngineTier,
  TextTranslationResult,
  TranslationFailureType,
  TranslationJobSettings,
  TranslationJobStatus,
  TranslationProgress,
  TranslationQueueJob,
  TranslationStrategy,
  InferenceRuntimeLimits,
} from "../types/translation";

interface TranslationQueueState {
  jobs: TranslationQueueJob[];
  selectedJobId: string | null;
  maxConcurrency: number;
  maxConcurrencyLimit: number;
  startArmed: boolean;
  defaults: TranslationJobSettings;
  queueError: string | null;
  listenersInitialized: boolean;
}

const STORAGE_KEY = "hermeneia-translation-job-queue";

const DEFAULT_SETTINGS: TranslationJobSettings = {
  sourceLang: "en",
  targetLang: "es",
  strategy: "auto",
};

const FAST_ENGINE_LABEL = "Fast model";
const FAST_ENGINE_MESSAGE = "Using fast translation model for this language pair.";
const UNIVERSAL_ENGINE_LABEL = "Universal model";
const UNIVERSAL_ENGINE_MESSAGE = "Using a slower larger general model for broader language coverage.";

interface LegacyTranslationJobSettings {
  sourceLang?: unknown;
  targetLang?: unknown;
  strategy?: unknown;
  allowMadlad?: unknown;
}

interface ResolveTranslationModelMetadata {
  modelId?: unknown;
  modelName?: unknown;
  model_id?: unknown;
  model_name?: unknown;
  tier?: unknown;
  engineTier?: unknown;
  engine_tier?: unknown;
  label?: unknown;
  engineLabel?: unknown;
  engine_label?: unknown;
  message?: unknown;
  engineMessage?: unknown;
  engine_message?: unknown;
  speedLabel?: unknown;
  speed_label?: unknown;
  userHint?: unknown;
  user_hint?: unknown;
}

type ResolveTranslationModelResponse =
  | [string, string]
  | ResolveTranslationModelMetadata;

function makeId(): string {
  if (typeof crypto !== "undefined" && typeof crypto.randomUUID === "function") {
    return crypto.randomUUID();
  }
  return `${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

function getBaseName(path: string): string {
  return path.split("/").pop() || path.split("\\").pop() || path;
}

function normalizeStatus(status: unknown): TranslationJobStatus {
  if (
    status === "queued"
    || status === "waiting_resources"
    || status === "downloading_model"
    || status === "loading_model"
    || status === "running"
    || status === "cancelling"
    || status === "completed"
    || status === "failed"
    || status === "cancelled"
  ) {
    return status;
  }
  return "queued";
}

function isActiveTranslationStatus(status: TranslationJobStatus): boolean {
  return status === "waiting_resources"
    || status === "downloading_model"
    || status === "loading_model"
    || status === "running"
    || status === "cancelling";
}

function normalizeFailureType(value: unknown): TranslationFailureType | null {
  if (
    value === "oom"
    || value === "model_load"
    || value === "transient"
    || value === "unsupported"
    || value === "cancelled"
    || value === "unknown"
  ) {
    return value;
  }
  return null;
}

function normalizeRetryCount(value: unknown): number {
  const parsed = typeof value === "number" ? value : Number(value);
  if (!Number.isFinite(parsed) || parsed < 0) return 0;
  return Math.floor(parsed);
}

function normalizeStrategy(value: unknown, legacyAllowMadlad?: unknown): TranslationStrategy {
  if (value === "auto" || value === "fast_only" || value === "universal") {
    return value;
  }
  if (typeof legacyAllowMadlad === "boolean") {
    return legacyAllowMadlad ? "auto" : "fast_only";
  }
  return DEFAULT_SETTINGS.strategy;
}

function normalizeEngineTier(value: unknown): TranslationEngineTier | null {
  if (value === "fast" || value === "universal") return value;
  return null;
}

function inferEngineTier(modelId: string, modelName: string, strategy: TranslationStrategy): TranslationEngineTier {
  const text = `${modelId} ${modelName}`.toLowerCase();
  if (text.includes("madlad") || text.includes("nllb") || text.includes("m2m")) {
    return "universal";
  }
  if (strategy === "universal") return "universal";
  return "fast";
}

function createEngineMetadata(
  tier: TranslationEngineTier,
  modelId: string,
  modelName: string,
  label?: unknown,
  message?: unknown
): TranslationEngineMetadata {
  return {
    tier,
    modelId,
    modelName,
    label: typeof label === "string" && label.trim().length > 0
      ? label
      : (tier === "fast" ? FAST_ENGINE_LABEL : UNIVERSAL_ENGINE_LABEL),
    message: typeof message === "string" && message.trim().length > 0
      ? message
      : (tier === "fast" ? FAST_ENGINE_MESSAGE : UNIVERSAL_ENGINE_MESSAGE),
  };
}

function normalizeEngineMetadata(value: unknown): TranslationEngineMetadata | null {
  if (!value || typeof value !== "object") return null;
  const record = value as {
    tier?: unknown;
    engineTier?: unknown;
    engine_tier?: unknown;
    modelId?: unknown;
    model_id?: unknown;
    modelName?: unknown;
    model_name?: unknown;
    label?: unknown;
    speedLabel?: unknown;
    speed_label?: unknown;
    message?: unknown;
    userHint?: unknown;
    user_hint?: unknown;
  };
  const modelId = typeof record.modelId === "string"
    ? record.modelId
    : typeof record.model_id === "string"
      ? record.model_id
      : null;
  const modelName = typeof record.modelName === "string"
    ? record.modelName
    : typeof record.model_name === "string"
      ? record.model_name
      : modelId;
  if (!modelId || !modelName) return null;

  const tier = normalizeEngineTier(record.tier)
    || normalizeEngineTier(record.engineTier)
    || normalizeEngineTier(record.engine_tier);
  if (!tier) return null;
  const label = record.label ?? record.speedLabel ?? record.speed_label;
  const message = record.message ?? record.userHint ?? record.user_hint;
  return createEngineMetadata(tier, modelId, modelName, label, message);
}

function strategyToLegacyAllowMadlad(strategy: TranslationStrategy): boolean {
  return strategy !== "fast_only";
}

function normalizeJobSettings(settings: unknown): TranslationJobSettings {
  const raw = (settings && typeof settings === "object" ? settings : {}) as LegacyTranslationJobSettings;
  return {
    sourceLang: typeof raw.sourceLang === "string" && raw.sourceLang.trim().length > 0
      ? raw.sourceLang
      : DEFAULT_SETTINGS.sourceLang,
    targetLang: typeof raw.targetLang === "string" && raw.targetLang.trim().length > 0
      ? raw.targetLang
      : DEFAULT_SETTINGS.targetLang,
    strategy: normalizeStrategy(raw.strategy, raw.allowMadlad),
  };
}

function normalizeResolvedModel(
  response: ResolveTranslationModelResponse,
  strategy: TranslationStrategy
): { modelId: string; modelName: string; engine: TranslationEngineMetadata } {
  if (Array.isArray(response)) {
    const modelId = typeof response[0] === "string" ? response[0] : "";
    const modelName = typeof response[1] === "string" ? response[1] : modelId;
    if (!modelId) {
      throw new Error("resolve_translation_model returned invalid model metadata");
    }
    const tier = inferEngineTier(modelId, modelName, strategy);
    return {
      modelId,
      modelName,
      engine: createEngineMetadata(tier, modelId, modelName),
    };
  }

  const modelId = typeof response.modelId === "string"
    ? response.modelId
    : typeof response.model_id === "string"
      ? response.model_id
      : "";
  const modelName = typeof response.modelName === "string"
    ? response.modelName
    : typeof response.model_name === "string"
      ? response.model_name
      : modelId;

  if (!modelId) {
    throw new Error("resolve_translation_model returned invalid model metadata");
  }

  const tier = normalizeEngineTier(response.tier)
    || normalizeEngineTier(response.engineTier)
    || normalizeEngineTier(response.engine_tier)
    || inferEngineTier(modelId, modelName, strategy);
  const label = response.label ?? response.engineLabel ?? response.engine_label ?? response.speedLabel ?? response.speed_label;
  const message = response.message ?? response.engineMessage ?? response.engine_message ?? response.userHint ?? response.user_hint;

  return {
    modelId,
    modelName,
    engine: createEngineMetadata(tier, modelId, modelName, label, message),
  };
}

function isTerminalStatus(status: TranslationJobStatus): boolean {
  return status === "completed" || status === "failed" || status === "cancelled";
}

function loadPersistedState(): Partial<TranslationQueueState> {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return {};

    const parsed = JSON.parse(raw) as {
      jobs?: Array<TranslationQueueJob & {
        settings?: LegacyTranslationJobSettings;
        engine?: unknown;
      }>;
      selectedJobId?: string | null;
      maxConcurrency?: number;
      defaults?: Partial<TranslationJobSettings> & { allowMadlad?: unknown };
    };

    const parsedMaxConcurrency = Number(parsed.maxConcurrency);
    const maxConcurrency = Number.isFinite(parsedMaxConcurrency)
      ? Math.max(1, Math.min(8, Math.floor(parsedMaxConcurrency)))
      : 2;

    const defaults = normalizeJobSettings(parsed.defaults);

    const jobs = (parsed.jobs || []).map((job) => {
      const status = normalizeStatus(job.status);
      const orphaned = status === "running"
        || status === "cancelling"
        || status === "loading_model"
        || status === "downloading_model"
        || status === "waiting_resources";

      return {
        ...job,
        settings: normalizeJobSettings(job.settings),
        engine: normalizeEngineMetadata(job.engine),
        status: orphaned ? "cancelled" : status,
        progress: orphaned ? null : job.progress,
        downloadProgress: orphaned ? null : (job.downloadProgress ?? null),
        error: orphaned ? null : (job.error ?? null),
        retryCount: normalizeRetryCount(job.retryCount),
        lastFailureType: normalizeFailureType(job.lastFailureType),
      } as TranslationQueueJob;
    });

    return {
      jobs,
      selectedJobId: parsed.selectedJobId ?? null,
      maxConcurrency: Math.max(1, Math.min(4, maxConcurrency)),
      defaults,
    };
  } catch {
    return {};
  }
}

const persisted = loadPersistedState();

const [state, setState] = createStore<TranslationQueueState>({
  jobs: persisted.jobs ?? [],
  selectedJobId: persisted.selectedJobId ?? null,
  maxConcurrency: Math.max(1, Math.min(4, persisted.maxConcurrency ?? 2)),
  maxConcurrencyLimit: 4,
  startArmed: false,
  defaults: persisted.defaults ?? { ...DEFAULT_SETTINGS },
  queueError: null,
  listenersInitialized: false,
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

let unlistenTranslation: UnlistenFn | null = null;
let unlistenDownload: UnlistenFn | null = null;
let initPromise: Promise<void> | null = null;
let initGeneration = 0;
let concurrencyLimitRefreshSeq = 0;

let schedulerPaused = false;
let schedulerRunning = false;

const runControllers = new Map<string, AbortController>();
const removeWhenTerminal = new Set<string>();

interface DownloadWaiter {
  jobId: string;
  modelId: string;
  resolve: (release: () => void) => void;
  reject: (reason: Error) => void;
}

let downloadLockOwnerJobId: string | null = null;
let downloadLockModelId: string | null = null;
const downloadWaiters: DownloadWaiter[] = [];

function findJobIndex(jobId: string): number {
  return state.jobs.findIndex((job) => job.id === jobId);
}

function isCancelJobCommandUnavailable(err: unknown): boolean {
  const message = String(err).toLowerCase();
  return message.includes("unknown command `cancel_job`")
    || message.includes("unknown command: cancel_job")
    || message.includes("command cancel_job not found")
    || (message.includes("not found") && message.includes("cancel_job"));
}

function hasActiveTranslationJob(jobId: string): boolean {
  return state.jobs.some((job) => job.id === jobId && isActiveTranslationStatus(job.status));
}

function restoreFailedCancellation(jobId: string, previous: TranslationQueueJob): void {
  removeWhenTerminal.delete(jobId);

  const index = findJobIndex(jobId);
  if (index < 0 || state.jobs[index].status !== "cancelling") return;

  setState("jobs", index, (job) => ({
    ...job,
    status: previous.status,
    progress: previous.progress,
    downloadProgress: previous.downloadProgress,
  }));
}

function removeTranslationJobIfClearPending(jobId: string): boolean {
  if (!removeWhenTerminal.delete(jobId)) return false;

  setState(
    produce((draft) => {
      draft.jobs = draft.jobs.filter((job) => job.id !== jobId);
      if (draft.selectedJobId === jobId) {
        draft.selectedJobId = draft.jobs[0]?.id ?? null;
      }
    })
  );

  return true;
}

function isAbortError(err: unknown): boolean {
  const message = String(err).toLowerCase();
  return message.includes("aborted") || message.includes("aborterror");
}

function classifyFailure(err: unknown): TranslationFailureType {
  const message = String(err).toLowerCase();
  if (message.includes("cancelled") || message.includes("operation cancelled")) {
    return "cancelled";
  }
  if (
    message.includes("out of memory")
    || message.includes("cuda out of memory")
    || message.includes("insufficient memory")
    || message.includes("oom")
  ) {
    return "oom";
  }
  if (
    message.includes("loading model")
    || message.includes("load model")
    || message.includes("model initialization")
    || message.includes("tokenizer")
    || message.includes("weights")
  ) {
    return "model_load";
  }
  if (
    message.includes("unsupported")
    || message.includes("not supported")
    || message.includes("language pair")
    || message.includes("invalid language")
  ) {
    return "unsupported";
  }
  if (
    message.includes("timed out")
    || message.includes("timeout")
    || message.includes("network")
    || message.includes("connection")
    || message.includes("temporar")
    || message.includes("already in progress")
  ) {
    return "transient";
  }
  return "unknown";
}

function withSchedulerPaused<T>(fn: () => Promise<T>): Promise<T> {
  schedulerPaused = true;
  return fn().finally(() => {
    schedulerPaused = false;
    runScheduler();
  });
}

function updateTranslationProgress(jobId: string, progress: TranslationProgress) {
  const index = findJobIndex(jobId);
  if (index < 0) return;
  const current = state.jobs[index];
  if (isTerminalStatus(current.status) || current.status === "cancelling") return;

  setState("jobs", index, (job) => ({
    ...job,
    progress,
    status: progress.phase === "waiting_resources"
      ? "waiting_resources"
      : progress.phase === "downloading_model"
        ? "downloading_model"
        : progress.phase === "loading_model"
          ? "loading_model"
          : progress.phase === "translating"
            ? "running"
            : job.status,
  }));
}

function updateDownloadProgress(payload: DownloadProgress) {
  if (!downloadLockOwnerJobId || !downloadLockModelId) return;
  if (payload.model_id !== downloadLockModelId) return;

  const index = findJobIndex(downloadLockOwnerJobId);
  if (index < 0) return;
  const current = state.jobs[index];
  if (isTerminalStatus(current.status) || current.status === "cancelling") return;

  setState("jobs", index, (job) => ({
    ...job,
    status: payload.phase === "downloading" ? "downloading_model" : job.status,
    downloadProgress: payload.phase === "complete" ? null : payload,
    progress: payload.phase === "downloading"
      ? {
          job_id: job.id,
          phase: "downloading_model",
          current: payload.bytes_downloaded,
          total: payload.bytes_total,
          message: `Downloading ${payload.model_name}...`,
        }
      : job.progress,
  }));
}

function promoteNextDownloadWaiter() {
  if (downloadLockOwnerJobId) return;
  while (downloadWaiters.length > 0) {
    const waiter = downloadWaiters.shift();
    if (!waiter) break;
    const index = findJobIndex(waiter.jobId);
    if (index < 0) continue;
    const status = state.jobs[index].status;
    if (status === "cancelled" || status === "failed" || status === "completed") continue;

    downloadLockOwnerJobId = waiter.jobId;
    downloadLockModelId = waiter.modelId;
    waiter.resolve(() => {
      if (downloadLockOwnerJobId === waiter.jobId) {
        downloadLockOwnerJobId = null;
        downloadLockModelId = null;
        promoteNextDownloadWaiter();
      }
    });
    break;
  }
}

async function acquireDownloadLock(jobId: string, modelId: string, signal: AbortSignal): Promise<() => void> {
  if (!downloadLockOwnerJobId) {
    downloadLockOwnerJobId = jobId;
    downloadLockModelId = modelId;
    return () => {
      if (downloadLockOwnerJobId === jobId) {
        downloadLockOwnerJobId = null;
        downloadLockModelId = null;
        promoteNextDownloadWaiter();
      }
    };
  }

  return new Promise<() => void>((resolve, reject) => {
    const waiter: DownloadWaiter = {
      jobId,
      modelId,
      resolve,
      reject,
    };

    const onAbort = () => {
      const idx = downloadWaiters.indexOf(waiter);
      if (idx >= 0) {
        downloadWaiters.splice(idx, 1);
      }
      reject(new Error("Aborted while waiting for model download lock"));
    };

    signal.addEventListener("abort", onAbort, { once: true });

    waiter.resolve = (release) => {
      signal.removeEventListener("abort", onAbort);
      if (signal.aborted) {
        release();
        reject(new Error("Aborted while waiting for model download lock"));
        return;
      }
      resolve(release);
    };

    waiter.reject = (reason) => {
      signal.removeEventListener("abort", onAbort);
      reject(reason);
    };

    downloadWaiters.push(waiter);
  });
}

async function ensureModelDownloaded(
  jobId: string,
  modelId: string,
  modelName: string,
  signal: AbortSignal
): Promise<void> {
  const isCached = await invoke<boolean>("is_model_cached", { modelId });
  if (isCached) return;

  const waitingIndex = findJobIndex(jobId);
  if (waitingIndex >= 0) {
    setState("jobs", waitingIndex, (job) => ({
      ...job,
      status: "waiting_resources",
      progress: {
        job_id: job.id,
        phase: "waiting_resources",
        current: null,
        total: null,
        message: "Waiting for model download slot...",
      },
      downloadProgress: null,
    }));
    schedulePersist();
  }

  const release = await acquireDownloadLock(jobId, modelId, signal);
  try {
    if (signal.aborted) {
      throw new Error("Aborted before model download start");
    }

    const index = findJobIndex(jobId);
    if (index >= 0) {
      setState("jobs", index, (job) => ({
        ...job,
        status: "downloading_model",
        progress: {
          job_id: job.id,
          phase: "downloading_model",
          current: null,
          total: null,
          message: `Downloading ${modelName}...`,
        },
      }));
      schedulePersist();
    }

    await invoke("download_model", { modelId, modelName });
  } finally {
    const currentIndex = findJobIndex(jobId);
    if (currentIndex >= 0) {
      setState("jobs", currentIndex, "downloadProgress", null);
    }
    release();
  }
}

function runScheduler(): void {
  if (schedulerPaused) return;
  if (schedulerRunning) return;
  if (!state.startArmed) return;
  schedulerRunning = true;
  try {
    while (true) {
      const activeCount = state.jobs.filter(
        (job) =>
          job.status === "running"
          || job.status === "loading_model"
          || job.status === "downloading_model"
          || job.status === "waiting_resources"
          || job.status === "cancelling"
      ).length;

      const slots = state.maxConcurrency - activeCount;
      if (slots <= 0) break;

      const next = state.jobs.find((job) => job.status === "queued");
      if (!next) break;
      void runJob(next.id);
    }

    const hasQueued = state.jobs.some((job) => job.status === "queued");
    const hasActive = state.jobs.some(
      (job) =>
        job.status === "waiting_resources"
        || job.status === "downloading_model"
        || job.status === "loading_model"
        || job.status === "running"
        || job.status === "cancelling"
    );
    if (!hasQueued && !hasActive && state.startArmed) {
      setState("startArmed", false);
    }
  } finally {
    schedulerRunning = false;
  }
}

async function runJob(jobId: string): Promise<void> {
  const index = findJobIndex(jobId);
  if (index < 0) return;
  const initial = state.jobs[index];
  if (initial.status !== "queued") return;

  const controller = new AbortController();
  runControllers.set(jobId, controller);

  setState("jobs", index, (job) => ({
    ...job,
    status: "waiting_resources",
    error: null,
    result: null,
    engine: null,
    downloadProgress: null,
    progress: {
      job_id: job.id,
      phase: "waiting_resources",
      current: null,
      total: null,
      message: "Preparing translation resources...",
    },
  }));
  schedulePersist();

  try {
    const freshIndex = findJobIndex(jobId);
    if (freshIndex < 0) return;
    const current = state.jobs[freshIndex];

    const resolvedModel = await invoke<ResolveTranslationModelResponse>("resolve_translation_model", {
      sourceLang: current.settings.sourceLang,
      targetLang: current.settings.targetLang,
      strategy: current.settings.strategy,
      allowMadlad: strategyToLegacyAllowMadlad(current.settings.strategy),
    });

    const { modelId, modelName, engine } = normalizeResolvedModel(resolvedModel, current.settings.strategy);

    const resolvedIndex = findJobIndex(jobId);
    if (resolvedIndex >= 0) {
      setState("jobs", resolvedIndex, (job) => ({
        ...job,
        engine,
      }));
      schedulePersist();
    }

    if (controller.signal.aborted) {
      throw new Error("Operation cancelled");
    }

    await ensureModelDownloaded(jobId, modelId, modelName, controller.signal);

    if (controller.signal.aborted) {
      throw new Error("Operation cancelled");
    }

    const loadingIndex = findJobIndex(jobId);
    if (loadingIndex >= 0) {
      setState("jobs", loadingIndex, (job) => ({
        ...job,
        status: "loading_model",
        progress: {
          job_id: job.id,
          phase: "loading_model",
          current: null,
          total: null,
          message: "Loading translation model...",
        },
      }));
      schedulePersist();
    }

    const runIndex = findJobIndex(jobId);
    if (runIndex < 0) return;
    const runJobSnapshot = state.jobs[runIndex];

    const result = await invoke<TextTranslationResult>("translate_text_file", {
      filePath: runJobSnapshot.filePath,
      sourceLang: runJobSnapshot.settings.sourceLang,
      targetLang: runJobSnapshot.settings.targetLang,
      strategy: runJobSnapshot.settings.strategy,
      allowMadlad: strategyToLegacyAllowMadlad(runJobSnapshot.settings.strategy),
      jobId: runJobSnapshot.id,
      batchId: runJobSnapshot.batchId,
    });

    const completedIndex = findJobIndex(jobId);
    if (completedIndex < 0) return;
    const completedJob = state.jobs[completedIndex];

    if (completedJob.status === "cancelling" || completedJob.status === "cancelled") {
      setState("jobs", completedIndex, (job) => ({
        ...job,
        status: "cancelled",
        error: null,
        progress: null,
        downloadProgress: null,
      }));
      removeTranslationJobIfClearPending(jobId);
    } else if (completedJob.status === "loading_model" || completedJob.status === "running") {
      setState("jobs", completedIndex, (job) => ({
        ...job,
        status: "completed",
        result,
        error: null,
        progress: {
          job_id: job.id,
          phase: "completed",
          current: 1,
          total: 1,
          message: "Translation complete",
        },
        downloadProgress: null,
      }));
    }
  } catch (err) {
    const errorMessage = String(err);
    const failureType = classifyFailure(err);
    const freshIndex = findJobIndex(jobId);
    if (freshIndex < 0) return;
    const current = state.jobs[freshIndex];

    if (current.status === "cancelled" || current.status === "completed" || current.status === "failed") {
      return;
    }

    if (current.status === "cancelling" || (isActiveTranslationStatus(current.status) && failureType === "cancelled")) {
      setState("jobs", freshIndex, (job) => ({
        ...job,
        status: "cancelled",
        error: null,
        progress: null,
        downloadProgress: null,
        lastFailureType: "cancelled",
      }));
      removeTranslationJobIfClearPending(jobId);
    } else if (isActiveTranslationStatus(current.status) && isAbortError(err)) {
      setState("jobs", freshIndex, (job) => ({
        ...job,
        status: "cancelled",
        error: null,
        progress: null,
        downloadProgress: null,
        lastFailureType: "cancelled",
      }));
      removeTranslationJobIfClearPending(jobId);
    } else if (isActiveTranslationStatus(current.status)) {
      setState("jobs", freshIndex, (job) => ({
        ...job,
        status: "failed",
        error: errorMessage,
        lastFailureType: failureType,
        progress: job.progress,
        downloadProgress: null,
      }));
      removeTranslationJobIfClearPending(jobId);
    }
  } finally {
    runControllers.delete(jobId);
    schedulePersist();
    runScheduler();
  }
}

export async function initTranslationJobQueue(): Promise<void> {
  if (state.listenersInitialized) return;

  if (initPromise) {
    return initPromise;
  }

  const generation = initGeneration;
  const promise = (async () => {
    let translationUnlisten: UnlistenFn | null = null;
    let downloadUnlisten: UnlistenFn | null = null;

    try {
      await refreshInferenceConcurrencyLimit();
      if (generation !== initGeneration) {
        return;
      }

      translationUnlisten = await listen<TranslationProgress>("translation-progress", (event) => {
        const payload = event.payload;
        if (!payload) return;

        if (payload.job_id) {
          updateTranslationProgress(payload.job_id, payload);
          return;
        }

        const active = state.jobs.filter(
          (job) =>
            job.status === "loading_model"
            || job.status === "running"
            || job.status === "waiting_resources"
        );
        if (active.length === 1) {
          updateTranslationProgress(active[0].id, {
            ...payload,
            job_id: active[0].id,
          });
        }
      });
      if (generation !== initGeneration) {
        translationUnlisten();
        return;
      }
      unlistenTranslation = translationUnlisten;

      downloadUnlisten = await listen<DownloadProgress>("download-progress", (event) => {
        const payload = event.payload;
        if (!payload?.model_id) return;
        updateDownloadProgress(payload);
      });
      if (generation !== initGeneration) {
        downloadUnlisten();
        return;
      }
      unlistenDownload = downloadUnlisten;

      setState("listenersInitialized", true);
      setState("queueError", null);
    } catch (err) {
      if (unlistenTranslation) {
        unlistenTranslation();
        unlistenTranslation = null;
      } else if (translationUnlisten) {
        translationUnlisten();
      }
      if (unlistenDownload) {
        unlistenDownload();
        unlistenDownload = null;
      } else if (downloadUnlisten) {
        downloadUnlisten();
      }
      if (generation === initGeneration) {
        setState("listenersInitialized", false);
        setState("queueError", `Translation queue initialization failed: ${String(err)}`);
      }
    }
  })().finally(() => {
    if (initPromise === promise) {
      initPromise = null;
    }
  });

  initPromise = promise;

  return initPromise;
}

export function teardownTranslationJobQueue() {
  initGeneration += 1;
  initPromise = null;
  if (unlistenTranslation) unlistenTranslation();
  if (unlistenDownload) unlistenDownload();
  unlistenTranslation = null;
  unlistenDownload = null;
  setState("listenersInitialized", false);
}

export function setTranslationDefault<K extends keyof TranslationJobSettings>(
  key: K,
  value: TranslationJobSettings[K]
) {
  setState("defaults", key, value);
  void refreshInferenceConcurrencyLimit();
  schedulePersist();
}

export function setTranslationMaxConcurrency(n: number) {
  const clamped = Math.max(1, Math.min(state.maxConcurrencyLimit, n | 0));
  setState("maxConcurrency", clamped);
  schedulePersist();
  runScheduler();
}

async function refreshInferenceConcurrencyLimit(): Promise<void> {
  const seq = ++concurrencyLimitRefreshSeq;
  const snapshot = { ...state.defaults };

  try {
    const limits = await invoke<InferenceRuntimeLimits>("recommend_inference_concurrency", {
      translationStrategy: snapshot.strategy,
      translationSourceLang: snapshot.sourceLang,
      translationTargetLang: snapshot.targetLang,
    });

    if (
      seq !== concurrencyLimitRefreshSeq
      || state.defaults.strategy !== snapshot.strategy
      || state.defaults.sourceLang !== snapshot.sourceLang
      || state.defaults.targetLang !== snapshot.targetLang
    ) {
      return;
    }

    const maxAllowed = Math.max(1, Math.min(4, limits.max_inference_concurrency | 0));
    setState("maxConcurrencyLimit", maxAllowed);
    if (state.maxConcurrency > maxAllowed) {
      setState("maxConcurrency", maxAllowed);
      schedulePersist();
    }
  } catch (err) {
    if (seq === concurrencyLimitRefreshSeq) {
      console.warn("Failed to refresh translation inference concurrency limit:", err);
    }
  }
}

export function setSelectedTranslationJob(jobId: string | null) {
  setState("selectedJobId", jobId);
  schedulePersist();
}

export function dismissTranslationQueueError() {
  setState("queueError", null);
}

export function enqueueTranslationFiles(
  paths: string[],
  settingsOverride?: Partial<TranslationJobSettings>
): void {
  const filtered = paths.filter((path) => {
    const lower = path.toLowerCase();
    return lower.endsWith(".txt") || lower.endsWith(".srt");
  });
  if (filtered.length === 0) return;

  const settings: TranslationJobSettings = {
    ...state.defaults,
    ...settingsOverride,
  };

  const batchId = makeId();
  const createdAtBase = Date.now();

  const jobs: TranslationQueueJob[] = filtered.map((filePath, index) => ({
    id: makeId(),
    batchId,
    createdAt: createdAtBase + index,
    filePath,
    fileName: getBaseName(filePath),
    status: "queued",
    settings,
    engine: null,
    progress: null,
    downloadProgress: null,
    result: null,
    error: null,
    retryCount: 0,
    lastFailureType: null,
  }));

  setState(
    produce((draft) => {
      draft.jobs.push(...jobs);
      draft.startArmed = false;
      if (!draft.selectedJobId) {
        draft.selectedJobId = jobs[0].id;
      }
      draft.queueError = null;
    })
  );

  schedulePersist();
  runScheduler();
}

export async function cancelTranslationJob(jobId: string): Promise<void> {
  const index = findJobIndex(jobId);
  if (index < 0) return;
  const current = state.jobs[index];

  if (current.status === "completed" || current.status === "cancelled") return;
  if (current.status === "cancelling") return;

  if (current.status === "queued" || current.status === "failed") {
    setState("jobs", index, (job) => ({
      ...job,
      status: "cancelled",
      error: null,
      progress: null,
      downloadProgress: null,
      lastFailureType: "cancelled",
    }));
    schedulePersist();
    runScheduler();
    return;
  }

  setState("jobs", index, (job) => ({
    ...job,
    status: "cancelling",
    progress: job.progress
      ? {
          ...job.progress,
          message: "Cancelling...",
        }
      : {
          job_id: job.id,
          phase: "waiting_resources",
          current: null,
          total: null,
          message: "Cancelling...",
        },
  }));
  schedulePersist();

  const controller = runControllers.get(jobId);

  const canUseBackendCancel =
    current.status === "loading_model"
    || current.status === "running"
    || current.status === "cancelling";

  try {
    if (current.status === "downloading_model" && downloadLockOwnerJobId === jobId) {
      await invoke("cancel_download");
    }

    if (canUseBackendCancel) {
      const cancelled = await invoke<boolean>("cancel_job", { jobId });
      if (!cancelled) {
        await invoke("cancel_inference");
      }
    }

    controller?.abort();
  } catch (err) {
    if (isCancelJobCommandUnavailable(err)) {
      try {
        await invoke("cancel_inference");
        controller?.abort();
      } catch (fallbackErr) {
        setState("queueError", `Failed to cancel job: ${String(fallbackErr)}`);
        restoreFailedCancellation(jobId, current);
      }
    } else {
      setState("queueError", `Failed to cancel job: ${String(err)}`);
      restoreFailedCancellation(jobId, current);
    }
  } finally {
    schedulePersist();
    if (!hasActiveTranslationJob(jobId)) {
      runScheduler();
    }
  }
}

export async function clearAllTranslationJobs(): Promise<void> {
  await withSchedulerPaused(async () => {
    const active = state.jobs.filter(
      (job) =>
        job.status === "waiting_resources"
        || job.status === "downloading_model"
        || job.status === "loading_model"
        || job.status === "running"
        || job.status === "cancelling"
    );

    active.forEach((job) => removeWhenTerminal.add(job.id));
    await Promise.allSettled(active.map((job) => cancelTranslationJob(job.id)));

    setState(
      produce((draft) => {
        const activeIds = new Set(active.map((job) => job.id));
        draft.jobs = draft.jobs.filter(
          (job) => activeIds.has(job.id) && isActiveTranslationStatus(job.status)
        );
        if (!draft.jobs.find((job) => job.id === draft.selectedJobId)) {
          draft.selectedJobId = draft.jobs[0]?.id ?? null;
        }
        draft.startArmed = false;
      })
    );
    schedulePersist();
  });
}

export function startQueuedTranslationJobs(): void {
  if (!state.jobs.some((job) => job.status === "queued")) return;
  setState("startArmed", true);
  runScheduler();
}

export function clearCompletedTranslationJobs(): void {
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

export async function removeTranslationJob(jobId: string): Promise<void> {
  const index = findJobIndex(jobId);
  if (index < 0) return;
  const target = state.jobs[index];

  if (isActiveTranslationStatus(target.status)) {
    removeWhenTerminal.add(jobId);
    void cancelTranslationJob(jobId);
    return;
  }

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

export function retryFailedTranslationJob(jobId: string): void {
  const index = findJobIndex(jobId);
  if (index < 0) return;

  const now = Date.now();
  const newId = makeId();

  setState(
    produce((draft) => {
      const job = draft.jobs[index];
      if (!job || job.status !== "failed") return;

      const oldId = job.id;
      job.id = newId;
      job.createdAt = now;
      job.status = "queued";
      job.error = null;
      job.progress = null;
      job.downloadProgress = null;
      job.result = null;
      job.lastFailureType = null;
      job.retryCount = normalizeRetryCount(job.retryCount) + 1;
      draft.startArmed = false;

      if (draft.selectedJobId === oldId) {
        draft.selectedJobId = newId;
      }
    })
  );

  schedulePersist();
  runScheduler();
}

export function retryAllFailedTranslationJobs(): void {
  const now = Date.now();
  setState(
    produce((draft) => {
      const selected = draft.selectedJobId;
      const selectedMap = new Map<string, string>();
      let offset = 0;

      draft.jobs.forEach((job) => {
        if (job.status !== "failed") return;

        const oldId = job.id;
        const newId = makeId();

        job.id = newId;
        job.createdAt = now + offset;
        offset += 1;
        job.status = "queued";
        job.error = null;
        job.progress = null;
        job.downloadProgress = null;
        job.result = null;
        job.lastFailureType = null;
        job.retryCount = normalizeRetryCount(job.retryCount) + 1;

        selectedMap.set(oldId, newId);
      });

      if (selectedMap.size > 0) {
        draft.startArmed = false;
      }

      if (selected && selectedMap.has(selected)) {
        draft.selectedJobId = selectedMap.get(selected) || null;
      }
    })
  );
  schedulePersist();
  runScheduler();
}

export const translationQueueState = state;

export const selectedTranslationJob = createMemo(() =>
  state.jobs.find((job) => job.id === state.selectedJobId) ?? null
);

export const sortedTranslationJobs = createMemo(() =>
  state.jobs.slice().sort((a, b) => a.createdAt - b.createdAt)
);

export const translationQueuedCount = createMemo(() =>
  state.jobs.filter((job) => job.status === "queued").length
);

export const translationActiveCount = createMemo(() =>
  state.jobs.filter(
    (job) =>
      job.status === "waiting_resources"
      || job.status === "downloading_model"
      || job.status === "loading_model"
      || job.status === "running"
      || job.status === "cancelling"
  ).length
);

export const translationCompletedCount = createMemo(() =>
  state.jobs.filter((job) => job.status === "completed").length
);

export const translationFailedCount = createMemo(() =>
  state.jobs.filter((job) => job.status === "failed").length
);

export const translationCancelledCount = createMemo(() =>
  state.jobs.filter((job) => job.status === "cancelled").length
);

queueMicrotask(() => {
  void initTranslationJobQueue();
});
