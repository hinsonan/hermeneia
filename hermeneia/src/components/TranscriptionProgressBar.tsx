import { Component, Show } from "solid-js";
import type { AnnotationProgress, TranscriptionProgress } from "../types/transcription";
import "./TranscriptionProgressBar.css";

type JobProgress = TranscriptionProgress | AnnotationProgress;

interface TranscriptionProgressBarProps {
  progress: JobProgress | null;
}

const TranscriptionProgressBar: Component<TranscriptionProgressBarProps> = (props) => {
  const percentage = () => {
    const p = props.progress;
    if (!p) {
      return 0;
    }

    if (p.current === null || p.total === null || p.total === 0) {
      return 0;
    }

    return Math.min(100, Math.round((p.current / p.total) * 100));
  };

  const isIndeterminate = () => {
    const p = props.progress;
    if (!p) {
      return false;
    }

    if ("indeterminate" in p && typeof p.indeterminate === "boolean") {
      return p.indeterminate;
    }

    return p.phase === "loading_model";
  };

  return (
    <Show when={props.progress}>
      <div class="transcription-progress">
        <div class="progress-header">
          <Show when={isIndeterminate()}>
            <span class="progress-dot"></span>
          </Show>
          <span class="progress-message">{props.progress?.message}</span>
          <Show when={!isIndeterminate()}>
            <span class="progress-percentage">{percentage()}%</span>
          </Show>
        </div>

        <div class={`progress-bar-container ${isIndeterminate() ? "indeterminate" : ""}`}>
          <Show when={isIndeterminate()} fallback={
            <div
              class="progress-bar-fill"
              style={{ width: `${percentage()}%` }}
            />
          }>
            <div class="progress-bar-indeterminate" />
          </Show>
        </div>
      </div>
    </Show>
  );
};

export default TranscriptionProgressBar;
