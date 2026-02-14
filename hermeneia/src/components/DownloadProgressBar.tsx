import { Component, Show } from "solid-js";
import type { DownloadProgress } from "../types/models";
import "./DownloadProgressBar.css";

interface DownloadProgressBarProps {
  progress: DownloadProgress | null;
  onCancel?: () => void;
}

function formatBytes(bytes: number): string {
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(0)} KB`;
  if (bytes < 1024 * 1024 * 1024) return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  return `${(bytes / (1024 * 1024 * 1024)).toFixed(2)} GB`;
}

const DownloadProgressBar: Component<DownloadProgressBarProps> = (props) => {
  const percentage = () => {
    const p = props.progress;
    if (!p || !p.bytes_total || p.bytes_total === 0) return null;
    return Math.min(100, Math.round((p.bytes_downloaded / p.bytes_total) * 100));
  };

  const fileProgress = () => {
    const p = props.progress;
    if (!p) return "";
    return `File ${p.file_index + 1} of ${p.total_files}`;
  };

  const bytesDisplay = () => {
    const p = props.progress;
    if (!p) return "";
    const downloaded = formatBytes(p.bytes_downloaded);
    if (p.bytes_total) {
      return `${downloaded} / ${formatBytes(p.bytes_total)}`;
    }
    return downloaded;
  };

  const isActive = () => {
    const p = props.progress;
    return p && (p.phase === "downloading" || p.phase === "complete" || p.phase === "cancelled");
  };

  return (
    <Show when={isActive()}>
      <div class="download-progress">
        <Show when={props.progress?.phase === "downloading"}>
          <div class="download-progress-header">
            <div class="download-progress-info">
              <span class="download-progress-name">{props.progress?.model_name}</span>
              <span class="download-progress-file">
                {props.progress?.file_name} ({fileProgress()})
              </span>
            </div>
            <Show when={props.onCancel}>
              <button class="download-cancel-btn" onClick={props.onCancel}>
                Cancel
              </button>
            </Show>
          </div>

          <div class="download-progress-bar-container">
            <Show
              when={percentage() !== null}
              fallback={<div class="progress-bar-indeterminate" />}
            >
              <div
                class="progress-bar-fill"
                style={{ width: `${percentage()}%` }}
              />
            </Show>
          </div>

          <div class="download-progress-footer">
            <span class="download-progress-bytes">{bytesDisplay()}</span>
            <Show when={percentage() !== null}>
              <span class="download-progress-pct">{percentage()}%</span>
            </Show>
          </div>
        </Show>

        <Show when={props.progress?.phase === "complete"}>
          <div class="download-progress-status complete">
            Download complete
          </div>
        </Show>

        <Show when={props.progress?.phase === "cancelled"}>
          <div class="download-progress-status cancelled">
            Download cancelled
          </div>
        </Show>
      </div>
    </Show>
  );
};

export default DownloadProgressBar;
