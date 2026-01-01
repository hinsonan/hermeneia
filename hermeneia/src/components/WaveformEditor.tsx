import { Component, onMount, onCleanup, createEffect } from "solid-js";
import type { WaveformPeaks, TrimSelection } from "../types/audio";
import "./WaveformEditor.css";

interface WaveformEditorProps {
  peaks: WaveformPeaks;
  selection: TrimSelection;
  onSelectionChange: (start: number, end: number) => void;
}

const WaveformEditor: Component<WaveformEditorProps> = (props) => {
  let canvasRef: HTMLCanvasElement | undefined;
  let containerRef: HTMLDivElement | undefined;

  // Dragging state
  let isDragging = false;
  let dragHandle: "start" | "end" | "none" = "none";

  // Draw waveform on canvas
  const drawWaveform = () => {
    if (!canvasRef || !containerRef) return;

    const canvas = canvasRef;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    // Set canvas size to match container
    const rect = containerRef.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    ctx.scale(dpr, dpr);

    const width = rect.width;
    const height = rect.height;

    // Clear canvas
    ctx.fillStyle = getComputedStyle(document.documentElement)
      .getPropertyValue("--parchment");
    ctx.fillRect(0, 0, width, height);

    // Draw waveform
    const peaks = props.peaks;
    const numPeaks = peaks.num_peaks;
    const barWidth = width / numPeaks;
    const centerY = height / 2;
    const waveformColor = getComputedStyle(document.documentElement)
      .getPropertyValue("--ink-dark");

    ctx.fillStyle = waveformColor;

    for (let i = 0; i < numPeaks; i++) {
      const x = i * barWidth;
      const minPeak = peaks.min_peaks[i];
      const maxPeak = peaks.max_peaks[i];

      // Scale peaks to canvas height (accounting for amplitude range -1 to 1)
      const minY = centerY + (minPeak * centerY * 0.9);
      const maxY = centerY + (maxPeak * centerY * 0.9);
      const barHeight = Math.abs(maxY - minY);

      ctx.fillRect(x, Math.min(minY, maxY), Math.max(barWidth * 0.8, 1), barHeight);
    }

    // Draw selection overlay
    const selection = props.selection;
    const duration = peaks.duration_seconds;
    const startX = (selection.start / duration) * width;
    const endX = (selection.end / duration) * width;

    // Dimmed regions outside selection
    ctx.fillStyle = "rgba(0, 0, 0, 0.3)";
    ctx.fillRect(0, 0, startX, height);
    ctx.fillRect(endX, 0, width - endX, height);

    // Selection handles
    const handleWidth = 8;
    const burgundy = getComputedStyle(document.documentElement)
      .getPropertyValue("--burgundy");

    ctx.fillStyle = burgundy;
    ctx.fillRect(startX - handleWidth / 2, 0, handleWidth, height);
    ctx.fillRect(endX - handleWidth / 2, 0, handleWidth, height);

    // Time labels
    ctx.fillStyle = waveformColor;
    ctx.font = "12px 'Crimson Text', serif";
    ctx.textAlign = "left";
    ctx.fillText(`${selection.start.toFixed(2)}s`, startX + 4, 20);
    ctx.textAlign = "right";
    ctx.fillText(`${selection.end.toFixed(2)}s`, endX - 4, 20);
  };

  // Mouse event handlers for selection
  const handleMouseDown = (e: MouseEvent) => {
    if (!containerRef) return;

    const rect = containerRef.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const width = rect.width;
    const duration = props.peaks.duration_seconds;

    const startX = (props.selection.start / duration) * width;
    const endX = (props.selection.end / duration) * width;

    // Determine if clicking on a handle
    const handleThreshold = 10;
    if (Math.abs(x - startX) < handleThreshold) {
      isDragging = true;
      dragHandle = "start";
    } else if (Math.abs(x - endX) < handleThreshold) {
      isDragging = true;
      dragHandle = "end";
    }
  };

  const handleMouseMove = (e: MouseEvent) => {
    if (!isDragging || !containerRef) return;

    const rect = containerRef.getBoundingClientRect();
    const x = Math.max(0, Math.min(e.clientX - rect.left, rect.width));
    const duration = props.peaks.duration_seconds;
    const time = (x / rect.width) * duration;

    if (dragHandle === "start") {
      props.onSelectionChange(
        Math.min(time, props.selection.end - 0.1),
        props.selection.end
      );
    } else if (dragHandle === "end") {
      props.onSelectionChange(
        props.selection.start,
        Math.max(time, props.selection.start + 0.1)
      );
    }
  };

  const handleMouseUp = () => {
    isDragging = false;
    dragHandle = "none";
  };

  // Setup and cleanup
  onMount(() => {
    drawWaveform();
    window.addEventListener("resize", drawWaveform);
    window.addEventListener("mousemove", handleMouseMove);
    window.addEventListener("mouseup", handleMouseUp);
  });

  onCleanup(() => {
    window.removeEventListener("resize", drawWaveform);
    window.removeEventListener("mousemove", handleMouseMove);
    window.removeEventListener("mouseup", handleMouseUp);
  });

  // Redraw when peaks or selection changes
  createEffect(() => {
    props.peaks;
    props.selection;
    drawWaveform();
  });

  return (
    <div class="waveform-editor" ref={containerRef}>
      <canvas
        ref={canvasRef}
        onMouseDown={handleMouseDown}
        class="waveform-canvas"
      />
    </div>
  );
};

export default WaveformEditor;
