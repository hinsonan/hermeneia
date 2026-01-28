import { Component, createSignal, onMount, onCleanup } from "solid-js";
import { getCurrentWindow } from "@tauri-apps/api/window";
import { open } from "@tauri-apps/plugin-dialog";
import "./TextFileUploader.css";

interface TextFileUploaderProps {
  onFileSelected: (filePath: string) => void;
}

const TextFileUploader: Component<TextFileUploaderProps> = (props) => {
  const [isDragging, setIsDragging] = createSignal(false);
  let unlistenDrop: (() => void) | undefined;
  let unlistenHover: (() => void) | undefined;

  onMount(async () => {
    const appWindow = getCurrentWindow();

    // Listen for file drop events
    unlistenDrop = await appWindow.listen<{ paths: string[] }>('tauri://drag-drop', (event) => {
      if (event.payload.paths && event.payload.paths.length > 0) {
        const filePath = event.payload.paths[0];
        // Only accept .txt and .srt files
        if (filePath.toLowerCase().endsWith('.txt') || filePath.toLowerCase().endsWith('.srt')) {
          setIsDragging(false);
          props.onFileSelected(filePath);
        }
      }
    });

    // Listen for drag hover events
    unlistenHover = await appWindow.listen('tauri://drag', () => {
      setIsDragging(true);
    });
  });

  onCleanup(() => {
    if (unlistenDrop) unlistenDrop();
    if (unlistenHover) unlistenHover();
  });

  // Handle click to open file picker
  const handleClick = async () => {
    try {
      const selected = await open({
        multiple: false,
        filters: [{
          name: "Text Files",
          extensions: ["txt", "srt"],
        }],
      });

      if (selected && typeof selected === "string") {
        props.onFileSelected(selected);
      }
    } catch (err) {
      console.error('Error opening file picker:', err);
    }
  };

  return (
    <div
      class={`text-file-uploader ${isDragging() ? "dragging" : ""}`}
      onClick={handleClick}
    >
      <div class="upload-icon">
        <svg viewBox="0 0 24 24" width="64" height="64">
          <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
          <polyline points="14 2 14 8 20 8" />
          <line x1="16" y1="13" x2="8" y2="13" />
          <line x1="16" y1="17" x2="8" y2="17" />
          <polyline points="10 9 9 9 8 9" />
        </svg>
      </div>
      <h3>Drop Text File Here or Click to Browse</h3>
      <p>Supports TXT (plain text) and SRT (subtitles)</p>
    </div>
  );
};

export default TextFileUploader;
