import { Component, createSignal, onMount, onCleanup } from "solid-js";
import { getCurrentWindow } from "@tauri-apps/api/window";
import { open } from "@tauri-apps/plugin-dialog";
import "./FileUploader.css";

interface FileUploaderProps {
  onFileSelected: (filePath: string) => void;
}

const FileUploader: Component<FileUploaderProps> = (props) => {
  const [isDragging, setIsDragging] = createSignal(false);
  let unlistenDrop: (() => void) | undefined;
  let unlistenHover: (() => void) | undefined;

  onMount(async () => {
    const appWindow = getCurrentWindow();

    // Listen for file drop events
    unlistenDrop = await appWindow.listen<{ paths: string[] }>('tauri://drag-drop', (event) => {
      console.log('File dropped:', event.payload);
      if (event.payload.paths && event.payload.paths.length > 0) {
        setIsDragging(false);
        props.onFileSelected(event.payload.paths[0]);
      }
    });

    // Listen for drag hover events
    unlistenHover = await appWindow.listen('tauri://drag', () => {
      console.log('Drag hover detected');
      setIsDragging(true);
    });

    // Note: There's no drag-cancelled event in Tauri 2, we handle it in drop
  });

  onCleanup(() => {
    if (unlistenDrop) unlistenDrop();
    if (unlistenHover) unlistenHover();
  });

  // Handle click to open file picker
  const handleClick = async () => {
    console.log('FileUploader clicked');
    try {
      const selected = await open({
        multiple: false,
        filters: [{
          name: "Audio Files",
          extensions: ["mp3", "wav", "flac", "m4a", "ogg"],
        }],
      });

      console.log('File selected:', selected);
      if (selected && typeof selected === "string") {
        props.onFileSelected(selected);
      }
    } catch (err) {
      console.error('Error opening file picker:', err);
    }
  };

  return (
    <div
      class={`file-uploader ${isDragging() ? "dragging" : ""}`}
      onClick={handleClick}
    >
      <div class="upload-icon">
        <svg viewBox="0 0 24 24" width="64" height="64">
          <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
          <polyline points="17 8 12 3 7 8" />
          <line x1="12" y1="3" x2="12" y2="15" />
        </svg>
      </div>
      <h3>Drop Audio File Here or Click to Browse</h3>
      <p>Supports MP3, WAV, FLAC, M4A, OGG</p>
    </div>
  );
};

export default FileUploader;
