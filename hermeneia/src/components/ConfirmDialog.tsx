import { Component, Show } from "solid-js";
import "./ConfirmDialog.css";

interface ConfirmDialogProps {
  open: boolean;
  title: string;
  message: string;
  confirmLabel?: string;
  cancelLabel?: string;
  onConfirm: () => void;
  onCancel: () => void;
}

const ConfirmDialog: Component<ConfirmDialogProps> = (props) => {
  return (
    <Show when={props.open}>
      <div class="confirm-overlay" onClick={props.onCancel}>
        <div class="confirm-dialog" onClick={(e) => e.stopPropagation()}>
          <h3 class="confirm-title">{props.title}</h3>
          <p class="confirm-message">{props.message}</p>
          <div class="confirm-actions">
            <button class="confirm-btn cancel" onClick={props.onCancel}>
              {props.cancelLabel || "Keep Working"}
            </button>
            <button class="confirm-btn confirm" onClick={props.onConfirm}>
              {props.confirmLabel || "Stop & Go Back"}
            </button>
          </div>
        </div>
      </div>
    </Show>
  );
};

export default ConfirmDialog;
