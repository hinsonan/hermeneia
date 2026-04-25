import { Component, JSX, Show, createEffect, createSignal } from "solid-js";
import "./Tooltip.css";

interface TooltipProps {
  content: string | JSX.Element;
  children: JSX.Element;
  position?: "top" | "bottom" | "left" | "right";
}

const Tooltip: Component<TooltipProps> = (props) => {
  const [isVisible, setIsVisible] = createSignal(false);
  const basePosition = () => props.position || "top";
  const [resolvedPosition, setResolvedPosition] = createSignal<TooltipProps["position"]>(basePosition());
  const [shiftX, setShiftX] = createSignal(0);

  let tooltipEl: HTMLDivElement | undefined;

  createEffect(() => {
    setResolvedPosition(basePosition());
    setShiftX(0);
  });

  createEffect(() => {
    if (!isVisible()) return;
    if (!tooltipEl) return;

    const viewportPadding = 8;
    const rect = tooltipEl.getBoundingClientRect();

    let nextPosition = resolvedPosition();
    if (nextPosition === "right" && rect.right > window.innerWidth - viewportPadding) {
      nextPosition = "left";
    } else if (nextPosition === "left" && rect.left < viewportPadding) {
      nextPosition = "right";
    } else if (nextPosition === "top" && rect.top < viewportPadding) {
      nextPosition = "bottom";
    } else if (nextPosition === "bottom" && rect.bottom > window.innerHeight - viewportPadding) {
      nextPosition = "top";
    }

    if (nextPosition !== resolvedPosition()) {
      setResolvedPosition(nextPosition);
      return;
    }

    if (nextPosition === "top" || nextPosition === "bottom") {
      let delta = 0;
      if (rect.right > window.innerWidth - viewportPadding) {
        delta = (window.innerWidth - viewportPadding) - rect.right;
      } else if (rect.left < viewportPadding) {
        delta = viewportPadding - rect.left;
      }
      if (delta !== shiftX()) {
        setShiftX(delta);
      }
      return;
    }

    if (shiftX() !== 0) {
      setShiftX(0);
    }
  });

  return (
    <div
      class="tooltip-wrapper"
      onMouseEnter={() => {
        setResolvedPosition(basePosition());
        setShiftX(0);
        setIsVisible(true);
      }}
      onMouseLeave={() => setIsVisible(false)}
    >
      {props.children}
      <Show when={isVisible()}>
        <div
          ref={tooltipEl}
          class={`tooltip-content tooltip-${resolvedPosition()}`}
          style={`--tooltip-shift-x: ${shiftX()}px;`}
        >
          {props.content}
        </div>
      </Show>
    </div>
  );
};

export default Tooltip;
