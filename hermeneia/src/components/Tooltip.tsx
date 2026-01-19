import { Component, JSX, createSignal, Show } from "solid-js";
import "./Tooltip.css";

interface TooltipProps {
  content: string | JSX.Element;
  children: JSX.Element;
  position?: "top" | "bottom" | "left" | "right";
}

const Tooltip: Component<TooltipProps> = (props) => {
  const [isVisible, setIsVisible] = createSignal(false);
  const position = props.position || "top";

  return (
    <div
      class="tooltip-wrapper"
      onMouseEnter={() => setIsVisible(true)}
      onMouseLeave={() => setIsVisible(false)}
    >
      {props.children}
      <Show when={isVisible()}>
        <div class={`tooltip-content tooltip-${position}`}>
          {props.content}
        </div>
      </Show>
    </div>
  );
};

export default Tooltip;
