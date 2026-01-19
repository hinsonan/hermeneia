import { Component, JSX } from "solid-js";
import Tooltip from "./Tooltip";
import "./InfoIcon.css";

interface InfoIconProps {
  content: string | JSX.Element;
  position?: "top" | "bottom" | "left" | "right";
}

const InfoIcon: Component<InfoIconProps> = (props) => {
  return (
    <Tooltip content={props.content} position={props.position}>
      <div class="info-icon" aria-label="Information">
        <svg viewBox="0 0 24 24" width="16" height="16">
          <circle cx="12" cy="12" r="10" />
          <line x1="12" y1="16" x2="12" y2="12" />
          <line x1="12" y1="8" x2="12.01" y2="8" />
        </svg>
      </div>
    </Tooltip>
  );
};

export default InfoIcon;
