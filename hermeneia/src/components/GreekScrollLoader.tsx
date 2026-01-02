import { Component } from "solid-js";
import "./GreekScrollLoader.css";

const GreekScrollLoader: Component = () => {
  return (
    <div class="greek-scroll-loader">
      <svg
        viewBox="0 0 400 200"
        class="scroll-animation"
        xmlns="http://www.w3.org/2000/svg"
      >
        {/* Scroll parchment */}
        <defs>
          {/* Parchment texture gradient */}
          <linearGradient id="parchmentGrad" x1="0%" y1="0%" x2="0%" y2="100%">
            <stop offset="0%" style="stop-color:#f4e8d0;stop-opacity:1" />
            <stop offset="50%" style="stop-color:#e8dcc4;stop-opacity:1" />
            <stop offset="100%" style="stop-color:#f4e8d0;stop-opacity:1" />
          </linearGradient>

          {/* Wood gradient for scroll rods */}
          <linearGradient id="woodGrad" x1="0%" y1="0%" x2="100%" y2="0%">
            <stop offset="0%" style="stop-color:#3a2416;stop-opacity:1" />
            <stop offset="50%" style="stop-color:#6b4e3d;stop-opacity:1" />
            <stop offset="100%" style="stop-color:#3a2416;stop-opacity:1" />
          </linearGradient>
        </defs>

        {/* Left scroll rod */}
        <rect x="10" y="40" width="15" height="120" rx="7.5" fill="url(#woodGrad)" />
        <ellipse cx="17.5" cy="40" rx="7.5" ry="8" fill="#6b4e3d" />
        <ellipse cx="17.5" cy="160" rx="7.5" ry="8" fill="#3a2416" />

        {/* Right scroll rod */}
        <rect x="375" y="40" width="15" height="120" rx="7.5" fill="url(#woodGrad)" />
        <ellipse cx="382.5" cy="40" rx="7.5" ry="8" fill="#6b4e3d" />
        <ellipse cx="382.5" cy="160" rx="7.5" ry="8" fill="#3a2416" />

        {/* Parchment */}
        <rect
          x="25"
          y="50"
          width="350"
          height="100"
          fill="url(#parchmentGrad)"
          stroke="#8b6f47"
          stroke-width="1"
        />

        {/* Decorative border on parchment */}
        <rect
          x="35"
          y="60"
          width="330"
          height="80"
          fill="none"
          stroke="#b8860b"
          stroke-width="0.5"
          opacity="0.3"
        />

        {/* Greek text being written - Koine Greek: "In the beginning was the Word" */}
        <text
          x="50"
          y="90"
          font-family="'Times New Roman', 'Palatino Linotype', serif"
          font-size="20"
          fill="#2c1810"
          class="greek-text"
        >
          Ἐν ἀρχῇ ἦν ὁ λόγος
        </text>

        {/* Second line - "and the Word was with God" */}
        <text
          x="50"
          y="115"
          font-family="'Times New Roman', 'Palatino Linotype', serif"
          font-size="18"
          fill="#2c1810"
          class="greek-text-line2"
        >
          καὶ ὁ λόγος ἦν
        </text>

        {/* Reed pen (calamus) - ancient writing instrument */}
        <g class="reed-pen">
          {/* Reed shaft - hollow cylindrical shape */}
          <defs>
            <linearGradient id="reedGrad" x1="0%" y1="0%" x2="100%" y2="0%">
              <stop offset="0%" style="stop-color:#8b7355;stop-opacity:1" />
              <stop offset="50%" style="stop-color:#c9b896;stop-opacity:1" />
              <stop offset="100%" style="stop-color:#8b7355;stop-opacity:1" />
            </linearGradient>
          </defs>

          {/* Main reed shaft - angled for writing */}
          <rect
            x="48"
            y="72"
            width="35"
            height="4"
            rx="2"
            fill="url(#reedGrad)"
            stroke="#6b5a45"
            stroke-width="0.3"
            transform="rotate(-35 65 74)"
          />

          {/* Reed segments (natural bamboo-like rings) */}
          <line x1="56" y1="76" x2="56" y2="79" stroke="#6b5a45" stroke-width="0.4" opacity="0.6" transform="rotate(-35 65 74)" />
          <line x1="62" y1="74.5" x2="62" y2="77.5" stroke="#6b5a45" stroke-width="0.4" opacity="0.6" transform="rotate(-35 65 74)" />
          <line x1="68" y1="73" x2="68" y2="76" stroke="#6b5a45" stroke-width="0.4" opacity="0.6" transform="rotate(-35 65 74)" />

          {/* Nib - cut at an angle with split tip */}
          <path
            d="M 42 92 L 48 88 L 48.5 88.5 L 43 92.8 Z"
            fill="#3a2416"
            stroke="#2c1810"
            stroke-width="0.3"
          />

          {/* Split in nib (characteristic of reed pens) */}
          <line x1="45" y1="90" x2="48.2" y2="88.2" stroke="#c9b896" stroke-width="0.3" />

          {/* Ink on nib tip */}
          <ellipse cx="42.5" cy="92.5" rx="1" ry="0.8" fill="#2c1810" opacity="0.8" />

          {/* Ink drop animation */}
          <circle cx="42" cy="93" r="0.6" fill="#2c1810" class="ink-drop">
            <animate
              attributeName="cy"
              values="93;96;96"
              dur="2s"
              repeatCount="indefinite"
            />
            <animate
              attributeName="opacity"
              values="0;1;0"
              dur="2s"
              repeatCount="indefinite"
            />
          </circle>
        </g>
      </svg>

      {/* Text below the scroll */}
      <div class="scroll-text">
        <p class="scroll-title">Preparing manuscript...</p>
        <p class="scroll-subtitle">Rendering sacred text</p>
      </div>
    </div>
  );
};

export default GreekScrollLoader;
