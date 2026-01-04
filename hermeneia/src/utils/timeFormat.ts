/**
 * Time formatting and parsing utilities for audio editor
 */

/**
 * Format seconds to hh:mm:ss.SSS format
 * @param seconds - Time in seconds (can include fractional seconds)
 * @param showMillis - Whether to show milliseconds (default: true)
 * @returns Formatted time string
 */
export function formatTime(seconds: number, showMillis: boolean = true): string {
  const hours = Math.floor(seconds / 3600);
  const minutes = Math.floor((seconds % 3600) / 60);
  const secs = Math.floor(seconds % 60);
  const millis = Math.floor((seconds % 1) * 1000);

  const hh = hours.toString().padStart(2, "0");
  const mm = minutes.toString().padStart(2, "0");
  const ss = secs.toString().padStart(2, "0");

  if (showMillis) {
    const ms = millis.toString().padStart(3, "0");
    return `${hh}:${mm}:${ss}.${ms}`;
  }

  return `${hh}:${mm}:${ss}`;
}

/**
 * Parse hh:mm:ss.SSS format to seconds
 * Accepts formats: hh:mm:ss.SSS, hh:mm:ss, mm:ss, ss
 * @param timeStr - Time string to parse
 * @returns Time in seconds, or null if invalid
 */
export function parseTime(timeStr: string): number | null {
  if (!timeStr || typeof timeStr !== "string") return null;

  const trimmed = timeStr.trim();

  // Split by decimal point to handle milliseconds separately
  const [mainPart, millisPart] = trimmed.split(".");

  // Split main part by colons
  const parts = mainPart.split(":").map(p => parseInt(p, 10));

  // Validate all parts are numbers
  if (parts.some(isNaN)) return null;

  let seconds = 0;

  if (parts.length === 1) {
    // Just seconds: "45"
    seconds = parts[0];
  } else if (parts.length === 2) {
    // mm:ss: "5:30"
    const [mins, secs] = parts;
    seconds = mins * 60 + secs;
  } else if (parts.length === 3) {
    // hh:mm:ss: "1:05:30"
    const [hours, mins, secs] = parts;
    seconds = hours * 3600 + mins * 60 + secs;
  } else {
    return null; // Invalid format
  }

  // Add milliseconds if present
  if (millisPart) {
    const millis = parseInt(millisPart.padEnd(3, "0").slice(0, 3), 10);
    if (!isNaN(millis)) {
      seconds += millis / 1000;
    }
  }

  return seconds;
}

/**
 * Validate time string format
 * @param timeStr - Time string to validate
 * @returns True if valid format
 */
export function isValidTimeFormat(timeStr: string): boolean {
  return parseTime(timeStr) !== null;
}
