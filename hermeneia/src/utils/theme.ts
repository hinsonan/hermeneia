import { createSignal, onMount } from 'solid-js';

export type Theme = 'light' | 'dark';

const THEME_STORAGE_KEY = 'hermeneia-theme';

// Create a signal for theme state
const [theme, setTheme] = createSignal<Theme>('light');

/**
 * Initialize theme from localStorage or system preference
 */
export function initTheme(): void {
  const savedTheme = localStorage.getItem(THEME_STORAGE_KEY) as Theme | null;

  if (savedTheme && (savedTheme === 'light' || savedTheme === 'dark')) {
    applyTheme(savedTheme);
  } else if (window.matchMedia('(prefers-color-scheme: dark)').matches) {
    applyTheme('dark');
  } else {
    applyTheme('light');
  }
}

/**
 * Apply theme to document and update state
 */
function applyTheme(newTheme: Theme): void {
  document.documentElement.setAttribute('data-theme', newTheme);
  setTheme(newTheme);
  localStorage.setItem(THEME_STORAGE_KEY, newTheme);
}

/**
 * Toggle between light and dark themes
 */
export function toggleTheme(): void {
  const newTheme = theme() === 'dark' ? 'light' : 'dark';
  applyTheme(newTheme);
}

/**
 * Get current theme
 */
export function getTheme() {
  return theme;
}

/**
 * Hook to use theme in components
 */
export function useTheme() {
  onMount(() => {
    initTheme();
  });

  return {
    theme: getTheme(),
    toggleTheme,
  };
}
