import { Component } from 'solid-js';
import { useTheme } from '../utils/theme';
import './Home.css';

const Home: Component = () => {
  const { theme, toggleTheme } = useTheme();

  const navigateTo = (page: string) => {
    console.log(`Navigating to: ${page}`);
    // TODO: Integrate with router when implemented
  };

  return (
    <>
      {/* Theme Toggle */}
      <button
        class="theme-toggle"
        onClick={toggleTheme}
        aria-label="Toggle dark mode"
      >
        <svg class="sun-icon" viewBox="0 0 24 24">
          <circle cx="12" cy="12" r="5"/>
          <line x1="12" y1="1" x2="12" y2="3"/>
          <line x1="12" y1="21" x2="12" y2="23"/>
          <line x1="4.22" y1="4.22" x2="5.64" y2="5.64"/>
          <line x1="18.36" y1="18.36" x2="19.78" y2="19.78"/>
          <line x1="1" y1="12" x2="3" y2="12"/>
          <line x1="21" y1="12" x2="23" y2="12"/>
          <line x1="4.22" y1="19.78" x2="5.64" y2="18.36"/>
          <line x1="18.36" y1="5.64" x2="19.78" y2="4.22"/>
        </svg>
        <svg class="moon-icon" viewBox="0 0 24 24">
          <path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"/>
        </svg>
      </button>

      <div class="scroll-container">
        {/* Top Scroll Rod */}
        <div class="scroll-rod"></div>

        {/* Parchment Content */}
        <main class="parchment">
          {/* Corner Flourishes */}
          <div class="flourish flourish-tl">
            <svg viewBox="0 0 80 80">
              <path d="M5 75 Q5 5 75 5" stroke="var(--border-ornament)" stroke-width="2" fill="none"/>
              <path d="M15 65 Q15 15 65 15" stroke="var(--gold-accent)" stroke-width="1" fill="none"/>
              <circle cx="8" cy="8" r="3" fill="var(--gold-accent)"/>
            </svg>
          </div>
          <div class="flourish flourish-tr">
            <svg viewBox="0 0 80 80">
              <path d="M5 75 Q5 5 75 5" stroke="var(--border-ornament)" stroke-width="2" fill="none"/>
              <path d="M15 65 Q15 15 65 15" stroke="var(--gold-accent)" stroke-width="1" fill="none"/>
              <circle cx="8" cy="8" r="3" fill="var(--gold-accent)"/>
            </svg>
          </div>
          <div class="flourish flourish-bl">
            <svg viewBox="0 0 80 80">
              <path d="M5 75 Q5 5 75 5" stroke="var(--border-ornament)" stroke-width="2" fill="none"/>
              <path d="M15 65 Q15 15 65 15" stroke="var(--gold-accent)" stroke-width="1" fill="none"/>
              <circle cx="8" cy="8" r="3" fill="var(--gold-accent)"/>
            </svg>
          </div>
          <div class="flourish flourish-br">
            <svg viewBox="0 0 80 80">
              <path d="M5 75 Q5 5 75 5" stroke="var(--border-ornament)" stroke-width="2" fill="none"/>
              <path d="M15 65 Q15 15 65 15" stroke="var(--gold-accent)" stroke-width="1" fill="none"/>
              <circle cx="8" cy="8" r="3" fill="var(--gold-accent)"/>
            </svg>
          </div>

          {/* Header */}
          <header class="header">
            <h1 class="title">Hermeneia</h1>
            <p class="subtitle">Divine Word Transcription & Translation</p>
          </header>

          {/* Divider */}
          <div class="divider">
            <span class="divider-line"></span>
            <span class="divider-symbol">✤</span>
            <span class="divider-line"></span>
          </div>

          {/* Introduction */}
          <p class="intro">
            <span class="drop-cap">P</span>reserve and share the sacred word with clarity and precision.
            Hermeneia brings together powerful tools for transcribing sermons,
            translating scripture, and preparing audio for distribution—all
            running locally on your machine, respecting both your privacy and
            the sanctity of the message.
          </p>

          {/* Feature Cards */}
          <section class="features">
            {/* Transcribe */}
            <article class="feature-card" onClick={() => navigateTo('transcribe')}>
              <div class="feature-icon">
                <svg viewBox="0 0 24 24">
                  <path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z"/>
                  <path d="M19 10v2a7 7 0 0 1-14 0v-2"/>
                  <line x1="12" y1="19" x2="12" y2="23"/>
                  <line x1="8" y1="23" x2="16" y2="23"/>
                </svg>
              </div>
              <h2 class="feature-title">Transcribe</h2>
              <p class="feature-desc">
                Convert spoken sermons and teachings into written text with
                AI-powered speech recognition, running entirely on your device.
              </p>
            </article>

            {/* Translate */}
            <article class="feature-card" onClick={() => navigateTo('translate')}>
              <div class="feature-icon greek-text">
                <div class="greek-letter">Α</div>
                <div class="greek-letter">Ω</div>
              </div>
              <h2 class="feature-title">Translate</h2>
              <p class="feature-desc">
                Bridge language barriers by translating transcriptions and
                texts into multiple languages, preserving meaning and reverence.
              </p>
            </article>

            {/* Audio Processing */}
            <article class="feature-card" onClick={() => navigateTo('audio')}>
              <div class="feature-icon">
                <svg viewBox="0 0 24 24">
                  <path d="M3 18v-6a9 9 0 0 1 18 0v6"/>
                  <path d="M21 19a2 2 0 0 1-2 2h-1a2 2 0 0 1-2-2v-3a2 2 0 0 1 2-2h3z"/>
                  <path d="M3 19a2 2 0 0 0 2 2h1a2 2 0 0 0 2-2v-3a2 2 0 0 0-2-2H3z"/>
                  <line x1="9" y1="9" x2="9" y2="9.01"/>
                  <line x1="15" y1="9" x2="15" y2="9.01"/>
                  <path d="M10 13h4"/>
                </svg>
              </div>
              <h2 class="feature-title">Audio Tools</h2>
              <p class="feature-desc">
                Prepare sermon recordings for distribution with trimming,
                noise reduction, and format conversion utilities.
              </p>
            </article>
          </section>

          {/* Scripture Quote */}
          <div class="divider">
            <span class="divider-line"></span>
            <span class="divider-symbol">❦</span>
            <span class="divider-line"></span>
          </div>

          <blockquote class="scripture">
            <p class="scripture-text">
              "So shall my word be that goeth forth out of my mouth: it shall not return
              unto me void, but it shall accomplish that which I please."
            </p>
            <cite class="scripture-ref">— Isaiah 55:11</cite>
          </blockquote>

          {/* Footer */}
          <footer class="footer">
            <p class="footer-text">
              <span class="footer-symbol">✦</span>
              Local-First
              <span class="footer-symbol">•</span>
              Privacy-Respecting
              <span class="footer-symbol">•</span>
              GPU-Accelerated
              <span class="footer-symbol">✦</span>
            </p>
          </footer>
        </main>

        {/* Bottom Scroll Rod */}
        <div class="scroll-rod"></div>
      </div>
    </>
  );
};

export default Home;
