import { Component } from 'solid-js';
import { useNavigate } from '@solidjs/router';
import { useTheme } from '../utils/theme';
import './Home.css';

const Home: Component = () => {
  const navigate = useNavigate();
  const { toggleTheme } = useTheme();

  const navigateTo = (page: string) => {
    if (page === 'audio') {
      navigate('/audio-editor');
    } else if (page === 'transcribe') {
      navigate('/transcription');
    } else if (page === 'translate') {
      navigate('/translation');
    }
  };

  return (
    <>
      {/* Model Library Button */}
      <button
        class="models-btn"
        onClick={() => navigate('/models')}
        aria-label="Model Library"
      >
        <svg viewBox="0 0 24 24" width="22" height="22">
          {/* Top roll */}
          <ellipse cx="12" cy="4" rx="8" ry="2.5" />
          {/* Bottom roll */}
          <ellipse cx="12" cy="20" rx="8" ry="2.5" />
          {/* Left edge */}
          <line x1="4" y1="4" x2="4" y2="20" />
          {/* Right edge */}
          <line x1="20" y1="4" x2="20" y2="20" />
          {/* Text lines on scroll */}
          <line x1="8" y1="9" x2="16" y2="9" />
          <line x1="8" y1="12" x2="16" y2="12" />
          <line x1="8" y1="15" x2="14" y2="15" />
        </svg>
      </button>

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
            <span class="divider-symbol">
              <svg viewBox="0 0 28 28" width="26" height="26">
                {/* Trunk */}
                <line x1="14" y1="26" x2="14" y2="4" stroke="var(--gold-accent)" stroke-width="1.5"/>
                {/* Bottom branches */}
                <path d="M14 21 C11 19, 8 18, 6 15" fill="none" stroke="var(--gold-accent)" stroke-width="1.2"/>
                <path d="M14 21 C17 19, 20 18, 22 15" fill="none" stroke="var(--gold-accent)" stroke-width="1.2"/>
                {/* Middle branches */}
                <path d="M14 15 C12 13, 9 12, 7 9" fill="none" stroke="var(--gold-accent)" stroke-width="1.2"/>
                <path d="M14 15 C16 13, 19 12, 21 9" fill="none" stroke="var(--gold-accent)" stroke-width="1.2"/>
                {/* Top branches */}
                <path d="M14 9 C13 7, 11 5, 10 3" fill="none" stroke="var(--gold-accent)" stroke-width="1.2"/>
                <path d="M14 9 C15 7, 17 5, 18 3" fill="none" stroke="var(--gold-accent)" stroke-width="1.2"/>
                {/* Leaf dots at branch tips */}
                <circle cx="6" cy="15" r="1.3" fill="var(--gold-accent)"/>
                <circle cx="22" cy="15" r="1.3" fill="var(--gold-accent)"/>
                <circle cx="7" cy="9" r="1.3" fill="var(--gold-accent)"/>
                <circle cx="21" cy="9" r="1.3" fill="var(--gold-accent)"/>
                <circle cx="10" cy="3" r="1.3" fill="var(--gold-accent)"/>
                <circle cx="18" cy="3" r="1.3" fill="var(--gold-accent)"/>
              </svg>
            </span>
            <span class="divider-line"></span>
          </div>

          {/* Introduction */}
          <p class="intro">
            <span class="drop-cap">T</span>ranscribe and translate sermons with clarity.
            Powerful local tools for turning spoken word into text and
            bridging languages—all on your machine, fully private.
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
                Prepare sermon recordings for distribution with trimming
                and format conversion utilities.
              </p>
            </article>
          </section>

          {/* Scripture Quote */}
          <div class="divider">
            <span class="divider-line"></span>
            <span class="divider-symbol">
              <svg viewBox="0 0 32 32" width="28" height="28">
                {/* Left page - curved to look open */}
                <path d="M3 6 C3 5, 4 4, 6 4 L15 4 C16 4, 16 5, 16 6 L16 27 C16 26, 15 25, 14 25 L6 25 C4 25, 3 24, 3 23 Z" fill="none" stroke="var(--gold-accent)" stroke-width="1.5"/>
                {/* Right page */}
                <path d="M29 6 C29 5, 28 4, 26 4 L17 4 C16 4, 16 5, 16 6 L16 27 C16 26, 17 25, 18 25 L26 25 C28 25, 29 24, 29 23 Z" fill="none" stroke="var(--gold-accent)" stroke-width="1.5"/>
                {/* Spine */}
                <line x1="16" y1="4" x2="16" y2="27" stroke="var(--gold-accent)" stroke-width="1.5"/>
                {/* Text lines - left */}
                <line x1="6" y1="10" x2="13" y2="10" stroke="var(--gold-accent)" stroke-width="1" opacity="0.5"/>
                <line x1="6" y1="14" x2="13" y2="14" stroke="var(--gold-accent)" stroke-width="1" opacity="0.5"/>
                <line x1="6" y1="18" x2="13" y2="18" stroke="var(--gold-accent)" stroke-width="1" opacity="0.5"/>
                {/* Text lines - right */}
                <line x1="19" y1="10" x2="26" y2="10" stroke="var(--gold-accent)" stroke-width="1" opacity="0.5"/>
                <line x1="19" y1="14" x2="26" y2="14" stroke="var(--gold-accent)" stroke-width="1" opacity="0.5"/>
                <line x1="19" y1="18" x2="26" y2="18" stroke="var(--gold-accent)" stroke-width="1" opacity="0.5"/>
                {/* Cross - centered on right page */}
                <line x1="22.5" y1="20.5" x2="22.5" y2="24" stroke="var(--gold-accent)" stroke-width="1.2" opacity="0.7"/>
                <line x1="21" y1="21.5" x2="24" y2="21.5" stroke="var(--gold-accent)" stroke-width="1.2" opacity="0.7"/>
              </svg>
            </span>
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
