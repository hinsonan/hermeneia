document.addEventListener('DOMContentLoaded', () => {
  const html = document.documentElement;
  const themeToggle = document.querySelector('.theme-toggle');

  const savedTheme = localStorage.getItem('hermeneia-theme');
  if (savedTheme) {
    html.setAttribute('data-theme', savedTheme);
  } else if (window.matchMedia('(prefers-color-scheme: dark)').matches) {
    html.setAttribute('data-theme', 'dark');
  }

  themeToggle.addEventListener('click', () => {
    const current = html.getAttribute('data-theme');
    const next = current === 'dark' ? 'light' : 'dark';
    html.setAttribute('data-theme', next);
    localStorage.setItem('hermeneia-theme', next);
  });

  const revealElements = document.querySelectorAll('.feature-card, .download-card, .install-instructions, .section-title, .section-description');
  revealElements.forEach((el) => el.classList.add('reveal'));

  const observer = new IntersectionObserver(
    (entries) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) {
          entry.target.classList.add('visible');
          observer.unobserve(entry.target);
        }
      });
    },
    { threshold: 0.1, rootMargin: '0px 0px -40px 0px' }
  );

  revealElements.forEach((el) => observer.observe(el));

  document.querySelectorAll('a[href^="#"]').forEach((link) => {
    link.addEventListener('click', (e) => {
      const target = document.querySelector(link.getAttribute('href'));
      if (target) {
        e.preventDefault();
        target.scrollIntoView({ behavior: 'smooth' });
      }
    });
  });

  // Auto-update download links from GitHub releases
  const downloadLinks = document.querySelectorAll('.dl-link[data-asset]');
  if (downloadLinks.length > 0) {
    console.log('Fetching latest release info...');
    fetch('https://api.github.com/repos/hinsonan/hermeneia/releases/latest')
      .then(response => {
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        return response.json();
      })
      .then(data => {
        console.log('Release data received:', data.tag_name);
        if (!data.assets) return;

        const assetMap = {
          'deb-cpu': n => n.endsWith('amd64-cpu.deb'),
          'deb-cuda': n => n.endsWith('amd64-cuda.deb'),
          'rpm-cpu': n => n.endsWith('cpu.rpm'),
          'rpm-cuda': n => n.endsWith('cuda.rpm.xz') || n.endsWith('cuda.rpm'),
          'appimage-cpu': n => n.endsWith('amd64-cpu.AppImage'),
          'appimage-cuda': n => n.endsWith('amd64-cuda.AppImage'),
          'dmg-aarch64': n => n.endsWith('aarch64.dmg'),
          'dmg-x64': n => n.endsWith('x64.dmg'),
          'exe-cpu': n => n.endsWith('x64-setup-cpu.exe'),
          'exe-cuda': n => n.endsWith('windows_x64_cuda_installer.exe')
        };

        let updatedCount = 0;
        downloadLinks.forEach(link => {
          const assetType = link.getAttribute('data-asset');
          const matcher = assetMap[assetType];
          if (matcher) {
            const matched = data.assets.find(a => matcher(a.name));
            if (matched) {
              link.href = matched.browser_download_url;
              updatedCount++;
            } else {
              console.warn(`No asset found for ${assetType}`);
            }
          }
        });
        console.log(`Updated ${updatedCount} download links.`);
      })
      .catch(err => {
        console.error('Failed to fetch release info:', err);
        console.warn('Links will fall back to the generic releases page.');
      });
  }
});
