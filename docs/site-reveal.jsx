// Reveal-on-scroll via scroll listener (more reliable than IO in some
// embedded/iframe contexts). One-shot: once visible, stays visible.

(function () {
  if (window.__p2bReveal) return;
  window.__p2bReveal = true;

  const TRIGGER = 0.88; // element top must be above this fraction of viewport

  function tick() {
    const vh = window.innerHeight || document.documentElement.clientHeight;
    const els = document.querySelectorAll('.reveal:not(.is-visible), .reveal-fade:not(.is-visible)');
    for (const el of els) {
      const r = el.getBoundingClientRect();
      // Visible if any part is in or past the trigger line.
      if (r.top < vh * TRIGGER && r.bottom > 0) {
        el.classList.add('is-visible');
      }
    }
  }

  let raf = 0;
  function schedule() {
    if (raf) return;
    raf = requestAnimationFrame(() => { raf = 0; tick(); });
  }

  window.addEventListener('scroll', schedule, { passive: true });
  window.addEventListener('resize', schedule);
  // Initial pass + recurring poll catches React mounts and embedded
  // contexts where programmatic scrolls don't fire scroll events.
  schedule();
  setInterval(tick, 120);

  // React hook (kept for compatibility)
  function useInView(ref, options = {}) {
    const [inView, setInView] = React.useState(false);
    React.useEffect(() => {
      const el = ref.current;
      if (!el) return;
      const check = () => {
        const r = el.getBoundingClientRect();
        const vh = window.innerHeight;
        if (r.top < vh * TRIGGER && r.bottom > 0) setInView(true);
      };
      check();
      window.addEventListener('scroll', check, { passive: true });
      return () => window.removeEventListener('scroll', check);
    }, [ref]);
    return inView;
  }
  window.useInView = useInView;
})();
