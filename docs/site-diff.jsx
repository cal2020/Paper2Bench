// Full diff view — variant A (side-by-side, git-style).
// Uses real CGCNN instruction.txt and instruction_gt.txt content from data.js.
// instruction_gt.txt column auto-scrolls down to the first added section
// ("## Experimental Design") once it scrolls into view.

function FullDiff({ paper: paperProp }) {
  const paper = paperProp || (window.P2B_DATA && window.P2B_DATA.current);
  const D = paper;
  const left = D.instructionReal.split('\n');
  const right = (D.instructionReal + '\n\n---\n\n' + D.instructionGtAddendum).split('\n');
  const shared = left.length;

  const isSection = (s) => /^##\s/.test(s);

  // Index of the first "## Experimental Design" line (where the diff starts)
  const firstSectionIdx = right.findIndex((l, i) => i >= shared && isSection(l));

  // Auto-scroll the GT body down to the first section heading the first
  // time the diff scrolls into view.
  const gtBodyRef = React.useRef(null);
  const containerRef = React.useRef(null);
  const hasScrolledRef = React.useRef(false);

  // Reset auto-scroll when paper changes (detail page can be reused for
  // multiple papers in principle).
  React.useEffect(() => { hasScrolledRef.current = false; }, [paper]);

  React.useEffect(() => {
    const container = containerRef.current;
    if (!container) return;
    const tryScroll = () => {
      if (hasScrolledRef.current) return;
      const r = container.getBoundingClientRect();
      const vh = window.innerHeight || 800;
      // Trigger when the diff stats row is comfortably in view.
      if (r.top < vh * 0.6 && r.bottom > 0) {
        hasScrolledRef.current = true;
        const body = gtBodyRef.current;
        if (!body) return;
        const target = body.querySelector('[data-first-section="true"]');
        if (!target) return;
        // Wait a beat for stats/reveals, then animate-scroll.
        setTimeout(() => {
          const t = target.offsetTop - 24; // leave a little headroom
          // Smooth scroll the gt body to t. We do it ourselves so the
          // outer page doesn't move.
          const start = body.scrollTop;
          const distance = t - start;
          const dur = 1400;
          const t0 = performance.now();
          const ease = (x) => 1 - Math.pow(1 - x, 3);
          const step = (now) => {
            const k = Math.min(1, (now - t0) / dur);
            body.scrollTop = start + distance * ease(k);
            if (k < 1) requestAnimationFrame(step);
          };
          requestAnimationFrame(step);
        }, 700);
      }
    };
    tryScroll();
    const id = setInterval(tryScroll, 200);
    window.addEventListener('scroll', tryScroll, { passive: true });
    return () => {
      clearInterval(id);
      window.removeEventListener('scroll', tryScroll);
    };
  }, []);

  // Line component used by both columns
  const Line = ({ text, idx, kind, isFirstSection }) => {
    const sec = kind === 'added' && isSection(text);
    return (
      <div
        className={`diff-line ${kind === 'added' ? 'added' : ''} ${sec ? 'section' : ''}`}
        data-first-section={isFirstSection ? 'true' : undefined}
      >
        <span className="ln">{idx + 1}</span>
        <span className="gutter">{kind === 'added' ? '+' : ' '}</span>
        <span className="text">{text || ' '}</span>
      </div>
    );
  };

  return (
    <div ref={containerRef}>
      {/* Side-by-side diff */}
      <div className="diff">
        {/* Left: instruction.txt — no placeholder padding */}
        <div className="diff-col">
          <div className="diff-col-head">
            <span className="filename">instruction.txt</span>
            <span className="label">agent sees · {(D.diff.sharedBytes/1024).toFixed(1)} KB · {left.length} lines</span>
          </div>
          <div className="diff-body">
            {left.map((line, i) => <Line key={i} text={line} idx={i} kind="shared" />)}
          </div>
        </div>
        {/* Right: instruction_gt.txt — shared prefix + addendum, auto-scrolls
            to first ## section once in view. */}
        <div className="diff-col">
          <div className="diff-col-head gt">
            <span className="filename">instruction_gt.txt</span>
            <span className="label accent">evaluator sees · {(D.diff.totalBytes/1024).toFixed(1)} KB · {right.length} lines</span>
          </div>
          <div className="diff-body" ref={gtBodyRef}>
            {right.slice(0, shared).map((line, i) => (
              <Line key={i} text={line} idx={i} kind="shared" />
            ))}
            {right.slice(shared).map((line, i) => {
              const absIdx = shared + i;
              return (
                <Line
                  key={`a${i}`}
                  text={line}
                  idx={absIdx}
                  kind="added"
                  isFirstSection={absIdx === firstSectionIdx}
                />
              );
            })}
          </div>
        </div>
      </div>
    </div>
  );
}

window.FullDiff = FullDiff;
