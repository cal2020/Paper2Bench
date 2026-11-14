// Walkthrough — 3 stages, tree-centric canvas with on-diagram annotations.
//
// Stage 1: PDF drops in. Tree empty.
// Stage 2: Tree builds — root, then RQ row, then experiment leaves (timed).
// Stage 3: Output file cards emerge from the leaves; diff callout appears.

const WSTAGES = [
  {
    n: 1,
    label: 'Drop the paper',
    desc: 'Pass a local PDF, or let paper2bench search arXiv by title. One LLM read decomposes the paper.',
  },
  {
    n: 2,
    label: 'Decompose the science',
    desc: 'A 3-level problem tree: root question → research questions → concrete experiments with dataset, model, metric, and a back-reference to the figure or table.',
  },
  {
    n: 3,
    label: 'Render the briefs',
    desc: 'instruction.txt is what the agent literally reads. instruction_gt.txt appends a generated reference plan for curator review. The example illustrates authoring output; conclusion scoring does not validate the scientific method.',
  },
];

// ---------- Layout coordinates (viewBox 1400 × 760) ----------
const VB = { w: 1400, h: 760 };
const PDF  = { x: 580, y: 24,  w: 240, h: 72 };
const ROOT = { x: 580, y: 140, w: 240, h: 64 };
const RQ_Y = 290;
const RQ_W = 220, RQ_H = 64;
const LEAF_Y = 458;
const LEAF_W = 160, LEAF_H = 76;
const FILE_W = 280, FILE_H = 100;
const FILES_Y = 624;

// Layout the tree based on the *current* paper's RQ + leaf counts. Returns
// concrete x/y positions in viewBox space.
function layoutTree(paperTree) {
  const rqCount = paperTree.rqs.length;
  // Evenly space RQs across the canvas with consistent margins.
  const margin = 110;
  const usable = VB.w - 2 * margin - RQ_W;
  const rqs = paperTree.rqs.map((rq, i) => ({
    ...rq,
    x: rqCount === 1
      ? VB.w / 2 - RQ_W / 2
      : margin + (usable / (rqCount - 1)) * i,
  }));
  const leaves = [];
  rqs.forEach((rq, ri) => {
    const parentCx = rq.x + RQ_W / 2;
    const lc = rq.leaves.length;
    rq.leaves.forEach((leaf, li) => {
      let cx;
      if (lc === 1) cx = parentCx;
      else if (lc === 2) cx = parentCx + (li === 0 ? -95 : 95);
      else cx = parentCx + (li - (lc - 1) / 2) * 110;
      leaves.push({ ...leaf, x: cx - LEAF_W / 2, parentIdx: ri });
    });
  });
  // Files always emerge from the first leaf.
  const firstLeaf = leaves[0];
  const firstLeafCx = firstLeaf ? firstLeaf.x + LEAF_W / 2 : VB.w / 2;
  const fileGap = 220;
  const files = [
    { id: 'instr',    x: Math.max(140, firstLeafCx - fileGap - FILE_W),       y: FILES_Y,
      name: 'instruction.txt',    sub: 'agent input',          featured: false },
    { id: 'instr_gt', x: Math.max(140, firstLeafCx - fileGap - FILE_W) + FILE_W + fileGap, y: FILES_Y,
      name: 'instruction_gt.txt', sub: '+ evaluator addendum', featured: true  },
  ];
  return { rqs, leaves, files };
}

// HTML version of the PDF icon — used inside the overlay card so we can
// scale + translate it via CSS transitions (SVG group transforms don't
// animate smoothly across browsers).
const PdfIconHtml = ({ size = 44 }) => (
  <svg className="pdf-icon-html" width={size} height={size * 1.25} viewBox="0 0 80 100" aria-hidden>
    <path d="M 0 0 L 56 0 L 80 24 L 80 100 L 0 100 z" fill="var(--surface)" stroke="var(--fg)" strokeWidth="2" strokeLinejoin="round" />
    <path d="M 56 0 L 80 24 L 56 24 z" fill="var(--surface-2)" stroke="var(--fg)" strokeWidth="2" strokeLinejoin="round" />
    <rect x="14" y="38" width="48" height="2.5" rx="1" fill="var(--rule-strong)" />
    <rect x="14" y="46" width="52" height="2.5" rx="1" fill="var(--rule-strong)" />
    <rect x="14" y="54" width="38" height="2.5" rx="1" fill="var(--rule-strong)" />
    <rect x="14" y="68" width="34" height="14" rx="2" fill="var(--accent)" />
    <text x="31" y="79" textAnchor="middle" fontFamily="var(--mono)" fontSize="9" fontWeight="700" fill="#fff" letterSpacing="0.5">PDF</text>
  </svg>
);

// A reusable PDF document icon. Renders an SVG <g> at a translated origin
// that's `size` pixels tall. The page has a folded top-right corner and a
// red "PDF" wordmark at the bottom.
const PdfIcon = ({ x = 0, y = 0, size = 100 }) => {
  // Base canvas 80×100 — scale to requested height.
  const s = size / 100;
  return (
    <g transform={`translate(${x}, ${y}) scale(${s})`}>
      <path d="M 0 0 L 56 0 L 80 24 L 80 100 L 0 100 z"
        fill="var(--surface)" stroke="var(--fg)" strokeWidth="2" strokeLinejoin="round" />
      <path d="M 56 0 L 80 24 L 56 24 z" fill="var(--surface-2)" stroke="var(--fg)" strokeWidth="2" strokeLinejoin="round" />
      <rect x="14" y="38" width="48" height="2.5" rx="1" fill="var(--rule-strong)" />
      <rect x="14" y="46" width="52" height="2.5" rx="1" fill="var(--rule-strong)" />
      <rect x="14" y="54" width="38" height="2.5" rx="1" fill="var(--rule-strong)" />
      <rect x="14" y="68" width="34" height="14" rx="2" fill="var(--accent)" />
      <text x="31" y="79" textAnchor="middle" fontFamily="var(--mono)" fontSize="9" fontWeight="700" fill="#fff" letterSpacing="0.5">PDF</text>
    </g>
  );
};

// Centerline helpers
const centerOf = (b) => ({ x: b.x + b.w / 2, y: b.y + b.h / 2 });
const botMid   = (b) => ({ x: b.x + b.w / 2, y: b.y + b.h });
const topMid   = (b) => ({ x: b.x + b.w / 2, y: b.y });

// Curved S-path between two y-different points
const sPath = (a, b) => {
  const my = (a.y + b.y) / 2;
  return `M ${a.x} ${a.y} C ${a.x} ${my}, ${b.x} ${my}, ${b.x} ${b.y}`;
};

// ---------- The canvas ----------
function CanvasViz({ stage, tier, paper }) {
  const { rqs: RQS, leaves: LEAVES, files: FILES } = React.useMemo(
    () => layoutTree(paper.tree),
    [paper]
  );
  const PDF_NAME = `${paper.id}.pdf`;
  const PDF_META = `arXiv · ${paper.year}`;
  const showPdf       = true;
  const pdfBig        = stage === 1;
  const showRoot      = stage >= 2 && tier >= 1;
  const showRqs       = stage >= 2 && tier >= 2;
  const showLeaves    = stage >= 2 && tier >= 3;
  const showFiles     = stage >= 3;
  const showAnnot1    = stage === 1;
  const showAnnotTree = stage === 2 && tier >= 3;

  // PDF position shifts: huge & centered at stage 1, smaller at top at stage 2+
  const pdfBox = pdfBig
    ? { x: VB.w / 2 - 200, y: VB.h / 2 - 100, w: 400, h: 200 }
    : PDF;

  const fadeStyle = (visible, delay = 0) => ({
    opacity: visible ? 1 : 0,
    transition: `opacity .8s ease ${delay}s`,
  });

  const dashStyle = (visible, delay = 0) => ({
    strokeDasharray: 200,
    strokeDashoffset: visible ? 0 : 200,
    transition: `stroke-dashoffset .9s cubic-bezier(.2,.7,.3,1) ${delay}s, opacity .6s ease ${delay}s`,
    opacity: visible ? 1 : 0,
  });

  return (
    <svg viewBox={`0 0 ${VB.w} ${VB.h}`} className="tree-svg" preserveAspectRatio="xMidYMid meet">
      <defs>
        <marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="5" markerHeight="5" orient="auto">
          <path d="M 0 0 L 10 5 L 0 10 z" fill="var(--rule-strong)" />
        </marker>
        <marker id="arrow-accent" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="5" markerHeight="5" orient="auto">
          <path d="M 0 0 L 10 5 L 0 10 z" fill="var(--accent)" />
        </marker>
      </defs>

      {/* PDF → root connector */}
      {!pdfBig && (
        <path
          d={sPath(botMid(PDF), topMid(ROOT))}
          stroke="var(--rule-strong)" strokeWidth="2" fill="none"
          markerEnd="url(#arrow)"
          style={dashStyle(showRoot, 0.1)}
        />
      )}

      {/* Root → RQ edges */}
      {RQS.map((rq, i) => {
        const rqBox = { x: rq.x, y: RQ_Y, w: RQ_W, h: RQ_H };
        return (
          <path key={`r2q${i}`}
            d={sPath(botMid(ROOT), topMid(rqBox))}
            stroke="var(--rule-strong)" strokeWidth="2" fill="none"
            markerEnd="url(#arrow)"
            style={dashStyle(showRqs, 0.1 + i * 0.08)}
          />
        );
      })}

      {/* RQ → leaf edges */}
      {LEAVES.map((leaf, i) => {
        const parent = RQS[leaf.parentIdx];
        const parentBox = { x: parent.x, y: RQ_Y, w: RQ_W, h: RQ_H };
        const leafBox = { x: leaf.x, y: LEAF_Y, w: LEAF_W, h: LEAF_H };
        const highlighted = i === 0;
        return (
          <path key={`q2l${i}`}
            d={sPath(botMid(parentBox), topMid(leafBox))}
            stroke={highlighted ? 'var(--accent)' : 'var(--rule-strong)'}
            strokeWidth={highlighted ? 2.5 : 2}
            fill="none"
            markerEnd={highlighted ? 'url(#arrow-accent)' : 'url(#arrow)'}
            style={dashStyle(showLeaves, 0.15 + i * 0.08)}
          />
        );
      })}

      {/* Leaf → file edges (stage 3) */}
      {showFiles && FILES.map((file, i) => {
        const startLeaf = LEAVES[0];
        if (!startLeaf) return null;
        const startBox = { x: startLeaf.x, y: LEAF_Y, w: LEAF_W, h: LEAF_H };
        const fileBox = { x: file.x, y: file.y, w: FILE_W, h: FILE_H };
        return (
          <path key={`l2f${i}`}
            d={sPath(botMid(startBox), topMid(fileBox))}
            stroke={file.featured ? 'var(--accent)' : 'var(--rule-strong)'}
            strokeWidth={file.featured ? 2.5 : 2}
            fill="none"
            markerEnd={file.featured ? 'url(#arrow-accent)' : 'url(#arrow)'}
            style={dashStyle(showFiles, 0.3 + i * 0.1)}
          />
        );
      })}

      {/* (PDF rendered as HTML overlay outside the SVG so it can smoothly
         scale + translate between the big-centered and small-top states.) */}


      {/* ---- ROOT ---- */}
      <g style={fadeStyle(showRoot, 0.05)}>
        <g transform={`translate(${ROOT.x}, ${ROOT.y})`}>
          <rect width={ROOT.w} height={ROOT.h} rx="4"
            fill="var(--surface-2)" stroke="var(--fg)" strokeWidth="1.5" />
          <text x="14" y="20" fontFamily="var(--mono)" fontSize="10"
            fill="var(--muted)" letterSpacing="1.5">ROOT</text>
          <foreignObject x="14" y="24" width={ROOT.w - 28} height="36">
            <div xmlns="http://www.w3.org/1999/xhtml" style={{
              fontFamily: 'var(--display)', fontStyle: 'italic',
              fontSize: 15, color: 'var(--fg)', lineHeight: 1.2,
            }}>{paper.tree.root}</div>
          </foreignObject>
        </g>
      </g>

      {/* ---- RQs ---- */}
      {RQS.map((rq, i) => (
        <g key={`rq${i}`} style={fadeStyle(showRqs, 0.15 + i * 0.1)}>
          <g transform={`translate(${rq.x}, ${RQ_Y})`}>
            <rect width={RQ_W} height={RQ_H} rx="4"
              fill={i === 0 ? 'var(--accent-soft)' : 'var(--surface-2)'}
              stroke={i === 0 ? 'var(--accent)' : 'var(--rule)'}
              strokeWidth={i === 0 ? 2 : 1} />
            <text x="14" y="20" fontFamily="var(--mono)" fontSize="10"
              fill={i === 0 ? 'var(--accent)' : 'var(--muted)'} letterSpacing="1.5">
              {rq.label.toUpperCase()}
            </text>
            <foreignObject x="14" y="24" width={RQ_W - 28} height="36">
              <div xmlns="http://www.w3.org/1999/xhtml" style={{
                fontFamily: 'var(--sans)', fontSize: 13,
                color: 'var(--fg)', fontWeight: 500, lineHeight: 1.25,
              }}>{rq.desc}</div>
            </foreignObject>
          </g>
        </g>
      ))}

      {/* ---- Leaves ---- */}
      {LEAVES.map((leaf, i) => {
        const highlighted = i === 0;
        return (
          <g key={`lf${i}`} style={fadeStyle(showLeaves, 0.22 + i * 0.1)}>
            <g transform={`translate(${leaf.x}, ${LEAF_Y})`}>
              <rect width={LEAF_W} height={LEAF_H} rx="4"
                fill={highlighted ? 'var(--accent)' : 'var(--surface)'}
                stroke={highlighted ? 'var(--accent)' : 'var(--rule)'}
                strokeWidth="1.5" />
              <text x="10" y="18" fontFamily="var(--mono)" fontSize="9"
                fill={highlighted ? '#fff' : 'var(--muted)'} letterSpacing="1.2">EXPERIMENT</text>
              <foreignObject x="10" y="22" width={LEAF_W - 20} height="36">
                <div xmlns="http://www.w3.org/1999/xhtml" style={{
                  fontFamily: 'var(--sans)', fontSize: 12,
                  color: highlighted ? '#fff' : 'var(--fg)',
                  fontWeight: 500, lineHeight: 1.2,
                }}>{leaf.label}</div>
              </foreignObject>
              <foreignObject x="10" y="56" width={LEAF_W - 20} height="16">
                <div xmlns="http://www.w3.org/1999/xhtml" style={{
                  fontFamily: 'var(--mono)', fontSize: 10,
                  color: highlighted ? '#fff' : 'var(--accent)',
                }}>{leaf.metric}</div>
              </foreignObject>
            </g>
          </g>
        );
      })}

      {/* ---- File cards (stage 3) ---- */}
      {FILES.map((file, i) => (
        <g key={`file${i}`} style={fadeStyle(showFiles, 0.4 + i * 0.15)}>
          <g transform={`translate(${file.x}, ${file.y})`}>
            <rect width={FILE_W} height={FILE_H} rx="4"
              fill={file.featured ? 'var(--accent-soft)' : 'var(--surface)'}
              stroke={file.featured ? 'var(--accent)' : 'var(--rule)'}
              strokeWidth={file.featured ? 2 : 1.5} />
            <text x="14" y="22" fontFamily="var(--mono)" fontSize="10"
              fill={file.featured ? 'var(--accent)' : 'var(--muted)'}
              letterSpacing="1.5" fontWeight={file.featured ? 600 : 400}>
              {file.featured ? '+ EVALUATOR-ONLY ADDENDUM' : 'AGENT BRIEF'}
            </text>
            <foreignObject x="14" y="28" width={FILE_W - 28} height="60">
              <div xmlns="http://www.w3.org/1999/xhtml">
                <div style={{
                  fontFamily: 'var(--mono)', fontSize: 16, fontWeight: 600,
                  color: 'var(--fg)', marginTop: 4,
                }}>{file.name}</div>
                <div style={{
                  fontFamily: 'var(--mono)', fontSize: 11,
                  color: 'var(--faint)', marginTop: 6,
                }}>{file.sub}</div>
              </div>
            </foreignObject>
          </g>
        </g>
      ))}

      {/* ---- Annotations ---- */}
      {/* Stage 1 annotation moved to the HTML overlay alongside the PDF card. */}

      {/* Stage 2: tree-layer annotations */}
      <g style={fadeStyle(showAnnotTree, 0.15)}>
        {/* Root annotation */}
        <path
          d={`M ${ROOT.x + ROOT.w + 20} ${ROOT.y + 32} C ${ROOT.x + ROOT.w + 60} ${ROOT.y + 32}, ${ROOT.x + ROOT.w + 80} ${ROOT.y + 0}, ${ROOT.x + ROOT.w + 130} ${ROOT.y + 0}`}
          stroke="var(--accent)" strokeWidth="1.2" fill="none" strokeDasharray="3 3"
        />
        <foreignObject x={ROOT.x + ROOT.w + 110} y={ROOT.y - 50} width="220" height="100">
          <div xmlns="http://www.w3.org/1999/xhtml" className="annot">
            <span className="annot-label">↙ ROOT</span>
            <div className="annot-body">The paper's central question, extracted from the abstract and introduction.</div>
          </div>
        </foreignObject>

        {/* RQ annotation — last RQ */}
        {RQS.length > 0 && (() => {
          const rq = RQS[RQS.length - 1];
          return (
            <React.Fragment>
              <path
                d={`M ${rq.x + RQ_W + 12} ${RQ_Y + 32} C ${rq.x + RQ_W + 60} ${RQ_Y + 32}, ${rq.x + RQ_W + 100} ${RQ_Y + 10}, ${rq.x + RQ_W + 140} ${RQ_Y + 10}`}
                stroke="var(--accent)" strokeWidth="1.2" fill="none" strokeDasharray="3 3"
              />
              <foreignObject x={rq.x + RQ_W + 120} y={RQ_Y - 40} width="230" height="120">
                <div xmlns="http://www.w3.org/1999/xhtml" className="annot">
                  <span className="annot-label">↙ RESEARCH QUESTIONS</span>
                  <div className="annot-body">Sub-questions the paper investigates — usually 2–4 per paper.</div>
                </div>
              </foreignObject>
            </React.Fragment>
          );
        })()}

        {/* Leaf annotation — last leaf */}
        {LEAVES.length > 0 && (() => {
          const leaf = LEAVES[LEAVES.length - 1];
          return (
            <React.Fragment>
              <path
                d={`M ${leaf.x + LEAF_W + 12} ${LEAF_Y + 36} C ${leaf.x + LEAF_W + 50} ${LEAF_Y + 36}, ${leaf.x + LEAF_W + 80} ${LEAF_Y + 20}, ${leaf.x + LEAF_W + 130} ${LEAF_Y + 20}`}
                stroke="var(--accent)" strokeWidth="1.2" fill="none" strokeDasharray="3 3"
              />
              <foreignObject x={leaf.x + LEAF_W + 110} y={LEAF_Y - 20} width="240" height="120">
                <div xmlns="http://www.w3.org/1999/xhtml" className="annot">
                  <span className="annot-label">↙ EXPERIMENTS</span>
                  <div className="annot-body">Each leaf carries a dataset, model, metric, and a back-reference to the source figure or table.</div>
                </div>
              </foreignObject>
            </React.Fragment>
          );
        })()}
      </g>

    </svg>
  );
}

// ---------- Walkthrough section orchestrator ----------
function Walkthrough({ paper: paperProp }) {
  const paper = paperProp || (window.P2B_DATA && window.P2B_DATA.current);
  const [step, setStep]   = React.useState(0); // 0 / 1 / 2
  const [stage, setStage] = React.useState(1);
  const [tier, setTier]   = React.useState(0);
  const scrollerRef       = React.useRef(null);
  const stepRef           = React.useRef(0);
  React.useEffect(() => { stepRef.current = step; }, [step]);

  // Step → (stage, tier). When entering step 1 freshly, animate the tree
  // build (root → +rqs → +leaves). When returning to step 1 from step 2,
  // jump straight to the full tree (user has already seen the build).
  const lastStepRef = React.useRef(0);
  React.useEffect(() => {
    const prev = lastStepRef.current;
    lastStepRef.current = step;
    if (step === 0) {
      setStage(1); setTier(0);
    } else if (step === 2) {
      setStage(3); setTier(3);
    } else {
      setStage(2);
      if (prev <= 0) {
        setTier(0);
        const t1 = setTimeout(() => setTier(1), 80);
        const t2 = setTimeout(() => setTier(2), 650);
        const t3 = setTimeout(() => setTier(3), 1250);
        return () => { clearTimeout(t1); clearTimeout(t2); clearTimeout(t3); };
      }
      setTier(3);
    }
  }, [step]);

  // Discrete scroll-hijack with gesture detection. One trackpad swipe (or
  // one wheel click) = exactly one step. Trackpad inertia is suppressed:
  // a gesture is considered "active" while wheel events keep firing; only
  // a gap of > GESTURE_GAP_MS counts as the start of a new gesture.
  React.useEffect(() => {
    const el = scrollerRef.current;
    if (!el) return;

    const GESTURE_GAP_MS = 180;  // no-wheel gap that ends a gesture
    const COOLDOWN_MS    = 250;  // minimum time between fires (safety floor)
    const EXIT_GUARD_MS  = 1100; // grace period after we smooth-scroll out

    let lastWheelAt    = 0;
    let lastFireAt     = 0;
    let exitedAt       = 0;
    let gestureLive    = false; // true between gesture start and gap-end
    let gestureFired   = false; // we've already advanced for this gesture
    let gapTimer       = null;
    let touchStartY    = 0;
    let touchFired     = false;

    const isLocked = () => {
      const r = el.getBoundingClientRect();
      const vh = window.innerHeight || 800;
      return r.top <= 0 && r.bottom >= vh - 4;
    };
    const advance = (delta) => {
      const now = Date.now();
      if (now - lastFireAt < COOLDOWN_MS) return false;
      lastFireAt = now;
      setStep((s) => Math.max(0, Math.min(2, s + delta)));
      return true;
    };
    const exit = (downward) => {
      const r = el.getBoundingClientRect();
      const top = window.scrollY + r.top;
      const target = downward ? top + el.offsetHeight + 10 : top - 10;
      window.scrollTo({ top: target, behavior: 'smooth' });
      exitedAt = Date.now();
    };
    const armGapTimer = () => {
      clearTimeout(gapTimer);
      gapTimer = setTimeout(() => {
        gestureLive = false;
        gestureFired = false;
      }, GESTURE_GAP_MS);
    };

    const onWheel = (e) => {
      // Guard period after a programmatic exit so the page's residual
      // inertia doesn't immediately re-hijack scroll on the way out.
      if (Date.now() - exitedAt < EXIT_GUARD_MS) {
        if (isLocked()) e.preventDefault();
        return;
      }
      if (!isLocked()) return;
      if (Math.abs(e.deltaY) < 4) return;

      e.preventDefault();
      lastWheelAt = Date.now();
      gestureLive = true;
      armGapTimer();
      if (gestureFired) return;   // ignore inertia after the first fire
      gestureFired = true;

      const down = e.deltaY > 0;
      if (down && stepRef.current === 2) { exit(true);  return; }
      if (!down && stepRef.current === 0) { exit(false); return; }
      advance(down ? 1 : -1);
    };

    const onTouchStart = (e) => {
      if (!e.touches[0]) return;
      touchStartY = e.touches[0].clientY;
      touchFired = false;
    };
    const onTouchMove = (e) => {
      if (Date.now() - exitedAt < EXIT_GUARD_MS) {
        if (isLocked()) e.preventDefault();
        return;
      }
      if (!isLocked() || !e.touches[0]) return;
      const dy = touchStartY - e.touches[0].clientY;
      if (Math.abs(dy) < 28) { e.preventDefault(); return; }
      e.preventDefault();
      if (touchFired) return;
      touchFired = true;
      const down = dy > 0;
      if (down && stepRef.current === 2) { exit(true);  return; }
      if (!down && stepRef.current === 0) { exit(false); return; }
      advance(down ? 1 : -1);
    };

    const onKey = (e) => {
      if (!isLocked()) return;
      const fwd  = e.key === 'ArrowDown' || e.key === 'PageDown' || e.key === ' ';
      const back = e.key === 'ArrowUp'   || e.key === 'PageUp';
      if (!fwd && !back) return;
      e.preventDefault();
      const now = Date.now();
      if (now - lastFireAt < COOLDOWN_MS) return;
      const down = fwd;
      if (down && stepRef.current === 2) { exit(true);  return; }
      if (!down && stepRef.current === 0) { exit(false); return; }
      advance(down ? 1 : -1);
    };

    window.addEventListener('wheel', onWheel, { passive: false });
    window.addEventListener('touchstart', onTouchStart, { passive: true });
    window.addEventListener('touchmove', onTouchMove, { passive: false });
    window.addEventListener('keydown', onKey);
    return () => {
      window.removeEventListener('wheel', onWheel);
      window.removeEventListener('touchstart', onTouchStart);
      window.removeEventListener('touchmove', onTouchMove);
      window.removeEventListener('keydown', onKey);
      clearTimeout(gapTimer);
    };
  }, []);

  const current = WSTAGES[stage - 1];

  return (
    <div className="walk-scroller" ref={scrollerRef}>
      <div className="walk-sticky">
        {/* Stage indicator pill — click to jump to a step */}
        <div className="walk-stages-bar">
          {WSTAGES.map((s, i) => (
            <div
              key={s.n}
              onClick={() => setStep(i)}
              role="button"
              className={`walk-stage-dot ${step === i ? 'active' : ''} ${step > i ? 'past' : ''}`}
            >
              <span className="dot" />
              <span className="dot-label">
                <span className="n">{String(s.n).padStart(2, '0')}</span> {s.label}
              </span>
            </div>
          ))}
        </div>

        {/* Big tree canvas */}
        <div className="walk-canvas">
          <div className={`pdf-overlay ${stage === 1 ? 'big' : ''}`} aria-hidden={stage !== 1 && stage !== 2}>
            <div className="pdf-card-html">
              <PdfIconHtml size={44} />
              <div className="pdf-meta">
                <div className="pdf-name">{paper.id}.pdf</div>
                <div className="pdf-sub">{paper.authors} · {paper.year}</div>
              </div>
            </div>
          </div>
          <div className={`pdf-annot ${stage === 1 ? 'show' : ''}`}>
            <span className="annot-label">↘ STEP 1</span>
            <div className="annot-title">Pass a paper.</div>
            <div className="annot-body">Local PDF or arXiv title. paper2bench reads it once.</div>
          </div>
          <CanvasViz stage={stage} tier={tier} paper={paper} />
        </div>

        {/* Stage description card */}
        <div className="walk-stage-card">
          <div className="walk-stage-card-inner" key={current.n /* re-mount for fade */}>
            <span className="label accent">Stage {current.n} / 3 · {current.label}</span>
            <p>{current.desc}</p>
          </div>
        </div>
      </div>
    </div>
  );
}

window.Walkthrough = Walkthrough;
