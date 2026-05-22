// Paper2Bench — Editorial site shell
// Composes: Nav, Hero, Stats, Walkthrough, Diff, Examples, Install, Footer.

function Nav() {
  const [mode, setMode] = React.useState(() => document.documentElement.getAttribute('data-mode') || 'light');
  const toggle = () => {
    const next = mode === 'dark' ? 'light' : 'dark';
    setMode(next);
    document.documentElement.setAttribute('data-mode', next);
    try { localStorage.setItem('p2b-mode', next); } catch {}
  };
  React.useEffect(() => {
    try {
      const saved = localStorage.getItem('p2b-mode');
      if (saved) { setMode(saved); document.documentElement.setAttribute('data-mode', saved); }
    } catch {}
  }, []);
  const jump = (id) => () => {
    document.getElementById(id)?.scrollIntoView({ behavior: 'smooth', block: 'start' });
  };
  return (
    <nav className="nav">
      <div className="container nav-inner">
        <div>
          <a href="index.html" className="brand-link" aria-label="Paper2Bench home">
            <span className="brand">Paper2Bench</span>
          </a>
        </div>
        <div className="nav-links">
          <a className="icon-link" href="https://github.com/AbhayAnandUCSD/Paper2Bench" target="_blank" rel="noreferrer" aria-label="GitHub repository" title="GitHub">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor" aria-hidden>
              <path d="M12 .5C5.65.5.5 5.65.5 12c0 5.08 3.29 9.39 7.86 10.91.58.1.79-.25.79-.56v-2.16c-3.2.7-3.87-1.36-3.87-1.36-.52-1.32-1.27-1.67-1.27-1.67-1.04-.71.08-.7.08-.7 1.15.08 1.76 1.18 1.76 1.18 1.03 1.76 2.7 1.25 3.36.96.1-.75.4-1.25.73-1.54-2.55-.29-5.24-1.28-5.24-5.69 0-1.26.45-2.29 1.18-3.09-.12-.29-.51-1.46.11-3.04 0 0 .97-.31 3.18 1.18a11 11 0 0 1 5.79 0c2.21-1.49 3.18-1.18 3.18-1.18.62 1.58.23 2.75.11 3.04.74.8 1.18 1.83 1.18 3.09 0 4.42-2.69 5.4-5.26 5.68.41.36.78 1.06.78 2.14v3.18c0 .31.21.67.8.56 4.57-1.52 7.85-5.83 7.85-10.91C23.5 5.65 18.35.5 12 .5z"/>
            </svg>
          </a>
          <button className="icon-toggle" onClick={toggle} aria-label={`Switch to ${mode === 'dark' ? 'light' : 'dark'} mode`} title={mode === 'dark' ? 'Light mode' : 'Dark mode'}>
            {mode === 'dark' ? (
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
                <circle cx="12" cy="12" r="4"/>
                <path d="M12 2v2M12 20v2M4.93 4.93l1.41 1.41M17.66 17.66l1.41 1.41M2 12h2M20 12h2M4.93 19.07l1.41-1.41M17.66 6.34l1.41-1.41"/>
              </svg>
            ) : (
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
                <path d="M21 12.79A9 9 0 1 1 11.21 3a7 7 0 0 0 9.79 9.79z"/>
              </svg>
            )}
          </button>
        </div>
      </div>
    </nav>
  );
}

function Copyable({ cmd, dark }) {
  const [copied, setCopied] = React.useState(false);
  const copy = async () => {
    try {
      await navigator.clipboard.writeText(cmd);
      setCopied(true);
      setTimeout(() => setCopied(false), 1400);
    } catch {}
  };
  return (
    <div className="code-block" style={dark ? { background: 'var(--surface-2)', borderColor: 'var(--rule-strong)' } : null}>
      <span className="prompt">$</span>
      <span>{cmd}</span>
      <button className={`copy-btn ${copied ? 'copied' : ''}`} onClick={copy}>
        {copied ? '✓ copied' : 'copy'}
      </button>
    </div>
  );
}

function TypeCycle({ words, intervalMs = 1800, typeMs = 75, eraseMs = 40 }) {
  const [idx, setIdx] = React.useState(0);
  const [text, setText] = React.useState(words[0]);
  const [phase, setPhase] = React.useState('hold'); // typing | hold | erasing
  React.useEffect(() => {
    const current = words[idx];
    let t;
    if (phase === 'hold') {
      t = setTimeout(() => setPhase('erasing'), intervalMs);
    } else if (phase === 'erasing') {
      if (text === '') {
        const next = (idx + 1) % words.length;
        setIdx(next); setPhase('typing');
      } else {
        t = setTimeout(() => setText(text.slice(0, -1)), eraseMs);
      }
    } else {
      if (text === current) {
        setPhase('hold');
      } else {
        t = setTimeout(() => setText(current.slice(0, text.length + 1)), typeMs);
      }
    }
    return () => clearTimeout(t);
  }, [text, phase, idx, words, intervalMs, typeMs, eraseMs]);
  // Longest word reserves layout space so cycling never re-flows the
  // heading vertically when a shorter word is typed.
  const longest = React.useMemo(
    () => words.reduce((a, w) => (w.length > a.length ? w : a), ''),
    [words]
  );
  return (
    <span className="type-cycle">
      <span className="type-cycle-ghost" aria-hidden>{longest}</span>
      <span className="type-cycle-text italic-accent">
        {text}
        <span className="type-cursor" aria-hidden>|</span>
      </span>
    </span>
  );
}

function Hero() {
  const D = window.P2B_DATA;
  return (
    <header className="hero container">
      <div className="hero-grid">
        <div className="hero-copy">
          <h1 className="reveal">
            Every paper is a <span className="italic-accent">benchmark</span> waiting to be{' '}
            <TypeCycle words={['unpacked', 'decomposed', 'reproduced', 'scored', 'evaluated']} />.
          </h1>
          <p className="lede reveal" data-delay="1">
            Convert any research paper across materials, biology, and ML into a reliable benchmark you can score agents against.
          </p>
        </div>
        <div className="install-card reveal" data-delay="1">
          <div className="install-card-head">
            <span className="label accent">Get started</span>
          </div>
          <h3>One command.</h3>
          <div className="install-step">
            <span className="install-step-num">1</span>
            <div className="install-step-body">
              <span className="install-step-label">Install</span>
              <Copyable cmd={D.install.pip} />
            </div>
          </div>
          <div className="install-step">
            <span className="install-step-num">2</span>
            <div className="install-step-body">
              <span className="install-step-label">Run on a paper</span>
              <Copyable cmd="paper2bench run --pdf paper.pdf --auto" />
            </div>
          </div>
        </div>
      </div>
      <a className="hero-scroll-cue" onClick={() => document.getElementById('walkthrough')?.scrollIntoView({ behavior: 'smooth' })}>
        <span>Scroll to see the pipeline</span>
        <span className="hero-scroll-arrow">↓</span>
      </a>
    </header>
  );
}

function WalkthroughSection() {
  return (
    <section className="section walkthrough-section container" id="walkthrough">
      <window.Walkthrough />
    </section>
  );
}

function DiffSection() {
  return (
    <section className="section container" id="diff">
      <h2 className="section-title reveal">
        We generate <span className="italic-accent">agent instructions</span> and a <span className="italic-accent">ground-truth key</span> for evaluation.
      </h2>
      <div className="reveal" data-delay="1" style={{ marginTop: 36 }}>
        <window.FullDiff />
      </div>
    </section>
  );
}

function Examples() {
  const D = window.P2B_DATA;
  return (
    <section className="section container" id="examples">
      <h2 className="section-title reveal">
        Five papers, run end-to-end.
      </h2>
      <div className="examples-grid" style={{ marginTop: 36 }}>
        {D.papers.map((paper, i) => (
          <a
            key={paper.id}
            className="example-card reveal"
            data-delay={Math.min(i, 5)}
            href={`example.html?paper=${paper.id}`}
          >
            <div className="example-card-num">{String(i + 1).padStart(2, '0')}</div>
            <div className="example-card-title">{paper.title}</div>
            <div className="example-card-meta">{paper.authors} · {paper.year} · {paper.domain}</div>
            <div className="example-card-rq">"{paper.rq}"</div>
            <div className="example-card-foot">
              <span>{paper.type.replace('_', ' ')}</span>
              <span>{paper.counts.rqs}·{paper.counts.leaves}·{paper.counts.datasets}</span>
            </div>
          </a>
        ))}
      </div>
    </section>
  );
}

function Install() {
  const D = window.P2B_DATA;
  return (
    <section className="section container" id="install">
      <h2 className="section-title reveal">
        Two-line install. <span className="italic-accent">Nine stages</span> on one command.
      </h2>

      <div className="reveal" data-delay="1" style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 32, marginTop: 8 }}>
        <div>
          <span className="label" style={{ display: 'block', marginBottom: 10 }}>1 · install from GitHub</span>
          <Copyable cmd={D.install.pip} />
          <div style={{ marginTop: 14, fontSize: 12.5, color: 'var(--muted)', lineHeight: 1.5 }}>
            For Claude support, install the optional <code style={{ background: 'var(--surface-2)', padding: '1px 5px', fontSize: 11 }}>anthropic</code> extra:
          </div>
          <div style={{ marginTop: 6 }}>
            <Copyable cmd={D.install.pipa} dark />
          </div>
        </div>
        <div>
          <span className="label" style={{ display: 'block', marginBottom: 10 }}>2 · one-shot pipeline</span>
          <div className="code-block" style={{ whiteSpace: 'pre' }}>
            <span className="prompt">$</span>
            paper2bench run \{'\n'}
            {'  '}--task-id reversal_curse \{'\n'}
            {'  '}--query "The Reversal Curse Berglund" \{'\n'}
            {'  '}--research-question "If an LLM is{'\n'}
            {'    '}fine-tuned on 'A is B,' does it{'\n'}
            {'    '}learn 'B is A'?" \{'\n'}
            {'  '}--output-dir ./output/ --auto
          </div>
          <div style={{ marginTop: 14, fontSize: 12.5, color: 'var(--muted)', lineHeight: 1.5 }}>
            Or use the Python SDK: <code style={{ background: 'var(--surface-2)', padding: '1px 5px', fontSize: 11 }}>import paper2bench; paper2bench.run(pdf=...)</code>.
          </div>
        </div>
      </div>

      <div className="reveal" data-delay="2" style={{ marginTop: 32 }}>
        <span className="label" style={{ display: 'block', marginBottom: 14 }}>Or run any stage on its own</span>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(260px, 1fr))', gap: 10 }}>
          {Object.entries({
            download: D.cli.download,
            parse:    D.cli.parse,
            classify: D.cli.classify,
            extract:  D.cli.extract,
            render:   D.cli.render,
            plan:     D.cli.plan,
            verify:   D.cli.verify,
          }).map(([k, v]) => (
            <div key={k} style={{ background: 'var(--surface)', border: '1px solid var(--rule)', padding: '12px 14px', borderRadius: 2 }}>
              <span className="label accent" style={{ display: 'block' }}>{k}</span>
              <code style={{ display: 'block', marginTop: 6, fontSize: 11, color: 'var(--fg)', wordBreak: 'break-all', lineHeight: 1.5 }}>{v}</code>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}

function Footer() {
  return (
    <footer className="container footer">
      <span>paper2bench · v0.3.0 · MIT</span>
      <span style={{ color: 'var(--faint)' }}>made for researchers who'd rather be reading papers than writing eval harnesses</span>
      <span>
        <a href="https://github.com/AbhayAnandUCSD/Paper2Bench" style={{ color: 'var(--accent)' }}>GitHub ↗</a>
      </span>
    </footer>
  );
}

function MadeBy() {
  const authors = [
    { name: 'Zhen Wang',     url: 'https://zhenwang9102.github.io/' },
    { name: 'Abhay Anand',   url: 'https://abhayanand.com' },
    { name: 'Zhongyan Luo',  url: 'https://www.linkedin.com/in/zhongyan-luo-a85439264/' },
  ];
  return (
    <section className="section container made-by-section">
      <div className="made-by reveal">
        <span className="label">Made by</span>
        <div className="made-by-names">
          {authors.map((a, i) => (
            <React.Fragment key={a.url}>
              <a className="made-by-name" href={a.url} target="_blank" rel="noreferrer">
                {a.name}
                <span className="made-by-arrow" aria-hidden> ↗</span>
              </a>
              {i < authors.length - 1 && <span className="made-by-sep">·</span>}
            </React.Fragment>
          ))}
        </div>
      </div>
    </section>
  );
}

function App() {
  return (
    <React.Fragment>
      <Nav />
      <Hero />
      <WalkthroughSection />
      <DiffSection />
      <Examples />
      <MadeBy />
    </React.Fragment>
  );
}

window.App = App;
window.Nav = Nav;
window.Footer = Footer;
window.Copyable = Copyable;
