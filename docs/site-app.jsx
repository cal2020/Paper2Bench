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
            <span className="brand-version">v{window.P2B_RELEASE.version}</span>
          </a>
        </div>
        <div className="nav-links">
          {window.P2B_REAL_PAPER && <a href="index.html#real-paper" className="paper-nav-link">Paper run</a>}
          <a href="index.html#demo" className="demo-nav-link">Try the demo</a>
          <a href={window.P2B_RELEASE.repository + "/blob/main/IMPLEMENTATION.md"} target="_blank" rel="noreferrer" className="guide-nav-link">Guide</a>
          <a className="icon-link" href={window.P2B_RELEASE.repository} target="_blank" rel="noreferrer" aria-label="GitHub repository" title="GitHub">
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
          <div className="release-eyebrow reveal"><span className="release-dot" /> Portable execution alpha · v{window.P2B_RELEASE.version}</div>
          <h1 className="reveal">
            Papers become <span className="italic-accent">protocols.</span><br />Evidence follows.
          </h1>
          <p className="lede reveal" data-delay="1">
            Build reviewable benchmark drafts, run isolated computational tasks, and keep each result tied to its source, method and runtime.
          </p>
          {window.P2B_REAL_PAPER && <a className="text-link paper-hero-link reveal" href="#real-paper">Inspect the recorded real-paper run ↘</a>}
        </div>
        <div className="install-card reveal" data-delay="1">
          <div className="install-card-head">
            <span className="label accent">Start with a real fixture</span>
          </div>
          <h3>No model required.</h3>
          <div className="install-step">
            <span className="install-step-num">1</span>
            <div className="install-step-body">
              <span className="install-step-label">Install this extension</span>
              <Copyable cmd={D.install.pip} />
            </div>
          </div>
          <div className="install-step">
            <span className="install-step-num">2</span>
            <div className="install-step-body">
              <span className="install-step-label">Check the execution runtime</span>
              <Copyable cmd="paper2bench doctor --execution" />
            </div>
          </div>
        </div>
      </div>
      <a className="hero-scroll-cue" href="#demo">
        <span>Explore the demo and evidence</span>
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
        Inspect <span className="italic-accent">agent instructions</span> beside a <span className="italic-accent">reference draft.</span>
      </h2>
      <div className="reveal" data-delay="1" style={{ marginTop: 36 }}>
        <p className="scope-note">These upstream-derived examples illustrate authoring output. Reference plans need curator review; conclusion similarity alone does not validate execution or scientific fidelity.</p>
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
        Five paper drafts, open for inspection.
      </h2>
      <p className="section-lede reveal">Explore illustrative extracted questions, problem trees and reference plans from the original project. These are not validated reproductions of the papers.</p>
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
  const R = window.P2B_RELEASE;
  return (
    <section className="section container" id="install">
      <div className="section-head reveal"><span className="label accent">Take it to your own runtime</span><span className="rule" /></div>
      <h2 className="section-title reveal">Inspect the source. <span className="italic-accent">Run the fixtures.</span></h2>
      <div className="install-grid reveal">
        <div><span className="label">1 · clone this extension</span><Copyable cmd={'git clone ' + R.repository + '.git'} /><Copyable cmd="cd Paper2Bench && python -m pip install -e '.[dev]'" /></div>
        <div><span className="label">2 · stage a trusted worker image</span><Copyable cmd="docker pull python:3.11-slim" /><Copyable cmd="paper2bench doctor --execution" /></div>
      </div>
      <p className="section-lede reveal">Python 3.10+ and a running Docker engine are required for the isolated examples. Foundation-model authoring uses an explicit runtime profile; choose your model and export its credential when hosted inference is needed.</p>
      <div className="release-links reveal"><a className="primary-button" href={R.repository + '/blob/main/IMPLEMENTATION.md'} target="_blank" rel="noreferrer">Read the implementation guide ↗</a><a className="text-link" href={R.repository + '/tree/main/examples'} target="_blank" rel="noreferrer">Browse runtime profiles & examples ↗</a></div>
    </section>
  );
}

function Footer() {
  return (
    <footer className="container footer">
      <span>Paper2Bench · v{window.P2B_RELEASE.version} · MIT</span>
      <span style={{ color: 'var(--faint)' }}>Portable execution extension · upstream attribution preserved</span>
      <span>
        <a href={window.P2B_RELEASE.repository} style={{ color: 'var(--accent)' }}>GitHub ↗</a>
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
        <span className="label">Original upstream authors</span>
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
      <RealPaperStudy />
      <ExecutionDemo />
      <ReleaseCapabilities />
      <WalkthroughSection />
      <DiffSection />
      <Examples />
      <Install />
      <MadeBy />
      <Footer />
    </React.Fragment>
  );
}

function ReleaseCapabilities() {
  const entries = [
    ['01', 'Choose the runtime', 'Route author, reference, agent and judge roles to local Ollama, hosted OpenAI or Anthropic, or explicit compatible endpoints. No automatic cloud fallback.'],
    ['02', 'Keep evidence attached', 'Index source spans and exact citations. Select bounded context with visible omissions, then keep generated requirements in a reviewable draft state.'],
    ['03', 'Run within a boundary', 'Use isolated Docker workers, pinned SSH workers, or explicitly selected NVIDIA devices. The controller broker keeps model credentials outside candidate code.'],
    ['04', 'Preserve the experiment', 'Freeze reviewed protocols and fixed seed schedules. Recompute metrics with installed scorers, checkpoint authoring stages, and retain unknown outcomes for reconciliation.']
  ];
  return (
    <section className="section container" id="capabilities" aria-labelledby="capabilities-title">
      <div className="section-head reveal"><span className="label accent">The portable execution extension</span><span className="rule" /><span className="label">v{window.P2B_RELEASE.version}</span></div>
      <h2 className="section-title reveal" id="capabilities-title">From a paper draft to <span className="italic-accent">inspectable evidence.</span></h2>
      <div className="capability-list">
        {entries.map(([n, title, description]) => (
          <article className="capability-row reveal" key={n}>
            <span className="capability-number">{n}</span><h3>{title}</h3><p>{description}</p>
          </article>
        ))}
      </div>
      <p className="scope-note reveal">Local Docker and Llama fixtures were exercised. Hosted accounts, external SSH hosts and physical NVIDIA execution remain unverified. Full training and simulation integrations require additional method plugins.</p>
    </section>
  );
}

function ExecutionDemo() {
  const R = window.P2B_RELEASE;
  const [scenarioId, setScenarioId] = React.useState('linear');
  const [result, setResult] = React.useState(null);
  const scenario = R.scenarios.find(item => item.id === scenarioId);
  const choose = (id) => { setScenarioId(id); setResult(null); };
  const run = () => setResult(window.P2B_runBrowserFixture(scenarioId));
  return (
    <section className="section container" id="demo" aria-labelledby="demo-title">
      <div className="section-head reveal"><span className="label accent">Try the small things. Inspect the hard things.</span><span className="rule" /><span className="label">Browser only · no API key</span></div>
      <h2 className="section-title reveal" id="demo-title">Explore the <span className="italic-accent">execution evidence.</span></h2>
      <p className="section-lede reveal">Run two small computations in your browser, or inspect recorded protocol and model-broker fixtures. This static demo never launches a container or sends a model request.</p>
      <div className="execution-demo reveal">
        <div className="demo-selector" role="group" aria-label="Choose an execution scenario">
          {R.scenarios.map(item => <button type="button" key={item.id} aria-pressed={scenarioId === item.id} className={scenarioId === item.id ? 'scenario-button active' : 'scenario-button'} onClick={() => choose(item.id)}>{item.title}<span>{item.kind}</span></button>)}
        </div>
        <div className="demo-detail">
          <div className="demo-heading"><span className="label accent">{scenario.paradigm}</span><span className="evidence-chip">{scenario.kind}</span></div>
          <h3>{scenario.title}</h3>
          <p className="demo-description">{scenario.description}</p>
          <ol className="demo-stages">{scenario.stages.map((step, i) => <li key={step}><span>{String(i + 1).padStart(2, '0')}</span>{step}</li>)}</ol>
          <dl className="demo-facts">{scenario.facts.map(([label, value]) => <div key={label}><dt>{label}</dt><dd>{value}</dd></div>)}</dl>
          <div className="demo-action-row"><button className="primary-button" type="button" onClick={run}>{scenario.button}<span aria-hidden> ↗</span></button><a className="text-link" href={R.repository + '/blob/main/' + scenario.source} target="_blank" rel="noreferrer">Inspect source ↗</a></div>
          <div className="demo-result" role="status" aria-live="polite" aria-atomic="true">
            {result ? <React.Fragment><span className="label accent">{result.origin}</span><strong>{result.result}</strong><p>{result.detail}</p></React.Fragment> : <p>Choose a scenario and use the button above to see its scoped result.</p>}
          </div>
          <p className="demo-scope">{scenario.scope}</p>
          <details className="demo-reproduce"><summary>Run the Python fixture yourself</summary><Copyable cmd={scenario.command} /><p>Use the repository checkout and staged Docker image. The local-model probe additionally needs the named Ollama model installed. See the <a className="text-link" href={R.repository + '/blob/main/IMPLEMENTATION.md'} target="_blank" rel="noreferrer">implementation guide</a> for setup and limits.</p></details>
        </div>
      </div>
      <div className="verification-strip reveal"><div><strong>{R.tests}</strong><span>passing tests</span></div><div><strong>3</strong><span>fixed synthetic seeds</span></div><div><strong>1 → 0</strong><span>broker call → replay calls</span></div><p>Release tests · {R.testDate}. Recorded local fixtures · {R.evidenceDate}. Synthetic fixtures establish software behavior within the stated scope, not published-paper reproduction. <a className="text-link" href="verification.json" target="_blank" rel="noreferrer">View the sanitized evidence summary ↗</a></p></div>
    </section>
  );
}

function paperNumber(value, precision = 6) {
  if (typeof value !== 'number' || !Number.isFinite(value)) return value == null ? 'Not recorded' : String(value);
  const digits = Number.isInteger(precision) ? Math.max(0, Math.min(12, precision)) : 6;
  return value.toLocaleString('en-US', { maximumFractionDigits: digits });
}

function RealPaperStudy() {
  const study = window.P2B_REAL_PAPER;
  if (!study || study.schema_version !== 1 || !study.paper?.title) return null;
  return <RealPaperStudyContent study={study} />;
}

function RealPaperStudyContent({ study }) {
  const [metric, setMetric] = React.useState('position');
  const checkpoints = React.useMemo(() => (Array.isArray(study.checkpoints) ? study.checkpoints : [])
    .filter(point => Number.isInteger(point.step) && point.step >= 0)
    .sort((a, b) => a.step - b.step), [study]);
  const [selectedIndex, setSelectedIndex] = React.useState(Math.max(0, checkpoints.length - 1));
  const selected = checkpoints[Math.min(selectedIndex, checkpoints.length - 1)];
  const paper = study.paper;
  const authoring = study.authoring || {};
  const execution = study.execution || {};
  const hasRegret = checkpoints.length > 1 && checkpoints.every(point =>
    Number.isFinite(point.adam_average_regret) && Number.isFinite(point.amsgrad_average_regret));
  const hasPositions = checkpoints.length > 1 && checkpoints.every(point =>
    Number.isFinite(point.adam_position) && Number.isFinite(point.amsgrad_position));
  const canPlot = metric === 'position' ? hasPositions : hasRegret;
  const sourceHref = typeof paper.url === 'string' && /^https:\/\//.test(paper.url) ? paper.url : undefined;
  const sourcePath = study.reproduction?.source_path;
  const sourceCodeHref = typeof sourcePath === 'string' && /^[a-zA-Z0-9_/-]+$/.test(sourcePath) && !sourcePath.split('/').includes('..')
    ? window.P2B_RELEASE.repository + '/tree/main/' + sourcePath : undefined;
  const metrics = Array.isArray(study.metrics) ? study.metrics : [];
  const steps = [
    ['01', 'Source paper', [paper.venue, paper.year].filter(Boolean).join(' · '), 'PDF identity pinned'],
    ['02', 'Parse · extract · plan', authoring.model || 'Authoring runtime recorded in receipt', authoring.status || 'See receipt'],
    ['03', 'Isolated execution', execution.steps ? paperNumber(execution.steps, 0) + ' iterations per run' : 'See recorded protocol', execution.status || 'See receipt'],
    ['04', 'Trusted metrics', 'Recomputed from recorded trajectories', 'Controller-side scoring'],
  ];
  const identities = [
    ['Paper PDF', paper.pdf_sha256], ['Task', execution.task_sha256],
    ['Submission', execution.submission_sha256], ['Protocol', execution.protocol_sha256],
    ['Container image', execution.image_id], ['Model digest', authoring.model_digest],
  ].filter(([, value]) => value);
  return (
    <section className="section container real-paper" id="real-paper" aria-labelledby="real-paper-title">
      <div className="section-head reveal"><span className="label accent">Recorded research case study</span><span className="rule" /><span className="label">A published paper · an explicit scope</span></div>
      <h2 className="section-title reveal" id="real-paper-title">A real paper, <span className="italic-accent">through the pipeline.</span></h2>
      <div className="paper-intro reveal">
        <div>
          <p className="paper-citation">{(Array.isArray(paper.authors) ? paper.authors : []).join(', ')}{paper.year ? ' · ' + paper.year : ''}{paper.venue ? ' · ' + paper.venue : ''}</p>
          <h3>{sourceHref ? <a href={sourceHref} target="_blank" rel="noreferrer">{paper.title} ↗</a> : paper.title}</h3>
          <p className="paper-question">{study.question}</p>
        </div>
        <aside className="paper-scope-box">
          <span className="label accent">What this run establishes</span>
          <p>{study.scope || 'A finite, paper-grounded computational experiment.'}</p>
          <p className="paper-finite-warning">Finite-horizon evidence, not a proof of asymptotic convergence or a full-paper reproduction.</p>
        </aside>
      </div>
      <ol className="paper-pipeline reveal" aria-label="Recorded paper pipeline">
        {steps.map(([number, label, detail, status]) => <li key={number}><span className="label accent">{number}</span><h4>{label}</h4><p>{detail}</p><span className="paper-step-state">{status}</span></li>)}
      </ol>
      <div className="paper-metrics reveal" aria-label="Recorded trusted metrics">
        {metrics.map((item, i) => <div key={item.label + '-' + i} data-testid={'paper-metric-' + i}><span>{item.label}</span><strong>{paperNumber(item.value, item.precision)}</strong>{item.unit && <small>{item.unit}</small>}</div>)}
      </div>
      <div className="paper-chart-card reveal">
        <div className="paper-chart-heading">
          <div><span className="label accent">Inspect the recorded checkpoints</span><h3>Two optimizers, one prescribed sequence.</h3></div>
          <div className="paper-chart-tabs" role="group" aria-label="Recorded trace metric">
            <button type="button" onClick={() => setMetric('position')} aria-pressed={metric === 'position'} disabled={!hasPositions}>Position</button>
            <button type="button" onClick={() => setMetric('regret')} aria-pressed={metric === 'regret'} disabled={!hasRegret}>Average regret</button>
          </div>
        </div>
        {canPlot ? <RecordedPaperChart checkpoints={checkpoints} metric={metric} selectedIndex={Math.min(selectedIndex, checkpoints.length - 1)} /> : <p className="paper-chart-empty">No numeric trace was published for this run.</p>}
        {checkpoints.length > 1 && <div className="paper-checkpoint-control">
          <label htmlFor="paper-checkpoint">Recorded checkpoint <strong>{selected ? 'Iteration ' + paperNumber(selected.step, 0) : ''}</strong></label>
          <input id="paper-checkpoint" type="range" min="0" max={checkpoints.length - 1} step="1" value={Math.min(selectedIndex, checkpoints.length - 1)} onChange={event => setSelectedIndex(Number(event.target.value))} aria-label="Recorded checkpoint" aria-valuetext={selected ? 'Iteration ' + selected.step : ''} />
          <div className="paper-checkpoint-values" aria-live="polite">
            <span><i className="paper-swatch adam" />Adam <strong>{paperNumber(selected?.[metric === 'position' ? 'adam_position' : 'adam_average_regret'])}</strong></span>
            <span><i className="paper-swatch amsgrad" />AMSGrad <strong>{paperNumber(selected?.[metric === 'position' ? 'amsgrad_position' : 'amsgrad_average_regret'])}</strong></span>
          </div>
        </div>}
        <p className="paper-chart-caption">Recorded checkpoints are joined for inspection; the plot does not rerun the experiment. Values shown here are rounded. Complete numeric evidence is available in the receipt.</p>
      </div>
      <div className="paper-assurance reveal">
        <div><h3>Authored, then checked.</h3><p>{authoring.review || 'Model-generated authoring artifacts are drafts. The executed method and protocol require a separate source and implementation review.'}</p><p>{study.assurance?.method_fidelity || 'No independent human expert certification is asserted.'}</p></div>
        <div><h3>Every limit stays visible.</h3><ul>{(Array.isArray(study.limitations) ? study.limitations : []).map((limit, i) => <li key={i}>{limit}</li>)}</ul><p>{study.assurance?.scientific_scope}</p></div>
      </div>
      {Array.isArray(authoring.stages) && authoring.stages.length > 0 && <ol className="paper-authoring-stages reveal" aria-label="Recorded authoring stages">{authoring.stages.map((stage, i) => <li key={i}><strong>{stage.label}</strong><span>{stage.status}</span>{stage.detail && <p>{stage.detail}</p>}</li>)}</ol>}
      <div className="paper-evidence-bar reveal">
        <a className="primary-button" href="results/adam-counterexample.json" download>Download recorded run JSON ↓</a>
        {study.reproduction?.raw_predictions_path && <a className="text-link" href={study.reproduction.raw_predictions_path} download>Download full trajectories ↓</a>}
        {sourceCodeHref && <a className="text-link" href={sourceCodeHref} target="_blank" rel="noreferrer">Inspect the executable task ↗</a>}
        {study.recorded_at && <span className="paper-recorded-at">Recorded <time dateTime={study.recorded_at}>{study.recorded_at}</time></span>}
      </div>
      <details className="paper-run-details reveal"><summary>Inspect source, runtime, and run identities</summary>
        <dl>{identities.map(([label, value]) => <div key={label}><dt>{label}</dt><dd><code>{value}</code></dd></div>)}</dl>
        <p>{authoring.model_calls != null ? paperNumber(authoring.model_calls, 0) + ' recorded model calls. ' : ''}{authoring.input_tokens != null ? paperNumber(authoring.input_tokens, 0) + ' input tokens. ' : ''}{authoring.output_tokens != null ? paperNumber(authoring.output_tokens, 0) + ' output tokens. ' : ''}{execution.cleanup_confirmed === true ? 'Worker cleanup confirmed.' : 'Worker cleanup: see receipt.'}</p>
        {Array.isArray(execution.seed_schedule) && <p>Prescribed seeds: {execution.seed_schedule.join(', ')}. A deterministic sequence is not a set of independent population samples.</p>}
        {execution.parameters && <pre>{JSON.stringify(execution.parameters, null, 2)}</pre>}
        {study.reproduction?.command && <React.Fragment><h4>Re-run the committed task locally</h4><Copyable cmd={study.reproduction.command} /></React.Fragment>}
      </details>
    </section>
  );
}

function RecordedPaperChart({ checkpoints, metric, selectedIndex }) {
  const x = step => 64 + (Math.log10(step + 1) - Math.log10(checkpoints[0].step + 1)) / Math.max(1e-12, Math.log10(checkpoints[checkpoints.length - 1].step + 1) - Math.log10(checkpoints[0].step + 1)) * 722;
  const keys = metric === 'position' ? ['adam_position', 'amsgrad_position'] : ['adam_average_regret', 'amsgrad_average_regret'];
  const values = checkpoints.flatMap(point => keys.map(key => point[key]));
  const low = Math.min(metric === 'position' ? -1 : 0, ...values);
  const high = Math.max(metric === 'position' ? 1 : 0, ...values);
  const span = Math.max(1e-9, high - low);
  const bottom = low - span * 0.06, top = high + span * 0.06;
  const y = value => 274 - (value - bottom) / (top - bottom) * 242;
  const pathFor = key => checkpoints.map((point, i) => `${i ? 'L' : 'M'}${x(point.step).toFixed(2)},${y(point[key]).toFixed(2)}`).join(' ');
  const selected = checkpoints[selectedIndex];
  const ticks = [0, 1, 2, 3, 4].map(index => low + (high - low) * index / 4);
  const xTicks = [...new Set([0, Math.floor((checkpoints.length - 1) / 2), checkpoints.length - 1])];
  return <figure className="paper-figure">
    <div className="paper-legend"><span><i className="paper-swatch adam" />Adam</span><span><i className="paper-swatch amsgrad" />AMSGrad</span><span>{metric === 'position' ? 'Position in the constrained domain' : 'Average regret against the prescribed comparator'}</span></div>
    <svg className="paper-trace-chart" viewBox="0 0 820 330" role="img" aria-labelledby="paper-chart-title paper-chart-description">
      <title id="paper-chart-title">Recorded Adam and AMSGrad {metric === 'position' ? 'positions' : 'average regret'}</title>
      <desc id="paper-chart-description">A finite recorded trajectory. The horizontal axis uses a logarithmic iteration scale. Use the checkpoint slider for exact displayed values; full precision is in the JSON receipt.</desc>
      {ticks.map((value, i) => <g key={i}><line x1="64" x2="786" y1={y(value)} y2={y(value)} className="paper-grid-line" /><text x="52" y={y(value) + 4} textAnchor="end">{paperNumber(value, 3)}</text></g>)}
      <line x1="64" x2="786" y1="274" y2="274" className="paper-axis-line" />
      {xTicks.map(index => <text key={index} x={x(checkpoints[index].step)} y="296" textAnchor="middle">{paperNumber(checkpoints[index].step, 0)}</text>)}
      <text x="425" y="320" textAnchor="middle">Iteration · logarithmic scale</text>
      <path d={pathFor(keys[0])} className="paper-trace adam" />
      <path d={pathFor(keys[1])} className="paper-trace amsgrad" />
      <line x1={x(selected.step)} x2={x(selected.step)} y1="24" y2="274" className="paper-selected-line" />
      {keys.map((key, index) => <circle key={key} cx={x(selected.step)} cy={y(selected[key])} r="5" className={'paper-point ' + (index ? 'amsgrad' : 'adam')} />)}
    </svg>
  </figure>;
}

window.App = App;
window.Nav = Nav;
window.Footer = Footer;
window.Copyable = Copyable;
