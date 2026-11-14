// Example detail page shell — re-uses Nav, Walkthrough, FullDiff with a
// per-paper header. Reads ?paper=<id> from the URL and sets P2B_DATA.current.

function ExampleHeader({ paper }) {
  return (
    <section className="example-header container">
      <div className="example-crumb">
        <a href="index.html">← back to examples</a>
      </div>
      <div className="example-header-grid">
        <div>
          <h1 className="example-h1 reveal">
            {paper.title}
          </h1>
          <div className="example-meta reveal" data-delay="1">
            <span>{paper.authors}</span>
            <span className="dot-sep">·</span>
            <span>{paper.year}</span>
          </div>
        </div>
        <div className="example-rq-card reveal" data-delay="1">
          <span className="label accent">Research question</span>
          <p>"{paper.rq}"</p>
        </div>
      </div>
    </section>
  );
}

function ExampleApp() {
  const paper = window.P2B_DATA.current;
  return (
    <React.Fragment>
      <window.Nav />
      <ExampleHeader paper={paper} />
      <div className="container"><p className="scope-note draft-banner"><strong>Illustrative paper draft.</strong> This upstream-derived content has not been validated as a reproduction of the published paper. Model names, loaders, assumptions and reference plans require source and curator review.</p></div>

      <section className="section walkthrough-section container" id="walkthrough">
        <window.Walkthrough paper={paper} />
      </section>

      <section className="section container" id="diff">
        <div className="section-head reveal">
          <span className="label accent">§ The agent brief vs the evaluator's key</span>
          <span className="rule" />
          <span className="label">{paper.title} · illustrative draft</span>
        </div>
        <h2 className="section-title reveal">
          What the <span className="italic-accent">agent</span> sees vs what the <span className="italic-accent">evaluator</span> sees.
        </h2>
        <p className="section-lede reveal" data-delay="1">
          <code style={{ background: 'var(--surface-2)', padding: '1px 6px', fontSize: 13 }}>instruction_gt.txt</code> = <code style={{ background: 'var(--surface-2)', padding: '1px 6px', fontSize: 13 }}>instruction.txt</code> with a generated reference plan appended for review. The candidate does not receive the evaluator addition; <code style={{ background: 'var(--surface-2)', padding: '1px 6px', fontSize: 13 }}>paper2bench verify</code> scores conclusion similarity. That score does not establish execution or method fidelity.
        </p>
        <div className="reveal" data-delay="2">
          <window.FullDiff paper={paper} />
        </div>
      </section>

      <section className="section container">
        <div className="example-actions reveal">
          <a href="index.html" className="example-action-btn">
            ← Back to the collection
          </a>
          <div className="example-other">
            <span className="label">Other papers</span>
            <div className="example-other-list">
              {window.P2B_DATA.papers
                .filter(p => p.id !== paper.id)
                .map(p => (
                  <a key={p.id} href={`example.html?paper=${p.id}`} className="example-other-link">
                    <span className="example-other-title">{p.title}</span>
                    <span className="example-other-meta">{p.domain}</span>
                  </a>
                ))}
            </div>
          </div>
        </div>
      </section>
    </React.Fragment>
  );
}

window.ExampleApp = ExampleApp;
