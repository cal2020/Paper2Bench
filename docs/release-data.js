// Public, deliberately selected facts from local verification receipts.
// This is a recorded software fixture summary, not a live execution service.
window.P2B_RELEASE = Object.freeze({
  version: '0.4.0a2',
  repository: 'https://github.com/cal2020/Paper2Bench',
  upstream: 'https://github.com/AbhayAnandUCSD/Paper2Bench',
  evidenceDate: '12 November 2026',
  tests: 572,
  testDate: '13 November 2026',
  scenarios: [
    {
      id: 'linear', title: 'Linear fitting', kind: 'Browser computation',
      paradigm: 'Numerical method · no foundation model',
      description: 'Fit y = ax + b to four training pairs, then compute three held-out predictions. The browser check mirrors this tiny example; the recorded Python submission was separately executed in Docker.',
      stages: ['Fit four training pairs', 'Predict held-out values', 'Recompute mean squared error'],
      facts: [['Training pairs', '(0, 1), (1, 3), (2, 5), (3, 7)'], ['Held-out inputs', '4, 5, 10'], ['Recorded Docker result', 'MSE 0 · cleanup confirmed']],
      command: 'paper2bench execute --task examples/tasks/linear-fit/task_manifest.json --evaluation-bundle examples/tasks/linear-fit/curator_manifest.json --evaluation-profile method_output/v1',
      source: 'examples/tasks/linear-fit/solution.py',
      button: 'Run browser check',
      scope: 'Synthetic data. This browser calculation does not launch Docker or call a model.'
    },
    {
      id: 'knapsack', title: 'Symbolic optimization', kind: 'Browser computation',
      paradigm: 'Discrete optimization · no foundation model',
      description: 'Choose a feasible subset under capacity 8. This browser check enumerates the 16 possible subsets; the recorded Python candidate used dynamic programming and a separate exhaustive oracle.',
      stages: ['Enumerate 16 subsets', 'Reject capacity violations', 'Compare feasible objectives'],
      facts: [['Weights / values', '[2, 3, 4, 5] / [3, 4, 5, 8]'], ['Capacity', '8'], ['Recorded Docker result', 'Optimal objective 12 · feasible']],
      command: 'paper2bench execute --task examples/tasks/knapsack/task_manifest.json --evaluation-bundle examples/tasks/knapsack/curator_manifest.json --evaluation-profile method_output/v1',
      source: 'examples/tasks/knapsack/solution.py',
      button: 'Run browser check',
      scope: 'Synthetic optimization fixture. Browser results are separate from the recorded isolated Python run.'
    },
    {
      id: 'science', title: 'Reviewed protocol', kind: 'Recorded Docker evidence',
      paradigm: 'Fixed scientific contract · three isolated seeds',
      description: 'Inspect the saved synthetic protocol run: source, code, data and criteria were frozen before three fresh Docker workers ran. A fixed schedule preserves every observation.',
      stages: ['Freeze exact task and review', 'Run seeds 0, 1 and 2', 'Apply protected criteria'],
      facts: [['Prescribed seeds', '0, 1, 2'], ['Trusted MSE observations', '0, 0, 0'], ['Critical criteria', 'Execution, curator fidelity, agreement: pass']],
      command: 'python examples/protocols/build_contract_fixture.py output/science-fixture && paper2bench execute --task output/science-fixture/task_manifest.json --evaluation-bundle output/science-fixture/curator_manifest.json --evaluation-profile reproduction/v1',
      source: 'examples/protocols/README.md',
      button: 'Inspect recorded run',
      scope: 'Synthetic contract and author review only. No published paper was reproduced; population uncertainty was not assessed.'
    },
    {
      id: 'broker', title: 'Local model broker', kind: 'Recorded local inference',
      paradigm: 'Foundation model · isolated file IPC',
      description: 'Inspect an actual Docker → controller broker → local Llama → fresh Docker round trip. Public target IDs cross the worker boundary; provider credentials remain in the controller.',
      stages: ['Worker writes bounded request', 'Controller calls local Llama', 'Fresh worker reads response'],
      facts: [['Target model', 'Ollama · llama3.2:latest'], ['Provider calls / replay calls', '1 / 0'], ['Worker checks', 'Network none · credentials absent · cleanup confirmed']],
      command: 'paper2bench doctor --runtime-config examples/runtime.certify-local.yaml --probe --role author --max-probe-tokens 32',
      source: 'examples/runtime.broker-local.yaml',
      button: 'Inspect recorded run',
      scope: 'A bounded arithmetic fixture worked locally. This site makes no model calls; server locality is operator-trusted, not independently attested offline.'
    }
  ]
});

if (window.P2B_DATA) window.P2B_DATA.install.pip = 'pip install git+' + window.P2B_RELEASE.repository + '.git';

window.P2B_runBrowserFixture = function (id) {
  if (id === 'linear') {
    const pairs = [[0, 1], [1, 3], [2, 5], [3, 7]];
    const n = pairs.length;
    const sx = pairs.reduce((v, p) => v + p[0], 0);
    const sy = pairs.reduce((v, p) => v + p[1], 0);
    const sxx = pairs.reduce((v, p) => v + p[0] * p[0], 0);
    const sxy = pairs.reduce((v, p) => v + p[0] * p[1], 0);
    const a = (n * sxy - sx * sy) / (n * sxx - sx * sx);
    const b = (sy - a * sx) / n;
    const predicted = [4, 5, 10].map(x => a * x + b);
    const labels = [9, 11, 21];
    const mse = predicted.reduce((sum, y, i) => sum + (y - labels[i]) ** 2, 0) / labels.length;
    return { status: mse === 0 ? 'pass' : 'fail', result: 'MSE ' + mse, detail: 'Slope ' + a + ', intercept ' + b + '. Predictions: ' + predicted.join(', ') + '.', origin: 'Computed in this browser' };
  }
  if (id === 'knapsack') {
    const weights = [2, 3, 4, 5], values = [3, 4, 5, 8];
    let best = -Infinity, selected = [], selectedWeight = 0;
    for (let mask = 0; mask < 16; mask++) {
      const items = weights.map((_, i) => i).filter(i => mask & (1 << i));
      const weight = items.reduce((s, i) => s + weights[i], 0);
      const value = items.reduce((s, i) => s + values[i], 0);
      if (weight <= 8 && value > best) { best = value; selected = items; selectedWeight = weight; }
    }
    return { status: best === 12 ? 'pass' : 'fail', result: 'Objective ' + best, detail: 'Selected indices: ' + selected.join(', ') + '. Weight ' + selectedWeight + ' / 8. All 16 subsets checked.', origin: 'Computed in this browser' };
  }
  if (id === 'science') return { status: 'pass', result: '3 seeds · 3 passing criteria', detail: 'MSE [0, 0, 0]. Exact-submission curator review and fixed protocol passed for the synthetic fixture. Statistical uncertainty: not assessed.', origin: 'Recorded local Docker run · ' + window.P2B_RELEASE.evidenceDate };
  if (id === 'broker') return { status: 'pass', result: '1 model call · 0 on replay', detail: 'Fresh isolated worker confirmed the local Llama answer. Both worker containers were cleaned up; credential absence was asserted.', origin: 'Recorded local Llama + Docker fixture · ' + window.P2B_RELEASE.evidenceDate };
  throw new Error('Unknown public fixture');
};
