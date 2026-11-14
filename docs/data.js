// Shared content for the Paper2Bench site.
// Each entry in `papers` is a fully self-contained paper bundle: metadata,
// problem tree, real instruction.txt content, the evaluator-only addendum
// appended by paper2bench plan, and diff stats computed from the texts.

(function () {

  const trim = (s) => s.replace(/^\n+|\n+$/g, '');

  // Build a per-paper bundle.
  function makePaper(meta, tree, instructionReal, instructionGtAddendum) {
    const ir = trim(instructionReal);
    const ad = trim(instructionGtAddendum);
    const full = ir + '\n\n---\n\n' + ad;
    const sharedLines = ir.split('\n').length;
    const totalLines  = full.split('\n').length;
    const addedLines  = totalLines - sharedLines - 2; // minus blank + `---`
    const addedSections = (ad.match(/^##\s.+/gm) || []).map(s => s.replace(/^##\s+/, ''));
    const addedTasks    = (ad.match(/^\d+\.\s+\*\*/gm) || []).length;
    return {
      ...meta,
      tree,
      instructionReal: ir,
      instructionGtAddendum: ad,
      diff: {
        sharedLines, addedLines,
        sharedBytes: ir.length,
        totalBytes:  full.length,
        addedSections,
        addedTasks,
      },
    };
  }

  // ---------- CGCNN ----------
  const cgcnn = makePaper(
    {
      id: 'cgcnn',
      title: 'CGCNN',
      longTitle: 'Crystal Graph Convolutional Neural Networks for an Accurate and Interpretable Prediction of Material Properties',
      authors: 'Xie & Grossman',
      year: 2018,
      domain: 'materials',
      type: 'novel_architecture',
      rq: 'Can a graph convolutional network predict bulk material properties directly from crystal structure?',
      branches: ['formation energy', 'band gap', 'elastic moduli', 'perovskite site E'],
      counts: { models: 1, datasets: 2, refs: 3, rqs: 3, leaves: 4 },
    },
    {
      root: 'Predict properties from crystal structure',
      rqs: [
        { label: 'RQ1', desc: 'DFT accuracy', leaves: [
          { label: 'Formation energy', metric: 'MAE 0.039' },
          { label: 'Band gap',         metric: 'MAE 0.388' },
        ]},
        { label: 'RQ2', desc: 'Interpretability', leaves: [
          { label: 'Site energies',    metric: 'MAE 0.130' },
        ]},
        { label: 'RQ3', desc: 'Materials discovery', leaves: [
          { label: 'Synthesizable',    metric: '33 new' },
        ]},
      ],
    },
`You are a research agent. Conduct research and experiment about the question: "Can a graph convolutional network predict bulk material properties directly from crystal structure?"

You have access to the following resources:

Models:
- CGCNN via API
- Load with HuggingFace: txie-93/cgcnn
- Computational budget: 10000 API calls per model

Datasets:
- Materials Project database: Contains 46744 inorganic crystals with various structure types and compositions.
  Loading code:
    from pymatgen import MPRester
    mpr = MPRester("YOUR_API_KEY")
    data = mpr.query(criteria={"nelements": {"$lte": 7}}, ...)
- Perovskite database: Includes 18928 perovskite crystals with energy above hull data.
  Loading code:
    import pandas as pd
    perovskite_data = pd.read_csv("perovskite_data.csv")

Experimental constraints:
- Do NOT use web search
- Run FULL end-to-end experiments

Please design and execute experiments to investigate this research question. Document your experimental plan, run end-to-end experiments, and provide conclusions at different levels of detail.`,
`## Experimental Design

1. **Task 1: Predict Formation Energy per Atom**
   - Procedure: Materials Project · 28,046 points · 60/20/20 split. Train CGCNN per Eq. 5.
   - Record: MAE vs DFT.  Target: 0.039 eV/atom.
   - Correspondence: Figure 2(b), Table I.

2. **Task 2: Predict Absolute Energy**
   - Procedure: Same setup as Task 1.
   - Record: MAE.  Correspondence: Table I.

3. **Task 3: Predict Band Gap**
   - Procedure: Materials Project · 16,458 points.
   - Record: MAE.  Target: 0.388 eV.  Correspondence: Table I.

4. **Task 4: Predict Fermi Energy**
   - Procedure: 28,046 training points.
   - Record: MAE.  Correspondence: Table I.

5. **Task 5–7: Bulk / Shear Moduli + Poisson Ratio**
   - Procedure: 2,041 training points each, CGCNN per Eq. 5.
   - Record: MAE per property.  Correspondence: Table I.

8. **Task 8: Extract Site Energies in Perovskites**
   - Procedure: 18,928 crystals · linear pooling on E above hull.
   - Record: MAE 0.130 eV/atom.  Correspondence: Fig. 3(b), Table V.

9. **Task 9: Identify Synthesizable Perovskites**
   - Procedure: Combinational search · E above hull < 0.2 eV/atom.
   - Record: # new perovskites.  Correspondence: Fig. 3(c, d), Table V.

## Evaluation Details

- **Metrics**: MAE = mean absolute difference between predicted and actual values.
- **Reference state**: Formation energy uses Materials Project DFT values.
- **Post-processing**: Average predictions over test set; check k-point convergence; smooth if needed.
- **Aggregation**: MAE per structure → per composition → overall.

## Implementation Notes

- **Pitfalls**: Correct 60/20/20 splits to avoid leakage; tune LR + regularization to prevent overfit.
- **Conventions**: Follow CGCNN docs for atom + bond feature vectors.
- **Order**: Train → validate → select best model → evaluate on held-out test set.`);

  // ---------- DNABERT-2 ----------
  const dnabert2 = makePaper(
    {
      id: 'dnabert2',
      title: 'DNABERT-2',
      longTitle: 'DNABERT-2: Efficient Foundation Model and Benchmark For Multi-Species Genome',
      authors: 'Zhou et al.',
      year: 2023,
      domain: 'biology',
      type: 'novel_architecture',
      rq: 'Does byte-pair tokenization and multi-species pretraining improve DNA foundation model quality at lower compute?',
      branches: ['BPE vocab sweep', 'GUE benchmark eval', 'FLOPs / accuracy'],
      counts: { models: 3, datasets: 2, refs: 9, rqs: 3, leaves: 3 },
    },
    {
      root: 'Efficient genome foundation model',
      rqs: [
        { label: 'RQ1', desc: 'BPE vs k-mer tokenization', leaves: [
          { label: 'BPE vocab sweep', metric: 'F1 · FLOPs' },
        ]},
        { label: 'RQ2', desc: 'GUE benchmark', leaves: [
          { label: '36 datasets · 9 tasks', metric: 'F1 · MCC' },
        ]},
        { label: 'RQ3', desc: 'Compute efficiency', leaves: [
          { label: 'Params + FLOPs', metric: '21× smaller' },
        ]},
      ],
    },
`You are a research agent. Conduct research and experiment about the question: "Does byte-pair tokenization and multi-species pretraining improve DNA foundation model quality at lower compute?"

You have access to the following resources:

Models:
- DNABERT-2 via API
- Load with HuggingFace: InstaDeepAI/nucleotide-transformer-500m-human-ref
- Computational budget: 10000 API calls per model

Datasets:
- Multi-species genome dataset: A dataset comprising genomes from 135 species, totaling approximately 32.49 billion nucleotide bases.
  Loading code:
    import pandas as pd
    dataset = pd.read_csv("path_to_multi_species_genome_dataset.csv")
- Genome Understanding Evaluation (GUE) benchmark: A comprehensive benchmark containing 36 datasets across 9 tasks for evaluating genome foundation models.
  Loading code:
    import pandas as pd
    gue_dataset = pd.read_csv("path_to_gue_benchmark.csv")

Experimental constraints:
- Do NOT use web search
- Run FULL end-to-end experiments

Please design and execute experiments to investigate this research question. Document your experimental plan, run end-to-end experiments, and provide conclusions at different levels of detail.`,
`## Experimental Design

1. **Task 1: BPE Tokenization Analysis**
   - Procedure: Multi-species sequences (filter for A/T/C/G). Train DNABERT-2 with BPE; sweep vocabulary sizes 2^8 / 2^12 / 2^15. 80/10/10 split. Evaluate on GUE.
   - Record: F1, average token length, FLOPs.
   - Correspondence: Figure 3, Table 3, Section 3.1.

2. **Task 2: DNABERT-2 on the GUE Benchmark**
   - Procedure: Use the predefined GUE splits across 36 datasets / 9 tasks.
   - Record: F1, Matthews Correlation Coefficient (MCC), ranking among models.
   - Correspondence: Table 4, Section 5.3.

3. **Task 3: Compute Efficiency vs Prior Models**
   - Procedure: Compare DNABERT-2 to DNABERT and Nucleotide Transformer under identical preprocessing.
   - Record: #parameters, FLOPs, mean scores.
   - Correspondence: Table 3, Section 5.1.

## Evaluation Details

- **F1**: sklearn.metrics.f1_score with average='macro' for multi-class.
- **MCC**: sklearn.metrics.matthews_corrcoef for binary tasks.
- **FLOPs**: count forward-pass operations of each architecture.
- **Aggregation**: per-dataset mean → per-task mean → overall mean.

## Implementation Notes

- **Pitfalls**: deduplicate before splits to avoid leakage; truncate or window sequences past the 512-token limit; keep preprocessing consistent across baselines.
- **Conventions**: FASTA input; structured (CSV) output for downstream evaluation.
- **Order**: deduplicate → split → tokenize → train → evaluate.`);

  // ---------- Chain-of-Thought ----------
  const cot = makePaper(
    {
      id: 'cot_prompting',
      title: 'Chain-of-Thought',
      longTitle: 'Chain-of-Thought Prompting Elicits Reasoning in Large Language Models',
      authors: 'Wei et al.',
      year: 2022,
      domain: 'ml_nlp',
      type: 'llm_evaluation',
      rq: 'Does chain-of-thought prompting improve LLM reasoning on arithmetic, commonsense, and symbolic tasks?',
      branches: ['arithmetic (GSM8K, …)', 'commonsense (CSQA)', 'symbolic reasoning'],
      counts: { models: 7, datasets: 10, refs: 18, rqs: 3, leaves: 3 },
    },
    {
      root: 'CoT prompting elicits reasoning',
      rqs: [
        { label: 'RQ1', desc: 'Arithmetic reasoning', leaves: [
          { label: 'GSM8K, SVAMP, AQuA…', metric: 'solve rate %' },
        ]},
        { label: 'RQ2', desc: 'Commonsense reasoning', leaves: [
          { label: 'CSQA, StrategyQA', metric: 'solve rate %' },
        ]},
        { label: 'RQ3', desc: 'Symbolic reasoning', leaves: [
          { label: 'Last-letter · Coin flip', metric: 'solve rate %' },
        ]},
      ],
    },
`You are a research agent. Conduct research and experiment about the question: "Does chain-of-thought prompting improve LLM reasoning on arithmetic, commonsense, and symbolic tasks?"

You have access to the following resources:

Models:
- text-ada-001 / text-babbage-001 / text-curie-001 / text-davinci-002 via API
- Load with HuggingFace: google/paLM
- Computational budget: 1000 API calls per model

Datasets:
- GSM8K — math word problems (load_dataset("openai/grade-school-math"))
- SVAMP — variant-structure math problems (load_dataset("arkilpatel/SVAMP"))
- ASDiv — diverse math problems
- AQuA  — algebraic word problems (multiple choice)
- MAWPS — math word problem repository
- CSQA — commonsense QA
- StrategyQA — multi-hop reasoning QA
- SayCan — robot instruction dataset

Experimental constraints:
- Do NOT use web search
- Run FULL end-to-end experiments

Please design and execute experiments to investigate this research question. Document your experimental plan, run end-to-end experiments, and provide conclusions at different levels of detail.`,
`## Experimental Design

1. **Arithmetic — GSM8K**
   - Procedure: 1,319 examples. 8-shot CoT prompts ⟨input, chain-of-thought, output⟩.
   - Record: solve rate, raw outputs, generated chains.
   - Correspondence: Figure 2, Table 1, Section 3.2.

2. **Arithmetic — SVAMP**
   - Procedure: 1,000 examples. Same 8-shot CoT prompts.
   - Record: solve rate.  Correspondence: Figure 4, Table 2.

3. **Arithmetic — ASDiv**
   - Procedure: 2,096 examples. 8-shot CoT.
   - Record: solve rate.  Correspondence: Figure 4, Table 2.

4. **Arithmetic — AQuA**
   - Procedure: 254 multiple-choice items. 4-shot CoT.
   - Record: solve rate.  Correspondence: Figure 4, Table 2.

5. **Arithmetic — MAWPS**
   - Procedure: 562 examples. 8-shot CoT covering varied operations.
   - Record: solve rate.  Correspondence: Figure 4, Table 2.

6. **Commonsense — CSQA**
   - Procedure: CoT prompts with randomly-selected training exemplars.
   - Record: solve rate.  Correspondence: Figure 7, Table 4.

7. **Commonsense — StrategyQA**
   - Procedure: CoT prompts formatted for multi-hop reasoning.
   - Record: solve rate.  Correspondence: Figure 7, Table 4.

8. **Symbolic — Last-Letter Concatenation**
   - Procedure: 8-shot CoT walking through concatenation.
   - Record: solve rate.  Correspondence: Figure 8, Table 5.

9. **Symbolic — Coin Flip**
   - Procedure: 8-shot CoT detailing state transitions.
   - Record: solve rate.  Correspondence: Figure 8, Table 5.

## Evaluation Details

- **Metric**: solve rate (%) via sklearn.metrics.accuracy_score.
- **Normalization**: strip whitespace, lowercase, then score.
- **Aggregation**: per-sample → per-condition → overall.

## Implementation Notes

- **Pitfalls**: keep the ⟨input, CoT, output⟩ structure exact; correct answers via wrong reasoning are possible.
- **Templates**: reuse the exact prompt templates from the paper.
- **Order**: same evaluation conditions across all models.`);

  // ---------- Lost in the Middle ----------
  const litm = makePaper(
    {
      id: 'lost_in_the_middle',
      title: 'Lost in the Middle',
      longTitle: 'Lost in the Middle: How Language Models Use Long Contexts',
      authors: 'Liu et al.',
      year: 2023,
      domain: 'ml_nlp',
      type: 'llm_evaluation',
      rq: 'Do LLMs degrade when relevant information is placed in the middle of long input contexts?',
      branches: ['multi-doc QA position', 'key-value retrieval'],
      counts: { models: 13, datasets: 2, refs: 8, rqs: 3, leaves: 4 },
    },
    {
      root: 'How LMs use long contexts',
      rqs: [
        { label: 'RQ1', desc: 'Position effects', leaves: [
          { label: 'Multi-doc QA', metric: 'U-shaped acc.' },
        ]},
        { label: 'RQ2', desc: 'Retrieval ability', leaves: [
          { label: 'Synthetic K/V', metric: 'U-shaped acc.' },
        ]},
        { label: 'RQ3', desc: 'Robustness factors', leaves: [
          { label: 'Architecture',  metric: 'enc-dec wins' },
          { label: 'Query-aware',   metric: 'helps K/V' },
        ]},
      ],
    },
`You are a research agent. Conduct research and experiment about the question: "Do LLMs degrade when relevant information is placed in the middle of long input contexts?"

You have access to the following resources:

Models:
- gpt-3.5-turbo-0613 and claude-1.3 via API
- Load with HuggingFace: meta-llama/MPT-30B-Instruct
- Load with HuggingFace: meta-llama/LongChat-13B-16K
- Computational budget: 1000 API calls per model

Datasets:
- NaturalQuestions-Open: real queries paired with Wikipedia-extracted answers.
  Loading code:
    from datasets import load_dataset
    dataset = load_dataset("natural_questions", split="train")
- A synthetic dataset consisting of key-value pairs formatted as JSON, where keys and values are unique randomly-generated UUIDs.

Experimental constraints:
- Do NOT use web search
- Run FULL end-to-end experiments

Please design and execute experiments to investigate this research question. Document your experimental plan, run end-to-end experiments, and provide conclusions at different levels of detail.`,
`## Experimental Design

1. **Multi-Document QA (MDQA)**
   - Procedure: question + k documents (k ∈ {10, 20, 30}), 100 tokens each. The relevant document is placed at beginning / middle / end. Distractors retrieved via MS-MARCO-tuned retriever.
   - Record: accuracy (correct answer in output).
   - Correspondence: Figure 5, Table 1.

2. **Synthetic Key-Value Retrieval**
   - Procedure: JSON of k key-value UUID pairs (k ∈ {75, 140, 300}). Vary key position.
   - Record: accuracy.
   - Correspondence: Figure 7.

3. **Architecture Comparison**
   - Procedure: decoder-only (GPT-3.5, MPT-30B) vs encoder-decoder (Flan-T5-XXL, Flan-UL2) at matched context length.
   - Record: accuracy across positions.
   - Correspondence: Figure 8.

4. **Query-Aware Contextualization**
   - Procedure: place query both before and after documents; compare to query-only-after.
   - Record: accuracy on K/V retrieval.
   - Correspondence: Figure 9.

## Evaluation Details

- **Metric**: accuracy = correct / total × 100.
- **Normalization**: strip extraneous text from model outputs.
- **Aggregation**: per-sample → per-condition → overall.

## Implementation Notes

- **Pitfalls**: ensure distractors are genuinely irrelevant; respect each model's context limit.
- **Prompt template**: "Document [i]: …  Question: {q}  Answer:".
- **Order**: build input → infer → record outputs immediately.`);

  // ---------- MACE ----------
  const mace = makePaper(
    {
      id: 'mace_potential',
      title: 'MACE',
      longTitle: 'MACE: Higher Order Equivariant Message Passing Neural Networks for Fast and Accurate Force Fields',
      authors: 'Batatia et al.',
      year: 2022,
      domain: 'materials',
      type: 'novel_architecture',
      rq: 'Does higher-body-order equivariant message passing improve accuracy and data efficiency for ML interatomic potentials?',
      branches: ['3BPA benchmark', 'rMD17 learning curves', 'extrapolation regimes'],
      counts: { models: 3, datasets: 2, refs: 11, rqs: 3, leaves: 3 },
    },
    {
      root: 'Higher-order equivariant force fields',
      rqs: [
        { label: 'RQ1', desc: 'Efficiency', leaves: [
          { label: '3BPA benchmark', metric: '2 layers' },
        ]},
        { label: 'RQ2', desc: 'Data efficiency', leaves: [
          { label: 'rMD17 learning curve', metric: 'steeper ν' },
        ]},
        { label: 'RQ3', desc: 'Accuracy vs SOTA', leaves: [
          { label: 'vs BOTNet / NequIP', metric: 'SOTA + fast' },
        ]},
      ],
    },
`You are a research agent. Conduct research and experiment about the question: "Does higher-body-order equivariant message passing improve accuracy and data efficiency for ML interatomic potentials?"

You have access to the following resources:

Models:
- MACE via API
- Load with HuggingFace: mace-mp/MACE-MP-0
- Computational budget: 10000 API calls per model

Datasets:
- rMD17 — molecular dynamics trajectories of small organic molecules.
- 3BPA  — DFT train/test splits of a flexible drug-like organic molecule.

Experimental constraints:
- Do NOT use web search
- Run FULL end-to-end experiments

Please design and execute experiments to investigate this research question. Document your experimental plan, run end-to-end experiments, and provide conclusions at different levels of detail.`,
`## Experimental Design

1. **Task 1: MACE on 3BPA**
   - Procedure: train L ∈ {0, 1, 2} MACE models on 3BPA (450 train / 50 val), radial cutoff 5 Å.
   - Record: MAE on energies + forces, training time.
   - Correspondence: Figure 1, Table 2, Section 5.3.

2. **Task 2: Learning Curves on rMD17**
   - Procedure: train MACE with ν ∈ {1, 2, 3} on rMD17 (950 / 50), cutoff 5 Å.
   - Record: learning-curve steepness, RMSE on train.
   - Correspondence: Figure 2, Section 5.1.

3. **Task 3: Extrapolation on 3BPA**
   - Procedure: evaluate the Task-1 models at 300 / 600 / 1200 K and on dihedral slices.
   - Record: RMSE on energies + forces.
   - Correspondence: Table 2, Section 5.3.

4. **Task 4: MACE on AcAc**
   - Procedure: MACE models on AcAc (450 / 50), cutoff 5 Å.
   - Record: RMSE on energies + forces at 300 K and 600 K.
   - Correspondence: Table 3, Section 5.3.

## Evaluation Details

- **MAE**: mean absolute difference between predicted and true energies / forces.
- **Learning-curve steepness**: power-law fit on log-log of RMSE vs dataset size.
- **Aggregation**: per-structure → per-composition → overall.

## Implementation Notes

- **Pitfalls**: keep radial cutoff fixed at 5 Å across models; watch overfitting on small training sets; extrapolation can fail far from train distribution.
- **Conventions**: follow MACE input-file format (Appendix A.5); implement restart-from-checkpoint.
- **Order**: train → validate per-epoch → final test.`);

  // ---------- Bundle ----------
  const papers = [cgcnn, dnabert2, cot, litm, mace];

  window.P2B_DATA = {
    papers,
    install: {
      pip:  'pip install git+https://github.com/cal2020/Paper2Bench.git',
      pipa: 'pip install -e ".[dev]"',
    },
    cli: {
      run: 'paper2bench run --pdf paper.pdf --research-question "Does the stated method improve the primary metric?" --runtime-config examples/runtime.local.yaml',
      download:  'paper2bench download "Paper title" -o ./papers/',
      parse:     'paper2bench parse paper.pdf -o tree.json',
      classify:  'paper2bench classify --pdf paper.pdf',
      extract:   'paper2bench extract --pdf paper.pdf -o config.yaml',
      render:    'paper2bench render --config task.yaml -o instruction.txt',
      plan:      'paper2bench plan --instruction instruction.txt --pdf paper.pdf -o instruction_gt.txt',
      verify:    'paper2bench verify --instruction-gt gt.txt --pdf paper.pdf',
    },
    // Default "current paper" for the landing page = CGCNN. The detail page
    // overrides this from the ?paper=<id> URL parameter before mount.
    current: cgcnn,
  };

  // Lookup helper used by example.html
  window.P2B_findPaper = (id) =>
    papers.find((p) => p.id === id) || cgcnn;

})();
