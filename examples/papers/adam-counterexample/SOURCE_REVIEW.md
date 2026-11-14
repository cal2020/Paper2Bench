# Source review: Adam counterexample

This experiment numerically checks one deterministic, one-dimensional construction from Sashank J. Reddi, Satyen Kale and Sanjiv Kumar, *On the Convergence of Adam and Beyond*, ICLR 2018. It compares the paper's Adam update with its AMSGrad update on the Appendix A loss sequence. It does not reproduce the paper's neural-network experiments or its Figure 1.

## Source identity

- Version: [arXiv:1904.09237v1](https://arxiv.org/abs/1904.09237v1), posted 19 April 2019; the manuscript identifies itself as the ICLR 2018 conference paper.
- Reviewed PDF: [versioned primary PDF](https://arxiv.org/pdf/1904.09237v1), 23 pages, 602,904 bytes.
- PDF SHA-256: `75a949b0751e81c490e7eee76efadd1eb8868c64f44275d8d1542a8f42cb7c01`.
- Conference record: [OpenReview](https://openreview.net/forum?id=ryQu7f-RZ). A [primary archived copy](https://openreview.net/references/pdf?id=HJRTlSAPf) provides another route to the conference manuscript; the digest above identifies the arXiv PDF specifically.
- Page numbers below are both printed page numbers and one-based PDF page numbers. The relevant pages were inspected as rendered images as well as extracted text.

## Paper-to-experiment mapping

- Page 2, Section 2: the decision `x_t` precedes the round's loss; regret compares cumulative incurred loss with the best single feasible decision in hindsight.
- Page 3, Algorithm 1 and Equation (1): moments start at zero; Adam uses exponential moving averages of the gradient and squared gradient, then a projected update. Footnote 1 explicitly omits the original Adam debiasing step in this presentation.
- Page 4, Section 3 and Theorem 1: the loss is `C*x` when `t mod 3 == 1` and `-x` otherwise, on `[-1, 1]`; the main text takes `C > 2` and `beta2 = 1/(1+C*C)` with `beta1 = 0`.
- Page 5, Algorithm 2: AMSGrad retains the coordinatewise running maximum of the second moment and uses that maximum in the denominator. Its first and second moments, and its running maximum, start at zero.
- Pages 10–11, Appendix A: the construction starts at `x_1 = 1`, uses `alpha_t = alpha/sqrt(t)` with `alpha < sqrt(1-beta2)`, and uses ordinary clipping as its one-dimensional projection. The proof gives positive Adam iterates and `x_(3k+1) = 1`, and a regret lower bound of `2*C-4` per complete three-round cycle.
- Page 6, Section 5 and Figure 1: the published synthetic plot instead uses a 101-round cycle with gradients `1010` and `-10`, with `beta1 = 0.9` and `beta2 = 0.99`. That plot is outside this experiment's scope.

## Prespecified numerical construction

Both optimizers receive exactly the same loss sequence and settings:

- One-based rounds `t = 1, ..., 30000`; no early stopping, random sampling or tuning after observing the candidate output.
- `C = 3`, hence gradients repeat `[3, -1, -1]` and `beta2 = 0.1`.
- `beta1 = 0`, `alpha = 0.1`, `alpha_t = 0.1/sqrt(t)`.
- Feasible domain `[-1, 1]`; `x_1 = 1`; zero initial moments and zero initial AMSGrad maximum.
- No bias correction and no epsilon. All gradients are nonzero, so the denominator is positive after the moment update.
- Binary64 arithmetic; complete traces are compared with the independently computed reference using an absolute tolerance of `1e-10`.
- Checkpoints after `3`, `30`, `300`, `3000` and `30000` rounds. These are all complete cycles.

The selected `C` is strictly greater than two, and `0.1 < sqrt(0.9)`, so this choice satisfies the Appendix A construction. The appendix's weaker printed `C >= 2` includes a zero lower-bound endpoint; this experiment deliberately avoids that endpoint.

At each round, first record and incur loss at `x_t`. Then compute `g_t`, update `m_t` and `v_t`, and update the position. Adam divides by `sqrt(v_t)`. AMSGrad first sets `vhat_t = max(vhat_(t-1), v_t)` and divides by `sqrt(vhat_t)`. Both clip the resulting position to the feasible interval. The candidate's `positions` array contains `x_1` through `x_T`; its separate `final_position` is `x_(T+1)`.

For these horizons the best fixed decision is `-1`. For either optimizer the average regret is therefore `sum(g_t * (x_t + 1))/T`. The paired metric is Adam average regret minus AMSGrad average regret, in loss per round. The scorer identifier is `adam_counterexample/v1`, and the metric identifier is `adam_vs_amsgrad_average_regret_gap`.

The numerical agreement reference comes from a separately implemented controller oracle, rather than a numerical value reported in the paper. The paper supplies the construction and qualitative/theoretical claim. In particular, the Appendix A lower bound implies Adam average regret of at least `2/3` at the complete-cycle checkpoints selected here. Code fidelity, clean execution, agreement with the controller's finite reference, and consistency with that bound are separate checks.

## Interpretation limits

This is a deterministic finite-horizon numerical check of a deliberately constructed special case. `beta1 = 0` is also the RMSProp special case of the paper's Adam formulation. A successful run establishes agreement of this exact implementation and trace with the declared finite experiment. It does not prove an infinite-time statement, reproduce all experiments in the paper, establish performance with typical deep-learning defaults, or rank general-purpose optimization methods or the generating model.

The pre-update loss convention matters. `x_T` and `x_(T+1)` must not be interchanged. The last displayed cycle-loss expression on page 11 repeats a round index; this implementation uses the regret definition on page 2 and one loss for every actual round, rather than copying that apparent typographical repetition.

This review was performed by an automated source-review agent separately from the candidate authoring and evaluator implementation. It is not an independent human expert review or a certification by the paper's authors. A method review of the exact candidate submission must additionally bind its code digest before execution.

## Completed exact-submission method review

Automated static method-review outcome: **PASS** for these exact bytes:

- `solution.py` SHA-256: `4e4a27a7ca3275a3f37c5a39c131753dc9f84c03c50bb7a204a1b25a8d04d2cc`.
- `input.json` SHA-256: `841d46b90348d7ac4178d545e585622b9ef1debdbf206944557e345fda0bf30b`.
- Source PDF SHA-256: `75a949b0751e81c490e7eee76efadd1eb8868c64f44275d8d1542a8f42cb7c01`.

The review checked `run_algorithm` and the fixed input against page 3 Algorithm 1, Equation (1) and footnote 1; page 5 Algorithm 2; and pages 10–11 Appendix A. The candidate correctly uses zero initialization, the prescribed one-based gradient cycle, moment updates before normalization, a running maximum including the current second moment for AMSGrad, the decreasing step schedule, clipping after every update, and no epsilon or bias correction. It records the pre-update positions and the separate final update. No source-fidelity correction was required for this sealed candidate. A changed candidate or input requires a new digest-bound review.

This is an automated curator review of code, not an independent human judgment, a formal proof, or an execution result. The source-review agent did not execute the candidate. Clean execution and measured agreement are established separately by controller-run evidence.

## Audit of the local model's authoring draft

The local model's generated configuration and instruction were reviewed separately from the sealed curated implementation. The reviewed private draft identities are:

- `task_config.yaml` SHA-256: `55f6e5e808ec789fa33cc35c771dd14c4a6467852f3855480d613452df6352fa`.
- `instruction.txt` SHA-256: `f16d0066c2c15aacaaa5265d890d342a55c54c31465e972f97648c4fa434de65`.

The draft preserves the bounded research question but its embedded implementation does not satisfy it. It requires these concrete curator corrections before it could be used:

- Replace its constant step size with `0.1/sqrt(t)` using one-based rounds.
- Project each update to `[-1, 1]`.
- Record `x_t` before updating and evaluate the linear loss against the fixed comparator, rather than recording a post-update squared distance.
- Carry the previous first moment as momentum state. The draft incorrectly uses the prior position; the prescribed zero momentum masks this defect in the selected condition.
- Keep the loss period fixed at three rather than coupling it to the gradient magnitude. The selected `C = 3` masks this distinction in the draft.
- Classify the task as a deterministic synthetic finite optimization check, attach the exact source identity and locators, and use the frozen schedule and evaluator instead of an ad hoc analysis protocol.

The generated instruction repeats the implementation defects. These authoring outputs remain `needs_review` and were not executed as the scored candidate. The separately curated, digest-bound executable above implements the corrected requirements. Its successful evaluation must not be presented as autonomous success of the flawed model draft. Raw draft text and paper content need not be published to report this correction summary.

The subsequent planning output, `instruction_gt.txt` (SHA-256 `89189597e1cba017e42cc6159eee66e706b58bbb5c04665124a91fa944f5983b`), also requires review. It explicitly substitutes a constant step size for the declared decreasing schedule, retains the squared-distance metric, introduces a beta-two parameter sweep outside the frozen condition, omits projection, and makes an incorrect claim of exact arithmetic from NumPy. Its regret and final-iterate descriptions are internally inconsistent. Curators must retain the single frozen condition, binary64 tolerances, best fixed comparator over the whole horizon, complete pre-update position trace and separate final update. The draft manifest correctly retains `needs_review`; the curated executable's task is an optimization adaptation rather than an unspecified method generated from scratch. No generated planning output was used as the scored executable.

## Rights and public artifacts

The [license linked by this arXiv record](https://arxiv.org/licenses/nonexclusive-distrib/1.0/license.html) grants distribution rights to arXiv. It does not provide this project with a Creative Commons redistribution grant. No broader redistribution permission was established in this review.

The public repository and demo should contain primary-source links, this original paraphrase, the independently written implementation, and generated run evidence. Do not include the downloaded PDF, page images, extracted full text or copied paper figures in public artifacts. The local source copy is used only to verify identity and review the method. This file contains no verbatim passage from the paper beyond its bibliographic title and method names.
