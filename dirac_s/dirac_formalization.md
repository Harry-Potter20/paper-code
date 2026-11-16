# DIRAC-S Mathematical Formalization

Operationalizes `DIRAC-S_Experimental_Specification_v1.0.md` Section 34, steps 1-2 ("freeze the mathematical
specification," "freeze the exact architecture") for 3D CT. Every symbol below is defined immediately before or
after its equation. No numeric thresholds appear here — those are `[CALIBRATE]`/`[FIXED]` tags in
`IMPLEMENTATION_SPEC.md`, which this document is upstream of.

**The contribution is the structure, not the parameter count.** This is DIRAC's version of the discipline the
sibling `covtoken` project runs under ("the contribution is the constraint, not the backbone"). Covtoken isolates
its claim by holding the backbone fixed and varying only a constraint; DIRAC's claim is architectural, so the
analogous discipline is the reverse shape but the same spirit: every advantage attributed to the structured
representation ($\rho$, $G$, $T$ jointly — §3-§6) must survive exact parameter matching against CNN/ViT baselines
(§9). If DIRAC only wins by having more effective capacity than a matched conventional model, H1 is not
supported — that would be a parameter-count result wearing a structure-shaped explanation, not evidence for the
central hypothesis (spec §32).

---

## 1. Setup and notation

A 3D CT volume is $X \in \mathbb{R}^{D \times H \times W}$ in Hounsfield units after preprocessing (orientation
and voxel-spacing normalized, HU-clipped to a documented window, but not yet quantized to 8-bit — spec §10
items 3-9; see `IMPLEMENTATION_SPEC.md` §7).

Let $N = 512$ be the number of spatial tokens the encoder produces (§2), $d$ the per-token embedding width
($[CALIBRATE]$, typically 128-256), and $K = 8$ the number of mixture components used to construct the
probabilistic state (§3, $[CALIBRATE]$). Complex quantities are written $\psi = \psi_{\mathrm{re}} + i\psi_{\mathrm{im}}$
with $\psi_{\mathrm{re}}, \psi_{\mathrm{im}} \in \mathbb{R}^d$; $\dagger$ denotes conjugate transpose.

The pipeline is $X \to \psi \to \rho \to (G, T) \to z$, matching the spec's §5/§9 diagram, with one resolved
ambiguity: $T$ is computed from $X$ directly (§5), not from $\rho$ — see §5 below for why.

---

## 2. The complex latent field $\psi(X)$

**ψ is a spatially-indexed field, not a global vector.** $G$ and $T$ are meaningless over a single point, so the
encoder must produce a *set* of tokens with retained spatial identity, not a pooled embedding.

**Construction.** A 3D convolutional stem — four strided `Conv3d` blocks reducing, e.g., a $128^3$ preprocessed
volume to an $8\times8\times8 = 512$-token grid — followed by a shallow Transformer encoder with full
self-attention over the 512 tokens (at this token count, full attention is cheap; no windowed/Swin-style
attention is needed at DIRAC-S scale). This is a standard conv-stem-plus-ViT hybrid; the novelty is entirely in
what happens after this stage, not in $\psi$'s backbone.

Let $H \in \mathbb{R}^{N \times d}$ be the real-valued output of the stem+transformer. Two parallel real linear
heads complexify it:

$$\psi_{\mathrm{re}} = W_{\mathrm{re}} H, \qquad \psi_{\mathrm{im}} = W_{\mathrm{im}} H, \qquad \psi = \psi_{\mathrm{re}} + i\psi_{\mathrm{im}} \in \mathbb{C}^{N \times d}$$

This is the "computationally justified" answer the spec (§12) demands of the complex representation: it is a
learned pair of real projections, nothing exotic, and it is directly checkable (§8 below).

**Engineering note (not part of the math, but load-bearing for implementation):** represent $\psi$ in code as a
`(re, im)` pair of real tensors throughout, and only form a genuine `torch.complex` tensor at the few sites that
need true complex arithmetic (the inner products in §3). No prior code in this vault exercises PyTorch's complex
autograd; minimizing surface area here is a deliberate risk-reduction choice, not a shortcut.

Each token doubles as a node of the relational graph $G$ (§4) — no separate graph-construction stage is needed
for $\psi$ to support both $\rho$ and $G$.

---

## 3. The probabilistic state $\rho(X)$

The spec requires $\rho \succeq 0$ and $\mathrm{Tr}(\rho) = 1$ **by construction**, not as a soft penalty (§12).
This section states a construction that satisfies this trivially, explains why the obvious first guess fails,
and names what's deferred.

### Construction A (rejected): rank-1 pure state

$$\rho_i = \frac{\psi_i \psi_i^\dagger}{\lVert \psi_i \rVert^2}$$

This is PSD and trace-1 by construction, and it is the cheapest possible option. **It must be rejected
explicitly, not silently avoided:** the von Neumann entropy of any pure state is identically zero,
$S(\rho_i) = -\mathrm{Tr}(\rho_i \log \rho_i) \equiv 0$ for every $\psi_i$. Since §14 and H4 require uncertainty
extracted from $\rho$ via exactly this entropy, a pure-state $\rho$ is structurally incapable of ever producing
a usable uncertainty signal — this is catchable at the math-spec stage, before any code is written, which is the
entire point of freezing the math first. Construction A is retained only as the `D_-prob` ablation floor (spec
§19): it isolates, as a testable claim, what a genuine mixture state buys over a pure state.

### Construction B (chosen for v1): mixture-of-$K$ rank-1 states

$$\rho_i = \sum_{k=1}^{K} \pi_{i,k} \cdot \frac{\varphi_{i,k}\varphi_{i,k}^\dagger}{\lVert \varphi_{i,k} \rVert^2}, \qquad \pi_{i,k} = \mathrm{softmax}_k(g_\theta(\psi_i)), \qquad \varphi_{i,k} = W_k \psi_i \;\; (k=1..K)$$

$g_\theta$ is a small learned head producing $K$ mixture logits per token; $W_k$ are $K$ learned complex
projections ("basis states"). PSD by construction (a nonnegative combination of PSD rank-1 terms is PSD).
Trace-1 by construction (each term is already trace-normalized; the weights sum to 1 via softmax). No penalty
term anywhere.

$S(\rho_i)$ now ranges meaningfully from $0$ (one $\pi_{i,k} \to 1$, the model "commits" to one basis state) to
$\log K$ (uniform mixture, maximal uncertainty) — a real, learnable, and directly interpretable signal: how many
distinct anatomical/morphological hypotheses the model entertains for this patch.

**Efficiency note:** because $\rho_i$ lives in the span of only $K$ vectors, its full spectrum equals the
spectrum of a $K \times K$ Gram-weighted matrix, so entropy is a $K \times K$ eigendecomposition — cheap,
independent of the ambient dimension $d$.

### Construction C (deferred to DIRAC-M, not v1)

A full-rank Cholesky parameterization, $\rho = LL^\dagger / \mathrm{Tr}(LL^\dagger)$ with $L$ a learned lower-
triangular complex $d \times d$ matrix. Maximally expressive — reaches any PSD trace-1 matrix — but $O(d^2)$
parameters per token and $O(d^3)$ eigendecomposition cost per token per forward pass, too expensive for
DIRAC-S's research-scale compute budget at $N=512$ tokens. Named here as an explicit DIRAC-M scale-up ablation
candidate.

---

## 4. The relational graph $G$

### Construction A (chosen for v1): patch-adjacency graph

Nodes = the same $N=512$ tokens $\psi$ already produces — zero extra pipeline stage, zero auxiliary
segmentation needed, keeps pretraining fully self-supervised as spec'd. Edges = fixed local 6- or
26-connectivity on the token lattice, plus learned top-$k$ ($k=8$, $[CALIBRATE]$) feature-similarity edges in
$\psi$-space, capturing non-local anatomical correspondence (e.g. left/right kidney symmetry) that lattice
adjacency alone cannot. This is what earns the word "relational" for H3.

**Message-passing layer:** a from-scratch two-layer Kipf-&-Welling graph convolution,

$$H' = \sigma\left(D^{-1/2} A D^{-1/2} H W\right)$$

with $A$ the (lattice + top-$k$) adjacency, $D$ its degree matrix. **GCN, not GAT, and not `torch_geometric`:**
a second attention mechanism stacked on top of the transformer backbone's own attention would muddy the H3
ablation story (is a gain from the graph specifically, or just "more attention somewhere") and complicate exact
parameter matching (§9). `torch_geometric`/`dgl` require compiled CUDA extensions never exercised on this
vault's HF Jobs images — an avoidable operational risk. At $N=512$ nodes a dense $N \times N$ adjacency
(~262K floats) is trivial; no sparse message-passing framework is needed.

### Construction B (named, deferred past v1): anatomical skeleton graph

Threshold the preprocessed HU volume into a coarse foreground mask (unsupervised, deterministic, and required
preprocessing anyway per spec §10 item 6), 3D-skeletonize via `skimage.morphology.skeletonize_3d`, then bridge
disconnected components with the vault's proven `graph_connect(skel, radius)` pattern (nearest-point bridging,
generalized from `skimage.draw.line` to `skimage.draw.line_nd` for 3D — same dependency, no new library).

This construction is named explicitly, not omitted, because it has a *proven real-data result* behind it:
`topology/FINDINGS.md` found a graph-native representation generalizes far better under domain shift than a
pixel-based one (DRIVE→CHASE zero-shot: pixel-U-Net Dice collapses $0.76 \to 0.38$, graph-native holds
$0.71 \to 0.45$) — directly relevant to H3/H5. Per the spec's own Non-negotiable Principle ("do not add
components merely because they sound theoretically attractive"), it is kept **out of the v1 frozen
architecture** anyway: it requires an auxiliary foreground-segmentation step that couples $G$ to preprocessing
choices in a way that complicates Gate 0's synthetic-input validation, and it is not needed for $z$ to be
well-defined. It becomes a documented `D_graph-patch-plus-anatomical` ablation once the core v1 pipeline runs.

---

## 5. Topological structure $T$

**Computed on the raw preprocessed HU volume, downsampled to roughly $32^3$–$64^3$ — not on $\rho$.** The
spec's own pipeline diagram (§9) shows $\rho$ branching into $G$ and $T$, which would suggest computing $T$ from
$\rho$; this formalization deliberately deviates and states why: a well-defined persistent-homology computation
needs a stable scalar field, and computing one from an *untrained* $\rho$ early in training is a chicken-and-egg
problem — topology of noise. The raw volume is well-defined from step 0, available before any training, and
ground-truth-adjacent.

**Backend:** `gudhi.CubicalComplex`, the exact library already used in `topology/jobs/topology_gate0_job.py`,
which generalizes to 3D directly via `dimensions=[D_z, H_y, W_x]` on a flattened scalar array. Summarized into a
fixed-length persistence vector via the same `ph_feature()` recipe already proven there — sorted $H_0$/$H_1$
persistence lifetimes reduced to count/sum/max amplitude statistics — extended with an $H_2$ dimension for the
3D case.

$$T(X) = \mathrm{PHsummary}(X) \in \mathbb{R}^{m}$$

for a fixed $m$ (number of summary statistics $\times$ homology dimensions). $T(X)$ is embedded by a small MLP
and fused into $z$; separately, $\psi/\rho$ are trained to be *predictive of* $T(X)$ via an auxiliary loss (§7,
$L_{\text{topology}}$) — this is what makes topology actually shape the learned representation rather than
being a static, disconnected post-hoc feature.

---

## 6. The structured representation $z$

$$z = f(\rho, G, T)$$

Concretely: $\rho$'s per-token entropy vector and mixture weights, $G$'s pooled node embeddings after the GCN
layer (§4), and $T$'s embedded PH-summary vector (§5) are concatenated and passed through a small fusion MLP
$f$. $z$ is the frozen representation evaluated in Stage II probing (spec §13) and Stage IV downstream transfer
(spec §15).

---

## 7. Pretraining objectives

Four terms, matching spec §11 Objectives A-D exactly.

**$L_{\text{recon}}$ (Objective A).** A lightweight 3D decoder (transposed-conv upsampling from the
$8\times8\times8$ token grid back to volume resolution) reconstructs $X$ from $z$; $L_1$ voxelwise loss (robust
to HU outliers relative to $L_2$), optionally plus a spatial-gradient-difference term for edge sharpness.

**$L_{\text{masked}}$ (Objective B).** Standard 3D MAE: mask 60-75% ($[CALIBRATE]$) of the 512 tokens before the
encoder, decode, compute $L_1$ only over masked-token voxels. Shares encoder/decoder weights with
$L_{\text{recon}}$, run as a separate forward pass.

**$L_{\text{struct}}$ (Objective C).** For a structure-preserving transform $g$ (small rotation, translation,
intensity jitter, or mild elastic deformation within clinically plausible bounds), encode both $X$ and $g(X)$
and minimize, per spatially matched token pair (matched via $g$'s known geometric mapping):

$$L_{\text{struct}} = \left\lVert \rho_X - \rho_{g(X)} \right\rVert_F^2$$

Frobenius/Hilbert-Schmidt distance is chosen explicitly over quantum fidelity
$\left(\mathrm{Tr}\sqrt{\sqrt{\rho}\,\sigma\,\sqrt{\rho}}\right)^2$: fidelity needs a matrix square root per token
per training step, expensive and unnecessary at v1. Noted as a later, more "physically principled" ablation
variant, not the default.

**$L_{\text{topology}}$ (Objective D).** Not backpropagated through `gudhi` — no differentiable-PH layer exists
or is proposed here. Instead, reusing the vault's own precedent from `topology/jobs/pi_differentiable_job.py`
(which trains toward a stop-gradient morphological target rather than differentiating through the topological
operator itself):

$$L_{\text{topology}} = \left\lVert \mathrm{PHsummary}(X) - \mathrm{PH\_head}(\psi, \rho) \right\rVert^2$$

an auxiliary regression that trains the latent to encode topology, computed against a fixed (stop-gradient)
target. A separate, gradient-free stability check, $\left\lVert \mathrm{PHsummary}(g(X)) - \mathrm{PHsummary}(X) \right\rVert$
for topology-preserving $g$, verifies the literal language of spec §11 Objective D ("remain stable under
topology-preserving transformations").

**Combined loss:**

$$L_{\text{DIRAC}} = \lambda_r L_{\text{recon}} + \lambda_m L_{\text{masked}} + \lambda_s L_{\text{struct}} + \lambda_t L_{\text{topology}}$$

Coefficients $[CALIBRATE]$, locked before final test evaluation per spec §12.

---

## 8. Uncertainty from $\rho$ — designed against a known failure mode

Spec H4 requires uncertainty extracted from $\rho$, via von Neumann entropy $S(\rho_i)$ (§3). This vault has a
directly relevant, documented failure: `sheaf/model.py`'s disagreement signal $D$ was validated on synthetic
data (`gate_reports/gate_0.json`-equivalent conflict-detection AUROC 0.886) but on real data
(`gate_reports/gate_2.json`) it **inversely** predicted error — AUROC 0.138, vs. a plain-entropy baseline's
0.829 — because it conflated disagreement with signal *magnitude*, not direction.

$S(\rho_i)$ is a genuinely different quantity in construction (§3's mixture entropy vs. sheaf's projection-
residual energy), but the failure mode it must be checked against is the same shape, and the check must run
before $S(\rho)$ is trusted for anything:

1. $S(\rho_i)$ vs. downstream prediction error (reconstruction error, or task-head error at probing time):
   AUROC / Spearman correlation.
2. Benchmark against a plain predictive-entropy baseline computed from a task head on the same frozen $z$ — the
   exact comparator that beat the sheaf signal.
3. **Mandatory magnitude-conflation check:** the partial correlation of $S(\rho_i)$ with error, controlling for
   $\lVert \psi_i \rVert$. If $S(\rho)$ is tracking token-norm magnitude rather than genuine uncertainty — the
   exact documented sheaf failure — this partial correlation collapses toward zero once magnitude is partialled
   out.
4. Repeat (1)-(3) with mixture logits computed from $L_2$-normalized (direction-only) $\psi_i$ as a second
   variant. If entropy still discriminates confident from uncertain regions after norm is removed, that is
   positive evidence it is not a magnitude proxy.

This is specified here, in the math document, rather than left implicit in the implementation spec, because it
is a property the *construction itself* must not preclude — Construction A's zero-entropy failure (§3) was
exactly this class of error, caught before code; this check is the empirical complement, catching the same
class of error after code exists but before the result is trusted.

---

## 9. Parameter accounting

Two conventions that must be fixed here, not left to implementation discretion, because getting either wrong
silently invalidates spec §17/§18's fairness requirement:

**Complex-parameter counting.** A complex linear layer $\mathrm{Linear}(d_{\text{in}}, d_{\text{out}})$
(whether implemented as `dtype=torch.cfloat` or as a `(re, im)` pair) has $2 \cdot d_{\text{in}} \cdot d_{\text{out}}$
real scalar parameters, not $d_{\text{in}} \cdot d_{\text{out}}$. Undercounting here would silently make the
"matched" CNN/ViT baselines smaller than DIRAC in real terms while appearing matched on paper.

**This section is where "the contribution is the structure, not the parameter count" (see opening framing) is
operationally enforced** — the two conventions below exist specifically so a structure-vs-capacity confound
cannot hide inside a sloppy parameter count.

**$D_{\text{classical}}$ is not a fairness baseline.** Spec §19's $D_{\text{classical}}$ ablation (DIRAC's own
conv/attention stem with the complex/probabilistic/graph/topology branches stripped, trained with DIRAC's own
pretraining recipe) answers "what do the DIRAC-specific components add over the shared stem," holding the
recipe fixed. Spec §17's parameter-matched CNN/ViT baselines answer "how does DIRAC compare to conventional
architectures," as independently reasonable architectures with their own standard training recipes, matched
only on parameter count and compute budget. These are different scientific claims and must not share
infrastructure beyond the data loader, parameter counter, and eval harness — collapsing them would quietly turn
an ablation into the fairness comparison.

Baselines are matched by tuning channel width primarily (depth changes qualitatively alter the model family) to
a locked tolerance ($\pm5\%$, $[CALIBRATE]$); exact final parameter counts for all three architectures are
reported side by side, never inferred from nominal names, per spec §18's own wording.

---

## 10. What this deliberately does not add

- No differentiable persistent homology layer. `gudhi` is used as a fixed feature extractor with a stop-gradient
  regression target (§5, §7), not backpropagated through — avoiding an entire, currently-unvalidated class of
  differentiable-topology machinery this vault has no experience with.
- No graph attention (GAT) or `torch_geometric`/`dgl` dependency in v1 (§4) — a plain two-layer GCN only.
- No anatomical/skeleton graph construction in v1 (§4, Construction B) despite having a proven real-data result
  behind it — named and deferred, not silently dropped, per the spec's anti-gimmick principle.
- No full-rank ($O(d^2)$-per-token) density-matrix parameterization in v1 (§3, Construction C) — the mixture-of-
  $K$ construction is deliberately the cheaper, still-expressive-enough-to-be-nonzero-entropy choice.
- No quantum-fidelity distance in the structural-consistency loss (§7) — Frobenius distance only, to avoid a
  matrix square root per token per training step.
- No literal quantum-mechanical or biological claim (spec §6) — every "quantum-inspired" term above resolves to
  a concrete, checkable linear-algebra construction with no physical interpretation asserted.
- No RL, control, clinical intervention, multimodal fusion, or longitudinal dynamics modeling (spec §4,
  "Explicitly deferred") — this document formalizes only the frozen pretraining architecture and objectives.

---

## 11. Positioning against closest prior art

| Prior work | What it does | What it lacks (relative to this formalization) |
|---|---|---|
| Standard 3D MAE / masked-image-modeling foundation models | Real-valued latent, reconstruction + masking objectives | No structured (complex/probabilistic/relational/topological) latent; uncertainty, if present, is typically a bolt-on head, not load-bearing in the representation |
| `sheaf/model.py` (this vault, A1) | Multi-modal fusion with a disagreement-based uncertainty signal | Documented failure: signal conflates disagreement with magnitude (`gate_reports/gate_2.json`); no complex representation, no topology |
| `topology/*` (this vault, A3-adjacent) | Graph-native representations with hard topological guarantees, validated real-data domain-shift advantage | No probabilistic state, no complex representation; single-formalism (topology alone), which `signal_topology/FINDINGS.md` itself notes is not, by itself, a novel claim |
| Complex-valued / tensor-network ML generally (flagged Tier C in `FORMALISMS_TRIAGE.md`) | Complex-amplitude representations exist in the ML literature | Prior triage's concern was unproven clinical payoff over standard probabilistic UQ *in isolation* — this formalization's answer is that the claim under test is the **combination** (complex + probabilistic + graph + topology jointly), evaluated under parameter-matched, gated, H0-honest conditions, not any single formalism alone |

---

## 12. Falsification gates (run before committing)

1. **$\rho$ construction validity.** Every $\rho_i$ must be numerically PSD (min eigenvalue $\geq -10^{-6}$) and
   $\mathrm{Tr}(\rho_i)$ within $10^{-5}$ of 1, on synthetic input, before any real data is touched. If this
   fails, the construction in §3 is broken at the implementation level and nothing downstream is trustworthy.
   *Stop: fix the construction, do not proceed to Phase 1.*
2. **Entropy is not degenerate.** $S(\rho_i)$ must take a range of values across tokens on real preprocessed
   data (not collapse to a constant near 0 or $\log K$ for every token). *Stop: revisit the mixture-weight
   head $g_\theta$ or $K$ before trusting any downstream uncertainty claim.*
3. **Uncertainty is not a magnitude proxy.** The §8 magnitude-conflation check must show $S(\rho)$'s correlation
   with error surviving the partial-correlation-controlling-for-$\lVert\psi\rVert$ test. *Stop: if it collapses
   like the sheaf signal did, H4 is not supported by this construction as specified, and the uncertainty module
   needs to be redesigned before Phase 5 (not silently reported as a positive result).*
4. **Graph connectivity.** $G$'s adjacency must have no orphan nodes on synthetic input. *Stop: the top-$k$
   feature-similarity edge construction needs revisiting if any node is unreachable.*
5. **Topology summary is finite.** $T(X)$ must be finite and NaN/Inf-free across the calibration set. *Stop:
   downsampling resolution or intensity normalization needs revisiting before Phase 1.*
6. **Parameter matching holds — "the contribution is the structure, not the parameter count," checked.**
   $N_{\text{DIRAC-S}}$, $N_{\text{CNN}}$, $N_{\text{ViT}}$ must land within the locked tolerance using the §9
   complex-counting convention. *Stop: baseline width needs retuning before any comparison is run — an
   unmatched comparison cannot distinguish a structure effect from a capacity effect, so it is not a valid test
   of H1.*

Gates 1, 4-6 above are operationalized as `tests/test_rho_psd_trace1.py`, `test_graph_connectivity.py`, and
`test_param_count.py` in `IMPLEMENTATION_SPEC.md`'s Gate 0; gate 3 (magnitude-conflation) is operationalized as
`jobs/uncertainty_gate_job.py` in Phase 5, since it requires real prediction-error data not available at Gate 0.
