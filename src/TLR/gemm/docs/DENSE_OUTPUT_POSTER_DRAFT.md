# Dual-Packed Exact-Rank TLR GEMM for Memory-Efficient GPU Linear Layers

**Suggested subtitle:** Symbolic scheduling and grouped low-precision GEMMs for dense accumulation

## One-sentence contribution

We introduce an exact-rank, dual-packed tile low-rank representation and a
workspace-aware symbolic scheduler that turns irregular low-rank products into
fused grouped GEMMs, including normal and transposed operands, while producing a
dense result.

## Positioning and terminology

Dense weight matrices are expensive both to store and to apply. Low-rank and
hardware-aware structured matrices, including Monarch matrices, reduce this
cost by restricting the structure of a linear transformation. A single global
low-rank factorization can, however, be too restrictive.

Tile low-rank (TLR) matrices provide a middle ground. The matrix is divided into
tiles, and every tile has its own factorization

\[
    A_{ij} \approx U_{ij}V_{ij}^{T},
    \qquad
    U_{ij}\in\mathbb{R}^{m_i\times r_{ij}},
    \quad
    V_{ij}\in\mathbb{R}^{n_j\times r_{ij}}.
\]

This is best described as **blockwise low numerical rank** or **data
sparsity**, not conventional elementwise structured sparsity. It is attractive
for machine-learning weights when independently trained or compressed weight
blocks admit small, nonuniform ranks.

Existing GPU TLR and block-low-rank libraries demonstrate that batched dense
kernels can accelerate these formats. Adaptive ranks nevertheless introduce
irregular GEMM shapes, fragmented factor storage, extra kernel launches, and
poor compatibility with low-precision Tensor Core execution.

Our work asks:

> Can exact, independently varying tile ranks preserve the fusion and GPU
> efficiency of a padded fixed-rank representation?

## Abstract / ACM SRC summary

Structured linear transformations, including low-rank and Monarch matrices,
can reduce the storage and arithmetic cost of neural-network layers. When a
weight matrix exhibits blockwise low numerical rank, tile low-rank (TLR)
storage offers a more expressive alternative to a single global
factorization: each tile has an independent rank. Existing GPU implementations
often recover regularity by padding every factor to a common maximum rank,
wasting memory and arithmetic, while exact ranks produce heterogeneous GEMMs
and significant launch overhead.

We present **CompressedFTLR**, an exact-rank, dual-packed TLR representation,
and a dense-output GEMM lowering designed for grouped low-precision GPU
kernels. Outer factors are packed by tile row and inner factors by tile column,
with prefix metadata locating each variable-rank factor. The two packing orders
preserve contiguous reduction stacks for both normal and transposed operands.
A reusable symbolic phase inspects only matrix geometry, rank metadata,
precision, and the workspace budget. It selects the lower-cost folding
direction, constructs grouped-GEMM descriptors, and schedules the largest
contiguous output-row run that fits memory. The numerical phase then executes
the prepared schedule without reconstructing metadata.

We implement the method in Julia, separating backend-neutral planning from a
CUDA grouped-GEMMEx backend. The prototype targets FP16, TF32, FP32, and BF16
execution because low precision is important to machine-learning inference yet
makes irregular alignment and Tensor Core utilization particularly difficult.
Our current kernel study evaluates dense accumulation for compressed ×
compressed and dense × compressed products. It demonstrates substantial
speedups over dense GEMM and lower factor storage than maximum-rank padding.
End-to-end model accuracy and inference throughput remain future work.

## What is new

### 1. Exact-rank dual-packed storage

- Every tile retains its actual rank \(r_{ij}\); no common logical maximum rank
  is required.
- Outer factors are packed by tile row.
- Inner factors are packed by tile column.
- Prefix offsets locate variable-length factors.
- Tail tiles and alignment padding are represented without changing logical
  tile ranks.
- The two packing orders support contiguous fused stacks under all four
  normal/transpose operand combinations.

The format should be illustrated as two physical arrays:

```text
outer storage, row packed             inner storage, column packed

row 1: U11 | U12 | ... | U1q          col 1: V11 | V21 | ... | Vp1
row 2: U21 | U22 | ... | U2q          col 2: V12 | V22 | ... | Vp2
...
       ^ prefix offsets                      ^ prefix offsets
```

This is dual packing of the two different factors, not duplication of either
factor.

### 2. Fusion-preserving dense lowering

For compressed × compressed accumulation,

\[
    C \leftarrow \alpha AB + \beta C,
\]

the lowering uses three grouped-GEMM stages. Conceptually:

1. Contract the inner factors of \(A\) and the outer factors of \(B\).
2. Propagate the contraction through one remaining factor.
3. Fuse the rank stack and accumulate into a dense output row or tile.

The planner may fold from the left or right. It evaluates both legal forms from
rank metadata and selects one fold for the scheduled run using exact workspace
and arithmetic costs. Dual packing preserves the required contiguous stacks
for `N/N`, `N/T`, `T/N`, and `T/T` products.

Dense × compressed and compressed × dense products use the corresponding
two-stage lowering. Dense × compressed is the most direct proxy for inference:
a dense activation matrix is multiplied by a compressed weight matrix and
accumulated into a dense activation.

### 3. Reusable symbolic analysis

The symbolic phase performs no numerical multiplication. From dimensions,
tile extents, ranks, operand transposes, precision, and the workspace budget,
it prepares:

- legal fold candidates and their exact costs;
- grouped-GEMM shapes and pointer mappings;
- aligned intermediate offsets;
- output-row run boundaries;
- zero-rank and beta-only work;
- workspace requirements.

For fixed compressed weights and fixed batch shapes, this metadata can be
reused over repeated numerical executions. This is especially natural for
inference, where a model's weight structure is fixed across requests or
tokens.

### 4. Memory-aware execution

The scheduler computes the memory needed for each output row and greedily
selects the largest contiguous run that fits the supplied workspace. A larger
workspace therefore increases concurrent rows without changing correctness or
requiring a different storage format.

The rows-per-run measurements show that concurrency is important for small
problems, whereas one or a few rows can already expose sufficient grouped work
for large matrices. This motivates dynamic scheduling rather than a fixed
rows-per-run constant.

### 5. Low-precision grouped execution

The numerical stages use CUDA grouped GEMMEx rather than launching one GEMM per
irregular tile. The implementation supports Tensor Core-oriented precision
policies, including:

- FP16 storage and FP32 accumulation;
- FP32 storage with TF32 compute;
- full FP32;
- BF16 storage and FP32 accumulation on supported GPUs.

Logical ranks remain exact. Physical leading dimensions and intermediate
offsets are aligned where required by low-precision grouped kernels.

## Poster layout

### Header

- Title and one-sentence contribution.
- One headline number from the final validated data.
- Small visual showing dense tiles replaced by independently ranked factors.

### Left column — Motivation and gap

1. Dense linear layers cost memory bandwidth, capacity, and arithmetic.
2. Structured matrices reduce these costs, but useful structure must also map
   efficiently to GPU hardware.
3. TLR gives every block its own basis and rank.
4. Existing fixed-rank/padded batching wastes work; exact ranks create
   irregular kernels.
5. Research question and contributions.

Include a small related-work box:

- **KBLAS:** GPU batched TLR GEMM.
- **H2OPUS:** adaptive-rank TLR approximation and factorization/update
  algorithms using dynamic batching.
- **STRUMPACK:** BLR/HSS and related structured dense solvers.
- **This work:** exact-rank dual packing, transpose-complete dense-output
  lowering, reusable symbolic grouped schedules, low precision, and explicit
  workspace control.

### Center column — Representation and algorithm

Use two diagrams:

1. Dual-packed factors with rank-prefix metadata.
2. Three-stage compressed × compressed lowering, showing where tile tasks
   become grouped GEMMs and where the final dense accumulation is fused.

Place the symbolic phase above the numerical phase:

```text
dimensions + ranks + transpose + precision + workspace
                         |
                  symbolic analysis
                         |
     fold | row runs | GEMM groups | aligned offsets
                         |
        repeated numerical executions with new values
```

### Right column — Results

Use no more than three main quantitative figures.

#### Figure 1: Inference-facing strong scaling

**Dense × CompressedFTLR → dense, speedup over dense GEMM**

- x-axis: matrix dimension, logarithmic base 2;
- y-axis: measured time of dense GEMM / compressed time;
- one curve per precision;
- fix \(b=N/8\);
- use one clearly defined rank distribution;
- add a horizontal \(1\times\) line.

This should be the largest plot because it most closely represents dense
activations multiplied by compressed weights.

**Data requirement:** rerun the variable-rank dense × compressed experiment
after aligning its dense row-slab leading dimension. Current heterogeneous-rank
measurements contain a fallback/per-task launch artifact and should not appear
on the poster.

#### Figure 2: Exact ranks versus maximum-rank padding

Use a two-part figure:

- runtime ratio of the exact-rank and padded dense-output algorithms;
- factor-storage ratio for uniform and skewed variable-rank distributions.

Match nominal tile size and maximum admissible rank. Label the comparison
honestly: exact ranks reduce storage and arithmetic, but heterogeneous grouped
low-precision GEMMs may be slower than uniform fixed-shape batching in some
large cases.

The current results show that exact-rank storage is not an unconditional
performance replacement for padding. Its strongest value is the
memory/performance trade-off under rank heterogeneity.

#### Figure 3: Why symbolic, memory-aware scheduling matters

Combine:

- a rows-per-run heat map normalized to one row; and
- transient execution / analyzed execution versus matrix size.

The message is:

- multiple rows are important at small sizes;
- large problems saturate with one or a few rows;
- reusable analysis removes meaningful repeated planning/allocation overhead,
  especially for fine tile grids.

## Claims supported by the current compressed × compressed data

These numbers must be regenerated directly by the final plotting script before
submission, but the presently inspected results support:

- the compressed algorithm wins against dense GEMM in 220 of 225 measured
  compressed × compressed cases;
- all measured cases from dimension 8192 onward are faster than dense GEMM;
- median speedup grows from approximately \(3.4\times\) at 4096 to
  \(6.6\times\) at 32768;
- selected 65536 cases reach approximately \(27\times\) speedup;
- reusable analysis is most important for fine tile grids;
- increasing rows per run helps most at small dimensions and can hurt at large
  dimensions.

Do not present effective dense FLOP/s as the primary metric. Use measured time,
speedup, executed low-rank FLOPs, and storage bytes. Dense-equivalent FLOP/s
can appear only as a clearly labeled secondary metric.

## What the poster should not claim yet

- No end-to-end neural-network inference speedup has been measured.
- No accuracy/quality result shows that a trained model retains quality after
  adopting this tile-rank distribution.
- The grouped exact-rank numerical backend is currently CUDA-specific; Julia
  and multiple dispatch make backend separation clean, but do not by
  themselves demonstrate performance portability.
- Square GEMM microbenchmarks do not cover the rectangular shapes of every
  Transformer or MLP layer.
- BF16 performance should not be reported until measured on SM80 or newer
  hardware.
- Exact-rank grouped execution does not beat maximum-rank padded batching in
  every regime.

Use the phrase **proof-of-concept kernel implementation for structured linear
layers** until end-to-end inference and model-quality experiments exist.

## Recommended conclusion

Exact tile ranks need not force a sequence of tiny GPU kernels. By packing the
two low-rank factors along complementary tile axes and moving shape decisions
to a reusable symbolic phase, irregular TLR products can retain fusion across
operand transposes and execute as a small number of grouped low-precision
GEMMs. The result is a controllable trade-off between factor memory, arithmetic
work, and temporary workspace. The current Julia/CUDA prototype establishes
the kernel-level opportunity; evaluating trained structured layers and
end-to-end inference is the next step.

## Evidence still needed before submission

1. Record GPU model, CUDA, Julia, CUDA.jl, and library versions.
2. Rerun dense × compressed after the alignment fix.
3. Export actual allocated factor bytes for padded and exact-rank formats.
4. Generate all figure data from one versioned plotting/analysis script.
5. Report medians of the three timed repetitions and show individual samples
   or min/max only if the poster has space.
6. Add at least one rectangular layer-like shape if time permits.
7. If using “inference” in the title, add a small repeated-weight experiment
   that reports analysis amortization over multiple numerical executions.

## Related work links

- Monarch: <https://arxiv.org/abs/2204.00595>
- H2OPUS-TLR: <https://arxiv.org/abs/2108.11932>
- Batched Tile Low-Rank GEMM on GPUs:
  <https://repository.kaust.edu.sa/items/baccd4bb-704d-4a49-9989-2ad59b0d46e3>
- STRUMPACK structured dense solvers:
  <https://portal.nersc.gov/project/sparse/strumpack/master/dense.html>
