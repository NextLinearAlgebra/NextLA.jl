# Comprehensive report across the available experiments

`comprehensive.jl` combines the completed 20-matrix catalog, the supplied earlier
synthesis of six original/shifted pairs, the completed six-input legacy rerun,
the 15-setting CPU core sweep, and the raw five-dimension synthetic
accuracy CSV. It preserves source precision and cohort settings in a normalized
CSV. The earlier real tables are explicitly rounded transcriptions, not fabricated
raw measurements; the superseded real rows in the older accuracy CSV are excluded.

```bash
julia test/catalog/comprehensive.jl --prepare earlier-synthesis.txt \
  accuracy_20261008_225048.csv /path/to/catalog-27277754 /path/to/output \
  /path/to/extended-27277754-20261029-230602
julia test/catalog/comprehensive.jl --html /path/to/output /path/to/catalog-27277754
```

The second command embeds the catalog's existing PNG figures and supplemental
Plotly charts. The pinned official Plotly library is downloaded once to assets
and embedded so the HTML works offline. Supplemental charts can be exported as
PNG from their toolbar. `COMPREHENSIVE_INTERACTIVE=1` forces supplemental charts
to remain interactive even when exported PNGs exist.

For automated local exports, the generated HTML exposes a test-only
`?export=1` URL mode: after all charts render, it writes a JSON script element
`exported-charts` containing base64 PNGs and marks the document
`data-exports-ready=true`. A standard headless browser can capture those exports.
After saving PNGs to figures/<id>.png, rerun --html for a fully static offline
report. This avoids depending on any GPU or local native plotting DLL.

Outputs include 445 retained observations, a 31-input inventory, 20 supplemental
chart datasets, per-input dossiers, threshold-sensitive accuracy coverage, and
actual CPU factor speedup and efficiency. `data/primary-results.csv` selects 370
observations for the primary tables: 260 current original real, 60 historical
shifted, and 50 synthetic. Historical original real rows stay in `all-results.csv`
and are used alongside their historical shifts, never replaced by a fresh
original to form a cross-run pair. Core sweep observations stay separate from
the primary per-matrix lookup. Raw CSVs, settings, logs, per-case output and
hardware evidence are copied into `data/extended` unchanged.
`data/matrix-outcomes.csv` classifies all 26 current real matrices at the declared
forward cutoff into all-depth success, partial success, breakdown without a pass,
and complete-without-a-pass. It retains application labels, measured properties,
passing/failed methods, severe finite-error counts and input-probe coverage.

The reading order is controlled by `REPORT_SECTIONS`: findings and three overview
plots first (coverage, forward tradeoff, backward tradeoff), then a matrix-property
patterns chapter, performance, CPU scaling, synthetic trends, depth,
error definitions, failures, quantization, shifting, inventory and dossiers.
Experimental provenance and the audit trail remain available at the end.
The patterns chapter contrasts application families, dominance, quantization,
global value range, density and size, with counterexamples and no causal or
universal selection claim. Fifteen sections retain the high-level-to-detail order.
All 39 figures have accompanying interpretation. The 19 original catalog plots
are explicitly labeled as a 20-input subset; new overview plots use all 26 real
inputs. Timeouts are not assigned numeric factor times or speedup lower bounds.
The coverage plot includes both forward and Frobenius-scaled backward criteria.
The backward tradeoff uses the same GPU speedup and 1e-3 numeric screen, explicitly
described as a different criterion rather than an equivalent accuracy guarantee.
The real depth curves now use rerun medians where available; synthetic curves
retain historical minima. Supplemental PNGs must be re-exported whenever their
datasets change; force interactive generation before browser export to avoid
reusing an older static image.

Validation:

```bash
julia test/catalog/comprehensive_tests.jl
```

Tests use only standard libraries. When the prepared local dataset is available,
they additionally audit all normalized counts, known transcriptions, shifted-input
timing exclusions and document coverage. The implementation is tailored to this
supplied report format and complete catalog selection; a different historical
document or incomplete survey requires adapting the parser, not guessing values.
