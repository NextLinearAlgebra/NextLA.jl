# Comprehensive report across the available experiments

`comprehensive.jl` combines the completed 20-matrix catalog, the supplied earlier
synthesis of six original/shifted pairs, and the raw five-dimension synthetic
accuracy CSV. It preserves source precision and cohort settings in a normalized
CSV. The earlier real tables are explicitly rounded transcriptions, not fabricated
raw measurements; the superseded real rows in the older accuracy CSV are excluded.

```bash
julia test/catalog/comprehensive.jl --prepare earlier-synthesis.txt \
  accuracy_20261008_225048.csv /path/to/catalog-27277754 /path/to/output
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

Outputs include 370 normalized observations, a 31-input inventory, 13 supplemental
chart datasets, per-input result dossiers, threshold-sensitive accuracy coverage,
and separate earlier/current CPU timing cohorts. The document discusses every
figure and finishes with an overall conclusion and prioritized measurements.

The generated report does not claim that pending reruns or CPU scaling completed.
Use fresh exported data to update those comparisons. Preserve the original raw
sources and input hashes alongside new run-specific settings.

Validation:

```bash
julia test/catalog/comprehensive_tests.jl
```

Tests use only standard libraries. When the prepared local dataset is available,
they additionally audit all normalized counts, known transcriptions, shifted-input
timing exclusions and document coverage. The implementation is tailored to this
supplied report format and complete catalog selection; a different historical
document or incomplete survey requires adapting the parser, not guessing values.
