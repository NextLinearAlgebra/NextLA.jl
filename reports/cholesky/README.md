# Comprehensive Cholesky report

The generated report snapshot is published on this branch independently of the
source code. Existing documentation is preserved.

- [Report file](index.html): download it, then open it in a browser. All 39 figures
  are embedded and no installation or network connection is needed to read it.
- [Interactive version](interactive.html): download and open to hover over the
  20 supplemental charts, zoom, toggle series and export chart images.
- [All normalized observations](data/all-results.csv): 445 rows with source labels.
- [Primary observations](data/primary-results.csv): 370 selected rows.
- [Matrix inventory](data/matrix-inventory.csv): 31 distinct inputs.
- [Real-matrix outcome groups](data/matrix-outcomes.csv): 26 current real inputs.
- [CPU scaling](data/cpu-scaling.csv): 15 measured settings and derived ratios.

Coverage includes the 20-matrix catalog, six fresh real-matrix reruns, historical
original/shifted pairs, five synthetic dense dimensions and the CPU core sweep.
Historical and current experiments retain their provenance and timing conventions.

Report generator: `vicki-development`, source commit `cd7bcf3`.

## Browser-viewable link

Once a repository administrator enables GitHub Pages using `gh-pages` and the
repository root (`/`), the static report will be available at:

https://nextlinearalgebra.github.io/NextLA.jl/reports/cholesky/

The interactive version will be at:

https://nextlinearalgebra.github.io/NextLA.jl/reports/cholesky/interactive.html

GitHub repository settings: Settings > Pages > Deploy from a branch > gh-pages >
/(root) > Save. This preserves the existing documentation at the site root/dev.
Until Pages is enabled, use the GitHub download links above.