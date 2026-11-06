"""Consolidate an ordering sweep into standalone HTML, PDF, and chart artifacts.

python -m pip install matplotlib numpy
python test/ordering/summarize.py INPUT_DIRECTORY OUTPUT_DIRECTORY
Only metrics.csv rows count as completed; leftover images never imply completion.
"""
import argparse
import base64
import csv
import hashlib
import html
import io
import json
import math
import re
import textwrap
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import ListedColormap
import numpy as np

METHODS = ["natural", "amd", "rcm", "scotch", "cholmod", "stiles", "random"]
COLORS = dict(zip(METHODS, ["#64748b", "#0891b2", "#d97706", "#7c3aed", "#0d9488", "#db2777", "#94a3b8"]))
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "savefig.facecolor": "white"})


def short(name):
    return name.removeprefix("inla_graph_")


def load(root):
    groups, metadata, provenance = {}, {}, []
    for path in sorted(root.glob("*/metrics.csv")):
        rows = list(csv.DictReader(path.open(encoding="utf-8", newline="")))
        if not rows:
            continue
        name = path.parent.name
        groups[name] = {r["method"]: r for r in rows}
        if len(groups[name]) != len(rows):
            raise ValueError(f"Duplicate method rows in {path}")
        for r in rows:
            if r["matrix"] != name:
                raise ValueError(f"Mismatched matrix in {path}")
            for key, value in list(r.items()):
                if key not in ("matrix", "method", "status", "detail") and value:
                    r[key] = float(value)
        page = path.parent / "index.html"
        match = re.search(r"<pre>(.*?)</pre>", page.read_text(encoding="utf-8"), re.S) if page.exists() else None
        metadata[name] = html.unescape(match.group(1)) if match else "Metadata unavailable"
        # The original sweep applied current sTiles' original->new map directly
        # as A[p,p]. Retain the measurements as audit evidence, exclude ranking.
        if "stiles" in groups[name] and "stiles_current_export=inverted_for_gather" not in metadata[name]:
            groups[name]["stiles"]["status"] = "DIRECTION_INVALID"
        provenance.append({"source": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    if not groups:
        raise ValueError("No per-matrix metrics.csv files found")
    return groups, metadata, provenance


def value(groups, name, method, key):
    r = groups[name].get(method, {})
    return r.get(key, math.nan) if r.get("status") == "OK" else math.nan


def winners(groups, name, key):
    values = [(r[key], method) for method, r in groups[name].items() if r["status"] == "OK" and key in r]
    if not values:
        return []
    best = min(v for v, _ in values)
    return [m for v, m in values if v == best]


def asset(root, name, file):
    for folder in (root/name/name, root/name):
        if (folder/file).exists():
            return folder/file
    return None


def occupancy(path):
    """Rasterize the generator's binary SVG cells exactly; never densify A."""
    tree = ET.parse(path).getroot()
    size = int(float(tree.attrib["viewBox"].split()[2]))
    cells = np.zeros((size, size), dtype=np.uint8)
    for node in tree:
        if node.tag.endswith("rect") and node.get("x") is not None:
            x, y = int(float(node.get("x"))), int(float(node.get("y")))
            cells[y, x] = 2 if node.get("fill") == "#e87924" else 1
    return cells


def splits(n, depth, offset=0):
    if depth <= 0 or n <= 1:
        return []
    power = 1 << (n.bit_length()-1)
    mid = n//2 if n == power else power
    return [(offset, offset+mid, offset+n)] + splits(mid, depth-1, offset) + splits(n-mid, depth-1, offset+mid)


def pattern(ax, root, groups, name, method, kind):
    file = f"{method}-{'matrix' if kind == 'input' else 'factor'}.svg"
    path = asset(root, name, file)
    if path is None or method not in groups[name] or groups[name][method]["status"] != "OK":
        label = "Needs corrected rerun" if method == "stiles" else "Not measured"
        ax.text(.5, .5, label, ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return
    cells = occupancy(path)
    ax.imshow(cells, cmap=ListedColormap(["white", "#164e63", "#e87924"]), vmin=0, vmax=2,
              interpolation="nearest", extent=(0, 1, 1, 0))
    n = int(groups[name][method]["n"])
    for lo, cut, hi in splits(n, 3):
        ax.plot([lo/n, hi/n], [cut/n]*2, color="#e11d48", lw=.45, alpha=.8)
        ax.plot([cut/n]*2, [lo/n, hi/n], color="#e11d48", lw=.45, alpha=.8)
    ax.set_xticks([]); ax.set_yticks([])
    if kind == "input":
        ax.set_title(method + " input", fontsize=10)
    else:
        ax.set_title(f"L: {groups[name][method]['nnz_L']:,.0f} nonzeros", fontsize=10)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    root, out = args.input.resolve(), args.output.resolve()
    if root == out:
        raise ValueError("Choose a separate output directory")
    out.mkdir(parents=True, exist_ok=True)
    figs = out/"figures"; figs.mkdir(exist_ok=True)
    groups, metadata, provenance = load(root)
    names = sorted(groups)
    rows = [r for g in groups.values() for r in g.values()]
    good = [r for r in rows if r["status"] == "OK"]
    counts = Counter(r["method"] for r in good)
    speedwins = Counter(m for n in names for m in winners(groups, n, "factor_median_s"))
    allcore = [n for n in names if all(m in groups[n] and groups[n][m]["status"] == "OK" for m in ("amd", "rcm", "scotch", "cholmod"))]
    maxres = max(r["solve_residual"] for r in good)
    maxfwd = max(r["forward_error"] for r in good)
    summary = [
        f"{len(names)} matrices contain {len(rows)} saved measurements. A native audit found a permutation-direction bug in the original sTiles adapter: its 22 saved rows are retained but excluded from ordering rankings, leaving {len(good)} valid comparisons. Animal2 was not included. These are CPU sparse Float64 CHOLMOD measurements, not H200 or mixed-precision timings.",
        f"SCOTCH has the lowest recorded median factor time on {speedwins['scotch']}/{len(names)} matrices among measured candidates. AMD and CHOLMOD are strong alternatives. Exact timing ranks from three repetitions are descriptive, especially for millisecond cases.",
        "The body matrices show large gains from conventional fill-reducing orders. For bodyy5, SCOTCH's median factor time is 0.0129 s versus 0.610 s in natural order (47.4x faster); AMD reduces factor nonzeros from 7.45 million to 0.636 million (11.7x smaller).",
        "Animal1 favors SCOTCH more modestly: 18.1% fewer factor entries and 17.0% lower median time than natural order. However, its top-level off-diagonal input block becomes about 42.8x denser. Sparse fill and recursive-block occupancy are different objectives.",
        "Withdraw the previous conclusions about sTiles wins and regressions: the current native API returns original node -> reordered position, while A[p,p] needs the inverse. On a scrambled 225-node grid, native sTiles has 1,780 factor entries, direct application gives 5,483, and inverse application restores exactly 1,780. The adapter is fixed; the real-matrix sTiles cases need rerunning.",
        f"All {len(good)} saved comparisons pass the runner's residual checks. Maximum solve residual is {maxres:.2e}; maximum forward error is {maxfwd:.2e}. This supports Float64 correctness for these probes, not a guarantee about mixed precision or all right-hand sides.",
        "Next: rerun only the corrected sTiles cases, then compare natural, AMD and SCOTCH on animal1/bodyy4/5/6 in a controlled Float64 NextLA GPU experiment. The existing AMD/RCM/SCOTCH/CHOLMOD measurements remain usable. Measure GPU timing and accuracy before changing precision schedules."
    ]
    # These narrative examples must never silently apply to a different dataset.
    required = {"bodyy5": ("scotch", "natural", "amd"), "inla_graph_animal1": ("scotch", "natural"), "inla_graph_83o4NNNo": ("stiles",)}
    for name, methods in required.items():
        if name not in groups or not all(m in groups[name] for m in methods):
            raise ValueError("This curated analysis requires the documented bodyy5, animal1, and 83o4NNNo cases")
    # Assert rounded claims against data to avoid publishing stale narrative.
    assert math.isclose(value(groups,"bodyy5","natural","factor_median_s")/value(groups,"bodyy5","scotch","factor_median_s"),47.4,rel_tol=.01)
    assert all(r["status"] in ("OK", "DIRECTION_INVALID") for r in rows)
    context = [
        ("What reordering changes", "We move the same rows and columns together: Ap = A[p,p]. The input nonzero count, exact eigenvalues, positive definiteness and spectral condition number stay the same. What changes is elimination order: Cholesky may create new nonzeros, called fill. Reordering also moves values into different recursive blocks."),
        ("What the stiles row actually measures", "Our runner calls sTiles.analyze(A), extracts a permutation and closes the native object. It then applies that permutation and factors with CHOLMOD using identity internal ordering. We do not time native sTiles numerical factorization, and we do not ask sTiles to pick the fastest entry in our CHOLMOD comparison table."),
        ("What sTiles means by best", "The paper describes testing RCM, AMD variants, METIS and SCOTCH by symbolic factorization. Candidates are scored using factor fill and active tiles. Sparse cases favor low fill; denser cases can favor fewer active tiles despite more fill. Nested dissection may receive a bounded fill allowance for parallelism. A separate selector then chooses the factor-storage route. These are heuristics for sTiles, not a promise of fastest CHOLMOD or NextLA execution."),
        ("Why the old sTiles pictures looked unexpectedly bad", "Our adapter used the current export in the wrong direction. The native check above demonstrates the difference against sTiles' own symbolic factor count. Small solve residuals did not catch it: a permutation and its inverse both define valid reordered SPD systems. All old stiles rows and pictures are excluded from conclusions until direction-corrected real-matrix runs are available."),
        ("How to judge good or bad", "Correctness is a prerequisite. For sparse storage, smaller nnz(L) is better. For a one-off solve, compare setup + permutation + factorization + solve time; for repeated values with the same pattern, amortize setup. For a tile solver, active tiles, cache locality and parallelism also matter. A narrow bandwidth or a prettier picture alone is not proof of lower cost."),
        ("What this means for NextLA and precision", "Use measured runtime and memory with the actual NextLA kernels as the performance target, subject to your accuracy tolerance. Permutation changes which values enter off-diagonal blocks, and floating-point errors can change with operation order, but these input pictures cannot predict precision safety. Lower density does not automatically save dense-kernel work. Our red lines show the storage split geometry, not a tested precision schedule.")
    ]
    sections, appendix, manifest = [], [], []
    pdf = PdfPages(out/"ordering-summary.pdf", metadata={"Title":"Cholesky ordering: consolidated analysis", "Author":"NextLA experiment report"})

    def pdf_page(fig, title, takeaway, caption):
        # Render a separate page so every figure has its takeaway BELOW it.
        buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=160, bbox_inches="tight"); buf.seek(0)
        page = plt.figure(figsize=(11.7, 10.5), facecolor="white")
        page.text(.05,.965,title,fontsize=15,weight="bold",va="top")
        ax = page.add_axes([.04,.27,.92,.63]); ax.imshow(plt.imread(buf)); ax.set_axis_off()
        page.text(.05,.235,textwrap.fill("Takeaway: "+takeaway,110),fontsize=10.5,weight="bold",va="top")
        page.text(.05,.12,textwrap.fill(caption,140),fontsize=8.5,va="top",color="#475569")
        pdf.savefig(page); plt.close(page)

    def add(fig, title, takeaway, caption, *, detail=False):
        index = len(manifest)+1
        file = f"{index:02d}-figure.png"
        fig.savefig(figs/file,dpi=180,bbox_inches="tight")
        fig.savefig(figs/file.replace(".png",".svg"),bbox_inches="tight")
        encoded = base64.b64encode((figs/file).read_bytes()).decode()
        block = f'<figure id="fig{index}"><h3>{html.escape(title)}</h3><img src="data:image/png;base64,{encoded}" alt="{html.escape(title)}"><figcaption><strong>Takeaway: {html.escape(takeaway)}</strong><p>{html.escape(caption)}</p></figcaption></figure>'
        (appendix if detail else sections).append(block)
        pdf_page(fig,title,takeaway,caption)
        manifest.append({"figure":index,"title":title,"takeaway":takeaway,"caption":caption,"file":f"figures/{file}"})
        plt.close(fig)

    page = plt.figure(figsize=(11.7,10.5)); page.text(.07,.94,"Cholesky ordering: overall takeaways",fontsize=22,weight="bold")
    y=.87
    for i, paragraph in enumerate(summary,1):
        lines=textwrap.fill(f"{i}. {paragraph}",105)
        page.text(.07,y,lines,fontsize=11,va="top",linespacing=1.35)
        y -= .025*(len(lines.splitlines())+1.1)
    pdf.savefig(page); plt.close(page)
    for start in (0,3):
        page=plt.figure(figsize=(11.7,10.5))
        page.text(.07,.94,"Context: sTiles and a good reordering" if start==0 else "Correction and how to judge an ordering",fontsize=20,weight="bold")
        y=.85
        for title,paragraph in context[start:start+3]:
            page.text(.07,y,title,fontsize=12,weight="bold",va="top")
            lines=textwrap.fill(paragraph,105)
            page.text(.07,y-.04,lines,fontsize=11,va="top",linespacing=1.3)
            y -= .04+.024*(len(lines.splitlines())+2)
        if start==0:
            page.text(.07,.07,"Source: Abdul Fattah et al., Section 3: https://arxiv.org/html/2609.29765v1#S3",fontsize=9)
        pdf.savefig(page);plt.close(page)

    # 1. Every matrix, normalized to the same available reference.
    fig, axes = plt.subplots(1,2,figsize=(14,11),layout="constrained")
    for ax,key,title in zip(axes,["factor_median_s","nnz_L"],["Median factor time / AMD", "Factor nonzeros / AMD"]):
        data=np.array([[value(groups,n,m,key)/value(groups,n,"amd",key) for m in METHODS] for n in names])
        cmap=plt.get_cmap("RdYlBu_r").copy(); cmap.set_bad("#e2e8f0")
        im=ax.imshow(np.log2(data),cmap=cmap,vmin=-2,vmax=6,aspect="auto")
        for i in range(len(names)):
            for j in range(len(METHODS)):
                v=data[i,j]
                ax.text(j,i,"—" if not np.isfinite(v) else f"{v:.1f}x",ha="center",va="center",fontsize=7,color="black")
        ax.set_xticks(range(len(METHODS)),METHODS,rotation=45,ha="right")
        ax.set_yticks(range(len(names)),[short(n) for n in names],fontsize=9); ax.set_title(title)
    fig.colorbar(im,ax=axes,shrink=.5,label="log2(ratio): blue better; red worse; color clipped, labels exact")
    add(fig,"1. The full comparison: no universal best ordering",
        f"SCOTCH has the fastest recorded median on {speedwins['scotch']}/{len(names)} matrices among valid completed candidates. Old sTiles rows are excluded pending corrected reruns.",
        f"All {len(names)} matrices are shown. Ratios are relative to AMD (1x = equal). Gray cells mean unavailable or withheld, never zero time: the 22 old sTiles measurements are withheld for the direction bug; natural/random exist on {counts['natural']} matrices. Candidate sets differ, so winner counts are not balanced rankings. Time includes fresh symbolic and numeric factorization, excluding setup/permutation.")

    # 2. Priority body-family results.
    body=[n for n in ["bodyy4","bodyy5","bodyy6"] if n in groups]
    fig,axes=plt.subplots(1,2,figsize=(13,4.4),layout="constrained")
    choices=["natural","amd","scotch","stiles"]
    for ax,key,title in zip(axes,["nnz_L","factor_median_s"],["Factor storage: nonzeros", "Median complete factorization (s)"]):
        for j,m in enumerate(choices):
            x=np.arange(len(body))+(j-1.5)*.19
            ax.bar(x,[value(groups,n,m,key) for n in body],width=.18,label=m,color=COLORS[m])
        ax.set_xticks(range(len(body)),body);ax.set_yscale("log");ax.set_title(title);ax.grid(axis="y",alpha=.2)
    axes[0].legend(ncol=2)
    add(fig,"2. Bodyy4/5/6: the clearest practical ordering opportunity",
        "On bodyy5, SCOTCH is 47.4x faster than natural order; AMD's factor is 11.7x smaller. The old sTiles measurements are invalid for evaluating its chosen order.",
        "Log scales reveal large differences; storage counts are numerical nnz(L), not allocated bytes. Bodyy6 lacks a completed natural baseline. The sTiles slot is blank because the original adapter used the wrong permutation direction; it does not mean zero storage or zero time.")

    fig,axes=plt.subplots(2,4,figsize=(13,7),layout="constrained")
    for j,m in enumerate(choices):
        pattern(axes[0,j],root,groups,"bodyy5",m,"input")
        pattern(axes[1,j],root,groups,"bodyy5",m,"factor")
    add(fig,"3. Bodyy5: see where the fill goes",
        "Reordering preserves all input nonzeros, but AMD and SCOTCH substantially reduce the nonzeros created during elimination.",
        "Top: reordered input. Bottom: L in the same coordinates. Blue marks occupied pixels; orange in L marks fill absent from that input. Red lines follow three levels of the SymmMixedPrec diagonal split rule. Each pixel aggregates a region for n > 600; orange takes priority, so apparent area is not an exact fill count.")

    fig,axes=plt.subplots(2,3,figsize=(11,7),layout="constrained")
    for j,m in enumerate(["natural","scotch","stiles"]):
        pattern(axes[0,j],root,groups,"inla_graph_animal1",m,"input")
        pattern(axes[1,j],root,groups,"inla_graph_animal1",m,"factor")
    add(fig,"4. Animal1: a modest sparse-factor win changes the recursive layout",
        "SCOTCH cuts animal1's factor entries by 18.1% and median factor time by 17.0% versus natural order; the recursive off-diagonal region becomes much denser.",
        "Natural: 11.04 million factor nonzeros, 0.639 s median; SCOTCH: 9.05 million, 0.530 s. AMD and CHOLMOD have the same nnz(L) as natural, which alone does not prove equal permutations. The sTiles panel is withheld for the direction bug. This is animal1 (n = 5,126); animal2 was not run. Pixel and fill colors follow Figure 3.")

    # 5. Independent native symbolic-factor audit, not a real-matrix rerun.
    fig,ax=plt.subplots(figsize=(10,4.5),layout="constrained")
    labels=["Native sTiles factor", "Old adapter: raw export", "Corrected: inverse export"]
    bars=ax.bar(labels,[1780,5483,1780],color=["#0d9488","#db2777","#0891b2"])
    ax.bar_label(bars,padding=4);ax.set_ylim(0,6500);ax.set_ylabel("Exact factor nonzeros");ax.set_title("Scrambled 15 × 15 grid (225 unknowns)")
    add(fig,"5. The sTiles outliers exposed a permutation-direction bug",
        "Inverting the current native export exactly matches sTiles' 1,780-entry factor; direct use creates 5,483 entries. Old real-matrix sTiles conclusions are withdrawn.",
        "A separate Python process loaded native sTiles 3.0.0 (2026-09-24), the Julia wrapper's default release, avoiding a Julia/Windows DLL conflict. A 225-node scrambled grid with its diagonal present was analyzed with one core, tile size 40 and auto tile regime. Independent graph elimination computed exact symbolic counts for both directions. This diagnoses the adapter; it is not a corrected measurement of bodyy5 or any other catalog matrix.")

    # 6. Actual storage hierarchy measurements.
    fig,axes=plt.subplots(1,2,figsize=(12,4.6),layout="constrained")
    blockstats={}
    for name in ["bodyy5","inla_graph_animal1"]:
        blockstats[name]={}
        for m in choices:
            p=asset(root,name,m+"-blocks.csv")
            if p and groups[name][m]["status"] == "OK":
                with p.open() as f:
                    blockstats[name][m]=next(csv.DictReader(f))
    for ax,key,title in zip(axes,["density","max_abs"],["Root off-diagonal input density", "Root off-diagonal input maximum |value|"]):
        for j,m in enumerate(choices):
            ax.bar(np.arange(2)+(j-1.5)*.19,[float(blockstats[n][m][key]) if m in blockstats[n] else math.nan for n in blockstats],width=.18,color=COLORS[m],label=m)
        ax.set_xticks(range(2),[short(n) for n in blockstats]);ax.set_yscale("log");ax.set_title(title)
    axes[0].legend(ncol=2)
    add(fig,"6. Recursion: reducing fill does not minimize off-diagonal occupancy",
        "Animal1's root off-diagonal density rises from 0.291% to 12.48% with SCOTCH (42.8x), despite its smaller sparse factor.",
        "Bodyy5's root block has 2,205 rows × 16,384 columns; animal1's has 1,030 × 4,096. The split sizes do not depend on permutation. These CSV statistics describe original input values within the storage hierarchy, before Schur updates. They neither predict mixed-precision error nor establish GPU speedup; dense NextLA kernels do not skip zeros just because input occupancy decreases.")

    # 7. Locality objective vs fill.
    fig,ax=plt.subplots(figsize=(10,5),layout="constrained")
    for m in ["rcm","scotch","stiles"]:
        common=[n for n in names if np.isfinite(value(groups,n,m,"nnz_L"))]
        x=[value(groups,n,m,"bandwidth")/max(value(groups,n,"amd","bandwidth"),1) for n in common]
        y=[value(groups,n,m,"nnz_L")/value(groups,n,"amd","nnz_L") for n in common]
        ax.scatter(x,y,label=m,color=COLORS[m],alpha=.75,s=45)
    ax.axhline(1,color="gray",lw=.8);ax.axvline(1,color="gray",lw=.8)
    ax.set_xscale("log");ax.set_yscale("log");ax.set_xlabel("Input bandwidth / AMD (left = narrower)");ax.set_ylabel("Factor nonzeros / AMD (down = smaller)");ax.legend()
    add(fig,"7. A narrow-looking band is not the same as a small factor",
        "Judge bandwidth and fill separately. RCM is useful as a locality comparison, but is not a consistent fill winner in this sample.",
        "Each point is one completed matrix/method comparison to AMD. Lower-left improves both measures. Bandwidth is the largest |i-j| among input nonzeros. RCM here is the implemented minimum-degree-root BFS variant, not a pseudo-peripheral-root variant. No causal or mixed-precision conclusion follows from this scatter alone.")

    # 8. Numerical diagnostics.
    fig,axes=plt.subplots(1,3,figsize=(13,4),layout="constrained")
    for ax,key,title in zip(axes,["factor_residual","solve_residual","forward_error"],["Factor-action residual","Solve residual","Forward error"]):
        for j,m in enumerate(METHODS):
            vals=[r[key] for r in good if r["method"]==m]
            if vals:
                offsets=np.linspace(-.15,.15,len(vals))
                ax.scatter(j+offsets,vals,color=COLORS[m],s=15,alpha=.65)
        ax.set_yscale("log");ax.set_xticks(range(7),METHODS,rotation=65);ax.set_title(title);ax.grid(axis="y",alpha=.15)
        if key in ("factor_residual","solve_residual"):
            ax.axhline(1e-10,color="#e11d48",ls="--",label="Runner threshold")
    add(fig,"8. Float64 correctness is strong across the completed comparisons",
        f"All {len(good)} saved rows pass residual checks: worst solve residual {maxres:.2e}; worst forward error {maxfwd:.2e}.",
        "Each dot is a completed matrix/method. Different methods have different coverage. Factor residual uses four random probe columns, solve/forward error use one seeded manufactured solution. Dashed lines are the runner's 1e-10 residual thresholds, not mixed-precision acceptance criteria. Small residual can coexist with larger forward error on a sensitive problem; conditioning was not measured here.")

    # 9. Coverage and experimental limits are visible, not hidden in footnotes.
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),layout="constrained")
    axes[0].bar(METHODS,[counts[m] for m in METHODS],color=[COLORS[m] for m in METHODS]);axes[0].axhline(len(names),color="gray",ls="--");axes[0].set_ylim(0,len(names)+2);axes[0].set_title("Completed comparisons, by method");axes[0].tick_params(axis="x",rotation=40)
    spreads=[]
    for n in names:
        for m in groups[n]:
            p=asset(root,n,m+"-timings.csv")
            if p and groups[n][m]["status"]=="OK":
                with p.open() as f: ts=[float(r["complete_factor_s"]) for r in csv.DictReader(f)]
                if ts and min(ts)>0:spreads.append((max(ts)/min(ts),n,m))
    spreads.sort(reverse=True)
    display=spreads[:10]
    axes[1].barh([short(n)+" / "+m for _,n,m in display],[v for v,_,_ in display],color="#64748b");axes[1].invert_yaxis();axes[1].set_xlabel("Maximum repetition time / minimum");axes[1].set_title("Largest within-case timing spreads")
    add(fig,"9. Coverage and timing variability limit how strongly we rank methods",
        f"AMD, RCM, SCOTCH and CHOLMOD cover all {len(names)} matrices. The 22 old sTiles rows need a direction-corrected rerun; missing entries are not numerical failures.",
        f"All four validated non-baseline candidates completed on {len(allcore)} matrices. Timeouts make later methods more likely to be missing; natural/random were omitted on resumed cases. sTiles rows are excluded for a separate adapter bug, not numerical failure. Three repetitions are too few for strong uncertainty estimates. Raw max/min timing spread is not a confidence interval. Source hashes and run metadata remain in the appendix.")

    # Comprehensive appendix: every matrix, in priority order after main findings.
    priority=["inla_graph_animal1","bodyy4","bodyy5","bodyy6","inla_graph_83o4NNNo"]
    order=[n for n in priority if n in groups]+[n for n in names if n not in priority]
    dossiers=[]
    for n in order:
        best=winners(groups,n,"factor_median_s")[0]
        smallest=winners(groups,n,"nnz_L")
        ms=groups[n][best]["factor_median_s"]*1000
        fig,axes=plt.subplots(1,3,figsize=(11,3.7),layout="constrained")
        original=asset(root,n,"original.svg")
        if original:
            axes[0].imshow(occupancy(original),cmap=ListedColormap(["white","#164e63","#e87924"]),vmin=0,vmax=2,interpolation="nearest");axes[0].set_axis_off();axes[0].set_title("Original input")
        pattern(axes[1],root,groups,n,best,"input");pattern(axes[2],root,groups,n,best,"factor")
        takeaway=f"Fastest recorded median: {best} ({ms:,.3f} ms). Smallest numerical factor: {', '.join(smallest)}. Coverage: {len(groups[n])}/7 candidate rows."
        add(fig,"Appendix · "+short(n),takeaway,
            f"n = {int(groups[n][best]['n']):,}; input nnz = {int(groups[n][best]['nnz_A']):,}. Only measured candidates are ranked. The before/after order shown is the median-time winner, not automatically the storage winner. Tiny timing differences should not be treated as stable rankings.",detail=True)
        fields=["method","status","nnz_L","factor_median_s","solve_min_s","setup_s","bandwidth","factor_residual","solve_residual","forward_error"]
        cells=['<table><tr>'+''.join('<th>'+html.escape(f)+'</th>' for f in fields)+'</tr>']
        for m in METHODS:
            r=groups[n].get(m)
            cells.append('<tr>'+''.join('<td>'+html.escape(m if f=="method" else ("not measured" if r is None else (f"{r[f]:.5g}" if isinstance(r.get(f),float) else str(r.get(f,"")))))+'</td>' for f in fields)+'</tr>')
        cells.append('</table>')
        block=appendix.pop()
        dossiers.append(f'<details class="matrix"><summary>{html.escape(short(n))} · {html.escape(takeaway)}</summary>'+block+''.join(cells)+'<details><summary>Saved run metadata</summary><pre>'+html.escape(metadata[n])+'</pre></details></details>')
    pdf.close()

    caveats = [
        "The timed operation is CPU CHOLMOD with identity internal permutation after explicitly applying A[p,p]. It includes fresh symbolic and numeric factorization; it does not reuse symbolic analysis. This is not native sTiles, dense NextLA, or GPU performance.",
        "Metadata for these matrix reports records Julia 1.12.6, Sapphire Rapids CPU class, one BLAS thread, STILES_CORES=14, seed 20261105, depth 3 and three repetitions. STILES_CORES controls sTiles preprocessing, not the CHOLMOD factor thread count. Reports were accumulated across resumed allocations; exact node identity and contemporaneous load are not retained per case.",
        "Setup scopes differ: CHOLMOD's extraction includes an entire factorization, while current sTiles analysis includes tile layout. Setup is excluded from the factor-time figures, so these are not end-to-end speedups. Per-method solve_min_s and setup_s remain in the tables and CSV.",
        "SCOTCH direction was checked against its inverse mapping. A native symbolic audit subsequently established that the current sTiles export must be inverted for A[p,p]. The old adapter did not invert it, so those rows are DIRECTION_INVALID and excluded from every ranking/plot. The legacy ordering(...).iperm path is separate. Both p and invperm(p) can pass solve checks; that was not an adequate semantic test.",
        "Only current metrics.csv rows are counted. Stale pictures from interrupted attempts are ignored for missing methods. Timeouts are at the whole-matrix process level; the last printed method may not have completed. Raw status.csv has repeated history entries, not a single final numerical status.",
        "The SVG input plots contain aggregated occupancy only; there is no magnitude information to recover. Dense-looking pixels do not imply a dense matrix. Orange factor pixels indicate newly created nonzeros in a bin and may also contain original entries. Recursion overlays use the SymmMixedPrec storage split rule, not a full kernel execution trace."
    ]
    doc='''<!doctype html><html lang="en"><meta charset="utf-8"><title>Ordering study — consolidated analysis</title>
    <style>body{font:17px/1.6 system-ui;color:#172d3a;max-width:1250px;margin:40px auto;padding:0 25px}h1{font-size:38px;line-height:1.15}h2{margin-top:45px}h3{line-height:1.3}figure{margin:42px 0 60px}figure img{width:100%;height:auto}figcaption strong{display:block;border-left:5px solid #0891b2;background:#edf7f8;padding:14px;font-size:18px}figcaption p,.muted{color:#536572;font-size:14px}li{margin-bottom:13px}.summary{padding:22px 28px;background:#f1f7fa;border-radius:12px}details{margin:15px 0;padding:12px;border:1px solid #d6e1e7;border-radius:8px}summary{cursor:pointer;font-weight:600}table{display:block;overflow:auto;border-collapse:collapse;font-size:12px}td,th{padding:7px;border-bottom:1px solid #ddd;text-align:left;white-space:nowrap}pre{white-space:pre-wrap;font-size:12px}a{color:#096c87}nav a{margin-right:18px}@media print{body{max-width:none;font-size:11px}figure{break-inside:avoid}details{display:block}nav,button{display:none}}</style>
    <h1>Reordering sparse Cholesky<br>Overall findings and visual evidence</h1>
    <p class="muted">ORCD experiment · 25 matrices · CPU Float64 · 3 measured repetitions · animal2 not included</p>
    <nav><a href="#takeaways">Overall takeaways</a><a href="#evidence">Ranked figures</a><a href="#appendix">Every matrix</a><a href="#limits">Methods and limits</a></nav>
    <h2 id="takeaways">Overall takeaways and summary</h2><div class="summary"><ol>'''
    doc+=''.join('<li>'+html.escape(p)+'</li>' for p in summary)+'</ol></div>'
    doc+='<h2>Context: what sTiles selects, and what good means</h2>'
    doc+=''.join('<h3>'+html.escape(title)+'</h3><p>'+html.escape(paragraph)+'</p>' for title,paragraph in context)
    doc+='<p>Selector source: <a href="https://arxiv.org/html/2609.29765v1#S3">Abdul Fattah et al., Section 3: Ordering and Symbolic Factorization</a>. API source: <a href="https://github.com/esmail-abdulfattah/sTiles/blob/main/julia/src/factor.jl">Julia analyze/permutation implementation</a>. Native-direction audit: Figure 5; the small graph test is independent of the real-matrix measurements.</p>'
    doc+='<p>The figures below are ordered by relevance to choosing the next experiments, rather than by matrix name. Each figure has its takeaway in bold immediately underneath. The appendix contains every matrix and every saved candidate metric.</p><h2 id="evidence">Figures, most important first</h2>'
    doc+=''.join(sections)
    doc+='<h2 id="appendix">All 25 matrices</h2><p>Expand a matrix for its before/after figure, all candidate metrics and saved run metadata.</p><button onclick="document.querySelectorAll(\'details.matrix\').forEach(d=>d.open=true)">Expand all matrices</button>'
    doc+=''.join(dossiers)
    doc+='<h2 id="limits">Methods, limits and provenance</h2><ol>'+''.join('<li>'+html.escape(p)+'</li>' for p in caveats)+'</ol>'
    doc+='<p>Local evidence: per-matrix metrics.csv, *-timings.csv, *-blocks.csv and SVG patterns from the supplied sweep folder. Input hashes are included in each matrix\'s saved metadata. The accompanying source-manifest.json fingerprints every metrics CSV used in this report.</p></html>'
    (out/"ordering-summary.html").write_text(doc,encoding="utf-8")
    md="# Overall takeaways and summary\n\n"+'\n\n'.join(f"{i}. {p}" for i,p in enumerate(summary,1))
    md+='\n\n# Context: sTiles and a good reordering\n\n'+'\n\n'.join(f"## {t}\n\n{p}" for t,p in context)
    md+='\n\nSource: [sTiles paper, Section 3](https://arxiv.org/html/2609.29765v1#S3).\n\n# Figures in reading order\n'
    for f in manifest:
        md+=f"\n## {f['title']}\n\n![{f['title']}]({f['file']})\n\n**Takeaway: {f['takeaway']}**\n\n{f['caption']}\n"
    md+='\n# Methods and limits\n\n'+'\n\n'.join(caveats)
    (out/"ordering-summary.md").write_text(md,encoding="utf-8")
    (out/"source-manifest.json").write_text(json.dumps({"metrics_sources":provenance,"figures":manifest,"metadata":metadata},indent=2),encoding="utf-8")
    fields=list(rows[0])
    with (out/"all-metrics.csv").open("w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    print(json.dumps({"matrices":len(names),"completed":len(good),"coverage":dict(counts),"median_time_wins":dict(speedwins),"figures":len(manifest),"html":str(out/"ordering-summary.html"),"pdf":str(out/"ordering-summary.pdf")},indent=2))


if __name__ == "__main__":
    main()
