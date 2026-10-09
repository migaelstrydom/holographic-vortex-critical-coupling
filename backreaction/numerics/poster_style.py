"""Shared matplotlib style for the paper figures (the poster_*.py scripts and
tb_phase_diagram.py).

One place for three things every paper figure needs:

* the font sizes and line widths, so labels print at the same size in every
  figure;
* TrueType (Type 42) font embedding, so no Type-3 fonts end up in the PDFs or in
  the built paper (JHEP/arXiv prefer embedded outline fonts);
* the physical widths the paper prints the figures at.  The figures are made at
  exactly these widths and included at their natural size
  (\\includegraphics{name}, no width option), so the printed font size equals
  the size set here.  JHEP's \\textwidth is 430.20639pt (read from a jheppub
  probe), i.e. 5.953 in; a single-panel figure is 0.6\\textwidth.

`save` writes PNG and PDF reproducibly: the PDF carries no CreationDate, and the
PNG no timestamp, so a rerun on the pinned environment reproduces the committed
files byte for byte.
"""

TEXTWIDTH_IN = 430.20639 / 72.27  # JHEP \textwidth in inches (5.953)
SINGLE_IN = 0.6 * TEXTWIDTH_IN  # single-panel figures (3.572)
DOUBLE_IN = TEXTWIDTH_IN  # two-panel figures

RC = {
    "font.size": 8,
    "axes.labelsize": 9,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "font.family": "sans-serif",
    "mathtext.fontset": "dejavusans",
    # embed TrueType outlines (Type 42), never Type 3
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}


def pyplot():
    """Return matplotlib.pyplot on the Agg backend with the paper style set."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(RC)
    return plt


def save(fig, path_stem, dpi=300, exts=("png", "pdf")):
    """Write <path_stem>.png and <path_stem>.pdf (or `exts`) without timestamps."""
    for ext in exts:
        p = f"{path_stem}.{ext}"
        # no CreationDate in the PDF, so a rerun reproduces the committed file
        meta = {"CreationDate": None} if ext == "pdf" else None
        fig.savefig(p, dpi=dpi, metadata=meta)
        print(f"  saved {p}")
