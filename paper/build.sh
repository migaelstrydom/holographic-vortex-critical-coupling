#!/bin/bash
# Build paper/main.pdf: pdflatex, bibtex, then pdflatex until the labels
# settle (at most five more passes). Run from anywhere.
set -e
cd "$(dirname "$0")"
latex() { pdflatex -interaction=nonstopmode -halt-on-error main.tex > /dev/null; }
latex
# bibtex exits 1 on warnings (e.g. an empty field) and 2 or more on errors;
# only errors stop the build.
status=0
bibtex main > /dev/null || status=$?
if [ "$status" -ge 2 ]; then
    echo "bibtex failed (exit $status); see main.blg" >&2
    exit "$status"
fi
for pass in 1 2 3 4 5; do
    latex
    if ! grep -qE 'Rerun to get|Label\(s\) may have changed|Rerun LaTeX' main.log; then
        break
    fi
    if [ "$pass" -eq 5 ]; then
        echo "labels still changing after five passes" >&2
        exit 1
    fi
done
grep -E 'Warning: (Citation|Reference)|undefined|Overfull|Missing' main.log | sort | uniq -c | sort -rn | head -20 || true
echo "pdflatex passes after bibtex: $pass"
echo "pages: $(grep -o 'Output written on main.pdf ([0-9]* pages' main.log | grep -o '[0-9]* pages')"
