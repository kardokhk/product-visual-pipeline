#!/bin/bash
# Build the case study + template guide PDF from approved markdown.
# Usage: code/11_doc_pdf.sh <out.pdf> <md> [md ...]
set -euo pipefail
OUT=$1; shift
TMP=$(mktemp -d)
CSS=$TMP/print.css
cat > "$CSS" <<'CSS'
@page { size: A4; margin: 2.2cm 2.2cm 2.4cm; @bottom-center { content: counter(page); font: 9pt Helvetica, Arial, sans-serif; } }
body { font: 10.5pt/1.5 "Helvetica Neue", Helvetica, Arial, sans-serif; color: #000; }
h1 { font-size: 20pt; margin: 0 0 4pt; } h2 { font-size: 13pt; margin: 18pt 0 6pt; } h3 { font-size: 11pt; }
table { border-collapse: collapse; width: 100%; margin: 8pt 0; font-size: 9.5pt; }
th, td { border-bottom: 0.5pt solid #999; padding: 4pt 6pt; text-align: left; vertical-align: top; }
blockquote { margin: 6pt 0; padding: 4pt 10pt; border-left: 2pt solid #999; color: #222; }
code { font-size: 9pt; } img { max-width: 100%; } .pagebreak { page-break-before: always; }
CSS
ARGS=()
for i in "$@"; do ARGS+=("$i"); done
pandoc "${ARGS[@]}" -f markdown-smart -t html5 --resource-path=.:drafts:work_samples --standalone --metadata title=" " --css "$CSS" --embed-resources -o "$TMP/doc.html"
# drop the empty title block pandoc inserts
sed -i 's#<header id="title-block-header">.*</header>##' "$TMP/doc.html"
LD_LIBRARY_PATH=${BROWSER_LIBS:-} PLAYWRIGHT_BROWSERS_PATH=${PLAYWRIGHT_BROWSERS_PATH:-} ${BROWSER_PY:-python} - "$TMP/doc.html" "$OUT" <<'PY'
import sys
from pathlib import Path
from playwright.sync_api import sync_playwright
with sync_playwright() as p:
    b = p.chromium.launch(); pg = b.new_page()
    pg.goto(Path(sys.argv[1]).resolve().as_uri())
    pg.pdf(path=sys.argv[2], format="A4", print_background=True, display_header_footer=True,
           header_template="<span></span>",
           footer_template='<div style="font:8pt Helvetica,Arial;width:100%;text-align:center"><span class="pageNumber"></span></div>',
           margin={"top": "2.2cm", "bottom": "2.4cm", "left": "2.2cm", "right": "2.2cm"})
    b.close()
PY
rm -rf "$TMP"
echo "$OUT"
