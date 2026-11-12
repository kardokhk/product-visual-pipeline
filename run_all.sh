#!/bin/bash
# End-to-end pipeline. GPU stages are Slurm batch jobs; review gates sit between them.
#   1. t2i     -> inspect results/<run>/t2i + qc_t2i.csv, write results/<run>/selection.yaml
#   2. edit    -> inspect results/<run>/edit + qc_edit.csv, add edit picks to selection.yaml
#   3. video   -> inspect results/<run>/video + qc_video.csv, add video picks
#   4. assembly (CPU, login node): boards, promo cuts, reveal video
set -euo pipefail
RUN=${RUN:-results/2026-11-11_campaign}
source ./local.env
case "${1:-}" in
  t2i|edit|video)
    sbatch --account="$SLURM_ACCOUNT" --partition="$SLURM_PARTITION" --job-name=gen-$1 --qos=default --time=${TIME:-01:00:00} code/slurm_stage.sh "$1" "$RUN" ;;
  assemble)
    "$PY" code/05_boards.py --run-dir "$RUN" --out-dir work_samples
    "$PY" code/04_promo_cut.py --run-dir "$RUN" --out-dir work_samples
    "$PY" code/06_reveal.py --run-dir "$RUN" --out work_samples/07_before_after_reveal.mp4 ;;
  test)
    "$PY" -m pytest -q tests ;;
  *) echo "usage: run_all.sh {test|t2i|edit|video|assemble}"; exit 1 ;;
esac
