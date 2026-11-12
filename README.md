# Product visualisation pipeline

Reproducible workflow for product visuals, before-and-after previews and short promotional videos, with an automatic quality check on every output.

**Live demo:** https://kardokhk.github.io/product-visual-pipeline/

Built as a working demonstration for two fictional brands, HALVIA (botanical skincare) and KIVO (running footwear). Everything in the demo came out of this repository.

## What it does

1. **Master shot.** Text to image from a written brand kit (exact label text, palette, mood). Several seeded candidates per product; one is approved.
2. **Variants by editing.** Scenes, colourways, personalisation and before-and-after transformations are instruction edits of an approved image, so the product's shape, label and materials carry over.
3. **Motion.** Approved stills are animated (5 s, 24 fps) in 16:9 or 9:16, then cut with a crossfade and a branded end card.
4. **Quality gate after every stage.** Each candidate is scored, and a person approves the pick in `selection.yaml` before the next stage uses it.

| Check | Method |
|---|---|
| Prompt match | CLIP ViT-L/14 image-text score |
| Product identity | DINOv2 similarity of the product, located by OWLv2 in master and variant |
| Product colour | CIEDE2000 between median core colours of the product in master and variant |
| Palette distance | area-weighted CIEDE2000 from dominant colours to the brand palette |
| Object count | OWLv2 detections after non-maximum suppression (flags duplicated products) |
| Edge alignment | SSIM of Canny edge maps, before against after |
| Video identity, flicker | DINOv2 frame-to-still similarity; mean frame-to-frame change |

The scores rank and flag; they do not replace review. Known limits found in this run: the count check misses duplicates that overlap (one box around both), and it can flag a strong reflection as a second object.

## Configuration, not code

| File | Holds |
|---|---|
| `code/config/brand.yaml` | product description, label text, palette, colour lock, mood, tagline |
| `code/config/prompts.yaml` | templates with named slots, model, steps, guidance, negative prompt, QC caption |
| `code/config/campaign.yaml` | what to make: template, brand, slot values, seeds, input |

A new product in an existing category is a new brand block and a few campaign lines.

## Running it

```bash
cp local.env.example local.env        # set interpreter, model cache, Slurm account
pip install -r requirements.txt       # plus a CUDA build of torch
bash code/00_download_models.sh       # pinned open-weight checkpoints
./run_all.sh test                     # unit tests
./run_all.sh t2i                      # GPU batch job; then review and edit selection.yaml
./run_all.sh edit                     # GPU batch job; review
./run_all.sh video                    # GPU batch job; review
./run_all.sh assemble                 # boards, promo cuts, reveal video
```

Outputs that already exist are skipped, so interrupted runs resume. One worker runs per GPU. The 20B image models fit a 40 GB A100 by storing transformer weights in float8 and computing in bf16; weights are staged once per node in RAM so that four workers do not each read them from network storage.

## Reproducibility

- `results/<run>/manifest.json`: for every candidate, the prompt as run, negative prompt, seed, steps, guidance, size, model repository and revision, run time and peak GPU memory, and whether it was approved.
- `results/<run>/qc_*.csv`: all scores. `results/<run>/compare_lock*.csv`: the paired template comparison described in the demo.
- `environment.md` and `requirements.txt`: software versions.

## Models

Qwen-Image-2512, Qwen-Image-Edit-2511 and Wan2.2 TI2V-5B for generation; DINOv2 and OWLv2 for scoring (all Apache 2.0); CLIP ViT-L/14 for scoring (MIT).

## Licence

Code: MIT. Generated images and video in `docs/` are demonstration material for fictional brands.
