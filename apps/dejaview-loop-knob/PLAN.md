# Plan — dejaview-loop-knob

Planned with the planning-and-task-breakdown skill (the task referenced
`skills/project-planning/` in-repo; that path does not exist on `main`, so the
closest available planning skill was used). Written before the measured run.

## Goal

A clear, pre-registered, CPU-only experiment on the central knob of
DVLT / "Déjà View" (arXiv:2605.30215): the loop count **K** as an
inference-time compute/quality dial. One hypothesis, named metrics, numeric
bars frozen before the run, seeds 0/1/2, results table + plots, breakpoints,
honest PASS/PARTIAL/FAIL, screenshot + video of the running viewer, PR to main.

Definition of done: PR open against main containing the app under
`apps/dejaview-loop-knob/`, a Pages card, a `tracking/seen-bookmarks.json`
entry, committed artifacts, and a status judged only against the
pre-registered bars in `experiments/exp1_cpu_loop_knob/config.yaml`.

## Constraints

- CPU only (4 vCPU), fp32 (`Accelerator(cpu=True)`), no GPU, no paid compute.
- No X access at all; every claim verified from the arXiv abstract, the
  nv-tlabs/dvlt repository, and the Hugging Face model card.
- Weights not committed; downloaded from HF at run time (NVIDIA licence,
  non-commercial research/evaluation only). Licence texts copied per-app.
- Only `apps/dejaview-loop-knob/`, the Pages card/index, and
  `tracking/seen-bookmarks.json` are touched.

## Milestones and tasks

1. **Source verification + environment** — clone nv-tlabs/dvlt (pin commit),
   venv with torch 2.5.1 CPU, install dvlt without heavy extras.
   Accept: one `model.predict` returns cameras/depths on CPU. DONE (smoke.json).
2. **Data** — procedural multi-view scenes with exact GT cameras + depth via a
   numpy rasterizer (`src/make_scenes.py`); 4 scenes x 20 candidate views at
   252^2. Accept: previews visually coherent, depth/pose conventions verified
   by a one-off disclosed smoke at the single mid-grid K=12. DONE.
3. **Pre-registration** — freeze `config.yaml` thresholds + this plan + README
   hypothesis in a commit BEFORE the measured run. Accept: commit precedes
   raw.json; config_sha256 recorded in raw.json.
4. **Measured run** — `src/run_experiment.py`: K main grid {8,10,12,14,16} and
   breakpoint grid {1,2,4,6,20,24,32} x 4 scenes x seeds 0/1/2; plus 2-view,
   low-res (140), and 12-view breakpoints. Accept: raw.json complete.
5. **Analysis** — `src/analyze.py`: results.md/csv tables, quality-vs-K and
   latency-vs-K plots, status.json judged against the frozen bars.
6. **Viewer + artifacts** — static `viewer/index.html` with a K knob
   (per-K predicted poses/depth vs GT), screenshot PNG + MP4 video under
   `artifacts/`. Accept: both artifacts exist and show the real results.
7. **Integration** — Pages card in `docs/index.html` + assets, tracking entry,
   final README numbers, licences. Accept: only allowed paths touched.
8. **PR** — against main with results table, status, screenshot, video.

Dependencies: 3 blocks 4; 4 blocks 5; 5 blocks 6; 6 blocks 7, 8.
