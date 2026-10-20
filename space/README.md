---
title: PRIMO Benchmark
emoji: 🧬
colorFrom: indigo
colorTo: blue
sdk: gradio
sdk_version: 5.50.0
python_version: "3.13"
app_file: app.py
pinned: false
hf_oauth: true
thumbnail: https://huggingface.co/spaces/PRIMOmics/primo-eval/resolve/main/assets/primo-social-preview.png
---

# PRIMO: Patient Representations in Multi-Omics

**A blind benchmark for omics foundation models.**

PRIMO grades how well a model turns a patient's omics data into a useful
**patient embedding**. You select one modality, embed its datasets, and upload
**one** file; a fixed probe scores each hidden task, and the results roll up into a blind,
per-category leaderboard. The datasets are opaque (`d002`, `d003`…) and you never
see the disease, tissue, or target, which leaves you grading the *embedding*
itself with no room for per-task tuning.

A left **rail** navigates six pages: **Boards** (a grid of every board),
**Board** (one leaderboard at a time), **Tasks**, a **Submit** form (sign in with
Hugging Face), **Contribute**, and **Method**. Every board has its own URL,
`?board=oncology-bulk-rna`, and every page too, as `?tab=contribute`, so the
rail links and the open cards are shareable deep links.

Boards also shows **open boards**: greyed-out cards for the omics layers and the
therapeutic areas PRIMO does not cover yet, each linking to Contribute. They are
declared in `boards.py` (`OPEN_BOARDS`) and drop out on their own once the
registry covers that slice. Task categories deliberately get none, because a
category is pinned to a single metric, so an open one would advertise a probe
that does not exist.

🌐 Website: http://primomics.org/ ·
📦 Data: https://huggingface.co/datasets/PRIMOmics/primo

## What's in the data

PRIMO benchmarks any omics modality. It includes **bulk RNA** and **single-cell
RNA** tasks, all with real clinical labels or treatment-induced expression
responses from published cohorts. Tasks are grouped in four therapeutic areas:

- **Immunology & inflammation**: Crohn's disease, ulcerative colitis, celiac
  disease, psoriasis, atopic dermatitis, hidradenitis suppurativa, rheumatoid
  arthritis, systemic sclerosis, Sjögren's disease and multiple sclerosis
- **Oncology**: breast, ovarian, prostate, bladder, lung and rectal cancer,
  melanoma, and acute leukaemias
- **Infectious diseases**: sepsis, COVID-19 and tuberculosis
- **Metabolic diseases**: fatty liver disease (MASH)

A task that fits no area is marked **Other**: today COPD, kidney graft
rejection, graft-versus-host disease and lung fibrosis. It has no area board,
but it is still ranked on its modality and category boards.

Perturbation-response tasks cover adalimumab transfer between inflammatory skin
diseases, mouse-to-human dupilumab transfer, and paired pre/post treatment
cohorts in COPD (inhaled corticosteroids), prostate (enzalutamide), rectal
(galunisertib) and breast cancer (neoadjuvant chemotherapy).

## Submission format

One file, one row per (`dataset_id`, `sample_id`), spanning one modality:

- **CSV / TSV / Parquet**: a `dataset_id` column, a `sample_id` column, and one
  numeric column per embedding dimension. Embedding dim may differ per dataset
  (pad short datasets with blank columns; blank/NaN padding columns are dropped
  per dataset).
- **NPZ**: `dataset_ids`, `sample_ids`, and a 2-D `embeddings` array.

The valid `dataset_id`s and how to download each dataset's `expression.h5ad` are
listed in the public `datasets.yaml` manifest. Alignment is by join, so row
order does not matter; every labelled sample of a task must be present with no
NaN/inf, or that task is skipped.

Bulk H5AD files have one row per required submission sample. Single-cell H5AD
files have one sparse raw-count row per cell; opaque cell ids are in
`obs_names`, and the only public cell metadata is `obs["sample_id"]`, which maps
each cell to its opaque collection sample. Submissions remain sample-level: emit
exactly one embedding for every unique `sample_id`, not one embedding per cell.

## How it works

Each dataset is embedded once and scored on every hidden task defined for it.
Learned embedding rows are scaled to unit length; gene-space baselines keep
their row scale. Then, per task: standardise on the outer training fold, fit
logistic regression (classification) or ridge regression with the penalty
chosen by subject-grouped inner cross-validation, and predict each held-out
fold. Ridge penalties use an exact spectral search over the fixed grid; when
regression tasks have identical samples and fold assignments, their matrix
factorizations are shared while each target still selects its own penalty. We
calculate the native metric per fold and average it over frozen,
subject-grouped partitions: **AUROC**
(classification) or **Pearson r** (regression). Binary tasks with fewer than 30
subjects use leave-pair-out instead: each positive-negative pair of subjects is
held out once, the probe is fit on all other subjects, and the AUROC is the
share of held-out sample pairs ranked correctly. A few tasks instead use a
fixed train/test split: the probe is fit once on the training portion and
scored on the held-out portion, with the same subject-grouped penalty
selection.

Tasks are grouped into **four families**, each reported in its own native
metric. Two families are never merged into one column:

- **Treatment outcome**: response to anti-TNF therapy (**AUROC**)
- **Clinical scores**: disease-severity regression (**Pearson r**)
- **Endotype**: molecular-subtype classification (**AUROC**)
- **Perturbation response**: hidden post-minus-pre DEG decoding (**centered
  Spearman**), where 1 is perfect, 0 is uninformative, and -1 is an inverse gene
  ranking.

A leaderboard shows one column per family. A board holding more than one family
also shows **Mean**, the average of those columns. It is what orders the rows,
but it does mix AUROC, Pearson and centered Spearman, so treat it as a tie-break and
compare models on the family columns.

## Boards

Results are shown as **boards**, self-contained leaderboards over a slice of the
registry: the whole modality, one therapeutic area, one task family. A board
ranks only the models that covered **all of its tasks**, so a submission that
skipped Infectious diseases is still ranked on Oncology. Modality is where we draw
the line: an area or category board never spans two modalities, because an AUROC
on bulk RNA and an AUROC on single-cell are not measuring the same thing.

Partial and failed submissions still get feedback, and their scores always appear
in each board's **per-task** table even when they are not ranked.

## Public tasks

The tasks listed in `public_tasks/manifest.yaml` of the data repo have
downloadable labels. They are tagged **Public** in the task tables and are
scored like every other task.

The Boards, Board and Tasks pages carry an **All tasks / Private tasks** switch.
The private view, `&view=private`, leaves the public tasks out: boards are
re-ranked on their private tasks only, and every link keeps the view. The home
cards stay the full list of boards, so a board whose tasks are all public says
so instead of disappearing. All tasks is the default.

## Make a submission

`quickstart.py` is the shortest path: it downloads one modality and embeds its
datasets
(log2(CPM+1) → PCA for bulk; per-cell log2(CP10K+1) → sample mean → PCA for
single-cell) and writes the file the Submit tab wants. Swap its `embed`
function for your encoder and nothing else changes. `example_submission.csv`
shows the expected shape in four lines.

```bash
pip install anndata scikit-learn pandas pyyaml huggingface_hub
python quickstart.py --modality bulk-rna --out submission.parquet
```

## Run the scorer locally

```bash
pip install -r requirements.txt
export HF_TOKEN=...   # read access to the PRIMO datasets
python evaluator.py --modality bulk-rna --submission my_embeddings.parquet
```

Long local runs can be resumed task by task with `--checkpoint-dir PATH`.
Checkpoints are namespaced by the submission contents, representation,
reference features and scoring configuration, and are invalidated when private
task inputs change.

`--reference PATH` scores the `+ HVG` variant: the submission rows (after their
usual normalisation) are joined by sample with a gene-space file of the same
layout, and every concatenated row is L2-normalised.

## Baselines

`task_results.csv` carries an `is_baseline` flag. Reference submissions we
produce ourselves are published with it set, rendered as `name (baseline)`, and
**ranked in place**. The published `HVG-1200-genes` entry is the **HVG-1200
log-normalized linear baseline**: bulk samples use log2(CPM+1), while single-cell
profiles use per-cell log2(CP10K+1) followed by sample-level mean pooling. It
keeps 1,200 highest-variance genes and the resulting row scale rather than
receiving the L2 row normalisation applied to learned embeddings, then uses
train-fold standardisation and the regularisation-tuned linear probe. Gene
selection is label-blind but fit over every sample of a dataset, so this is an
information-rich transductive reference rather than a naïve raw-count baseline
or an expected lower bound.

## `+ HVG` entries

Every Space submission is scored a second time concatenated with the
`HVG-1200-genes` features and stored as `<model> + HVG`, with the parent's
owner, credits and `submission_id`. The features are kept privately at
`baselines/HVG-1200-genes/submission.parquet` in `PRIMOmics/primo-results`.
When the reference changes, the `+ HVG` entry of every live archived submission
is scored again, so each one follows the current reference.

## Space configuration

- **`hf_oauth: true`** (set above) turns on the Submit tab's *Sign in with
  Hugging Face* button; submitting requires a logged-in HF account.
- Set an **`HF_TOKEN`** Space secret (fine-grained) with: **read** on
  `PRIMOmics/primo` (the public `datasets.yaml` manifest) and
  `PRIMOmics/primo-labels` (the private `tasks.yaml` registry +
  `<task_id>/labels.csv` and perturbation `<task_id>/targets.npz`), and **read + write** on
  `PRIMOmics/primo-results` (the persisted leaderboard).
- Results persist as one normalized `task_results.csv` (`model_name, task_id,
  score, repeat_scores, diagnostics, submitted_at, is_baseline, hf_username`) in the results dataset; the
  leaderboard is recomputed from it by joining the registry, so it survives
  Space restarts.
- A board keeps the **latest result for each `(model name, is_baseline, task)`**,
  so submissions accumulate: a later one fills in the tasks it covers and leaves
  the rest standing, and resubmitting a task replaces just that result. Because a
  task belongs to exactly one modality, one name can hold bulk and single-cell
  results at once without either displacing the other.
- A model name is owned by the account that first submitted it: `hf_username`
  locks it, and Submit refuses a name somebody else holds. It is never rendered
  anywhere: it works as a lock rather than as a credit. Names colliding with a
  per-task column (`Task`, `Family`, `Area`, `Modality`, `Metric`, `Best`) are
  refused too, since model names become column headers, and so are names
  containing the `+ HVG` tag.
- The boards show a **Submissions count** per model — the most times any one of
  its task results was resubmitted — so a task tuned against by repeated
  resubmission is visible. Nothing is ever blocked.
- Submitter contact metadata (HF username, email, paper / model links, notes)
  persists to a separate `submissions.csv` in the same **private** results
  dataset, and never reaches the public leaderboard.

## Moving to another Hugging Face org

The three dataset repos are derived from one constant, `ORG` in `evaluator.py`.
The rest of the org name is spelled out and has to be changed by hand:

- `PUBLIC_REPO` in `quickstart.py` (standalone, so it cannot import `ORG`)
- `SOCIAL_PREVIEW_URL` in `app.py`
- the links in this file, including the YAML `thumbnail:` above, and in
  `pages/*.md`
- outside this folder: `DATA_REPO` in `src/primomics/data.py`, `SPACE` in
  `src/primomics/submit.py`, the links of the top-level `README.md`, and
  the deploy command in `CONTRIBUTING.md`

The theme follows the PRIMO charter: Funnel Display for headings, Funnel Sans for
everything else, Scienta Navy `#080F5F` / PRIMO Cyan `#16B3C0` on Paper
`#F3F8F8`. `colorFrom`/`colorTo` above stay `indigo`/`blue` because Hugging Face
only accepts eight named colours and none of them is cyan.
