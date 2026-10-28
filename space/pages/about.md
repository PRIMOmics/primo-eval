## What PRIMO measures

**Can a foundation model turn a patient's omics data into an embedding that
predicts meaningful outcomes?**

You embed each patient. We fit a fixed probe on top and check whether that
embedding predicts a real clinical outcome. You never see the labels, so you
cannot tune to them, and when a score comes out high the credit belongs to the
*embedding*.

Plenty of benchmarks already score reconstruction. Far fewer ask whether the
representation is any use in the clinic, and that is the gap PRIMO tries to fill.

## How a submission is scored

1. **You send embeddings, never predictions**: one vector per baseline sample.
2. **We fit the probe**: ridge for scalar regression, logistic for
   classification, or multi-output ridge regression for perturbation
   responses. Learned embedding rows are scaled to unit length first, since an
   embedding dimension has no unit; gene-space baselines keep their log-normalized row
   scale. Feature standardisation is fit on each outer training fold, and probe
   hyperparameters are selected with subject-grouped inner cross-validation.
3. **Scores always hold out whole subjects.** Most tasks use subject-grouped
   cross-validation: we average the fold-level metric over frozen partitions,
   three folds ten times over.
   Binary tasks with fewer than 30 subjects use leave-pair-out: each
   positive-negative pair of subjects is held out in turn, the probe is fit on
   everyone else, and the AUROC is the share of held-out sample pairs ranked
   correctly. These tasks have one score, so no ±.
   Transfer tasks use a fixed split instead: train on one source cohort, test
   on the target cohort. Perturbation tasks decode hidden post-minus-pre
   expression responses.
4. **Each task family keeps its own metric.** AUROC, Pearson r and centered
   Spearman never share a category column.

The probe is identical for everyone, which is what makes the numbers comparable:
what changes between two rows is the representation behind them. The rows marked
`(baseline)` are our own submissions. In particular, `HVG-1200-genes` is the
**HVG-1200 log-normalized linear baseline**: bulk samples use log2(CPM+1), while
single-cell profiles use per-cell log2(CP10K+1) followed by sample-level mean
pooling. We retain the 1,200 genes with highest dataset-wide variance, apply
train-fold standardisation, and fit the same tuned linear probe. Gene selection
never reads a label, but it is fit over every sample of a dataset. This is an
information-rich transductive reference, not a naïve raw-count baseline or an
expected lower bound for a foundation model.

Every submission also gets a second row, **`<model> + HVG`**: the same
embedding concatenated with the `HVG-1200-genes` features of each sample. The
embedding is scaled to unit length, the 1,200 log-normalized genes are appended,
and the concatenated row is scaled to unit length again before the same probe.
It shows whether a model adds information that expression alone does not carry.
These rows are shown by default; set **+ HVG models** to **Hide** to hide them,
which re-ranks the board and refits Elo without them.

Elo orders the rows. Each task family keeps its own column, in its own metric,
and no column averages across metrics.

**Elo** is pinned to our HVG-1200 log-normalized linear baseline, which sits at
exactly 1000. It turns every paired repeat into a win, draw, or loss and
discards the score margin, so a tiny win counts like a large one. A difference
of only a few Elo points is effectively a tie, not evidence that one
representation is meaningfully superior. Use mean rank, the family columns, and
per-task scores to interpret close rows. +400 Elo is a 10-to-1 expected win
rate. The scale does not move when a new model is submitted, so ratings stay
comparable over time, but only within one board, since each board is fitted on
its own tasks.

**Subm.** (submissions count) is the most times any *single* task result of
that model was resubmitted, not how many files were uploaded. A model that
covered twenty tasks across twenty submissions reads 1, because each task was
submitted once; a model that sent the same task five times reads 5. It shows
n/a for our baselines and the `+ HVG` rows, which we republish ourselves. A
published result can be corrected, so a 2 or 3 is ordinary; a high number on
one task is worth a second look.

## How the results are organised

A **board** is a self-contained leaderboard over a slice of the benchmark: the
whole modality, one therapeutic area, one task family, or an area and a family
together. Pick it with the pills of the **Leaderboard** page.

A board ranks only the models that covered **all of its tasks**. Skipping
Infectious diseases therefore costs you nothing on the Oncology board.

**Boards never cross modalities.** An AUROC on bulk RNA and an AUROC on
single-cell are not measuring the same thing, so we keep them apart.

## What stays hidden

The evaluation only stays honest while the labels stay ours, which prevents tuning
the model directly on the labels. We still publish the omics data, to compute
embeddings.

The exception is the tasks tagged **Public**: their labels are downloadable from
the [PRIMO dataset](https://huggingface.co/datasets/PRIMOmics/primo), so a model
may have been trained on them.

| Disclosure | What it covers |
|---|---|
| **Per task** | disease, tissue, therapeutic area, what is predicted, class names, patient count, metric |
| **In aggregate** | the public archives the cohorts sit in |
| **Public tasks** | labels, evaluation folds, post-treatment responses, and the `dataset_id` each is scored on, which can make its source study findable |
| **Never** | study accessions and, for every other task, which cohort is behind which `dataset_id`, labels, pairing identifiers and post-treatment responses |

Every cohort comes from a published study deposited in {repositories}. We name
the archive but never the accession, as publishing the study id behind a
`dataset_id` would make the dataset easy to reconstitute.

## Cite PRIMO

Cite this website and the board version that you used. The **Cite** button, at
the top of every page, gives the BibTeX entry.

Want to add a cohort, propose a task on the cohorts already here, open a new
modality or therapeutic area, or work with us on the methodology? The
**Contribute** tab explains how.

## Links

- 🌐 **Website**: [primomics.org](https://primomics.org/)
- 📦 **Data**: [PRIMOmics/primo](https://huggingface.co/datasets/PRIMOmics/primo)
- 📄 **Paper**: [OpenReview](https://openreview.net/forum?id=v2SA8gHwqo)
- 💬 **Questions**: [karim.elkanbi@scientalab.com](mailto:karim.elkanbi@scientalab.com)
