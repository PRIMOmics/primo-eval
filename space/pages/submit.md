**The fastest way in.** Two files, both in this [Space's repo](https://huggingface.co/spaces/PRIMOmics/primo-eval/blob/main/quickstart.py):

- 📥 [`quickstart.py`](https://huggingface.co/spaces/PRIMOmics/primo-eval/blob/main/quickstart.py):
  downloads the selected modality, embeds it, and writes a valid submission.
  Swap its `embed` function for your model and you are done.
- 📄 [`example_submission.csv`](https://huggingface.co/spaces/PRIMOmics/primo-eval/blob/main/example_submission.csv):
  four lines, fake numbers, the exact shape we expect.

```bash
pip install anndata scikit-learn pandas pyyaml huggingface_hub
python quickstart.py --modality bulk-rna --out submission.parquet
# or: --modality single-cell-rna
```

---

Or do it by hand, in **three steps**:

1. **Choose one modality** → use the selector in the form. The quickstart
   downloads only its datasets from [PRIMOmics/primo](https://huggingface.co/datasets/PRIMOmics/primo).
2. **Embed that modality** → build **one** file: `dataset_id`, `sample_id`, then one column per embedding dim (`e0`, `e1`, …). CSV / TSV / Parquet, or NPZ.
3. **Sign in, fill the form, and hit Evaluate.** Add an institution for group submissions, check **Submitted by the model's authors** when applicable, and provide a paper link to make the model name clickable. A fixed task probe scores each hidden task (AUROC, Pearson or centered Spearman), reported per task category in its native metric.

**Example file**

```
dataset_id,sample_id,e0,e1,e2
d002,S1,0.12,-0.44,0.98
d003,S1,0.31,0.02,-0.15
```

For bulk datasets, each H5AD row is one submission sample. For single-cell
datasets, H5AD rows are cells and `obs["sample_id"]` maps them to opaque
collection samples. Aggregate the cells however your model requires and submit
exactly one embedding per unique `sample_id`; the submission schema is unchanged.

Files containing dataset IDs from another modality are rejected. **Partial
submissions are welcome, and they add up.** Cover fewer datasets and you are still
scored: you get ranked on every **board** whose scored tasks you covered in
full, and your numbers still show up in each board's **per-task** table, so
nothing you send is thrown away.

Results are keyed by **model name and task**. Submit under a name you already
used and it fills in the tasks it covers, leaving the rest of that model's
results standing — so you can build coverage up one submission at a time, and the
same name can hold results for both modalities at once. Submitting a task you
have already covered replaces that one result, which is how a bad run gets fixed.
A name belongs to whichever HF user submitted it first; nobody else can write under it.

**Each submission yields two entries.** We also score your embedding
concatenated with our `HVG-1200-genes` features (your embedding is put on the
unit sphere, then the concatenated row) and rank it as `<your model> + HVG`,
under your name and credits. Scoring therefore takes about twice as long. Names containing `+ HVG` are reserved.

**Your first target is the baselines.** We run our own reference submissions on
log-normalized expression, whole or cut down to its most variable genes, and they
sit on the boards labelled `(baseline)`. `HVG-1200-genes` is the HVG-1200
log-normalized linear baseline, not a raw-count baseline.
