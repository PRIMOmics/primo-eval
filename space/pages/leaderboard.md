**🏆 Ranked**: only models that covered every task of this board the
HVG-1200 log-normalized linear baseline (`HVG-1200-genes`) scored.
**🔍 Per task**: everyone, partial submissions included.

**`+ HVG`** rows are a submitted model concatenated with the `HVG-1200-genes`
features, scored automatically with every submission: they show what the model
adds on top of expression.

One column per task family, each in its own metric. **Mean** is the average of
those columns: it orders the table, but it averages AUROC, Pearson and residual
Spearman, so read it as a tie-break and compare models on the family columns.

**Submissions count** is the most times any *single* task result of that model was
resubmitted — not how many files were uploaded. A model that covered twenty tasks
across twenty submissions reads 1, because each task was submitted once; a model
that sent the same task five times reads 5. Blank for our baselines, which we
republish on a schedule. A published result can be corrected, so a 2 or 3 is
ordinary; a high number on one task is worth a second look.

**Elo** is pinned to our HVG-1200 log-normalized linear baseline, which sits at exactly
1000. It turns every paired repeat into a win, draw, or loss and discards the
score margin, so a tiny win counts like a large one. A difference of only a few
Elo points is effectively a tie, not evidence that one representation is
meaningfully superior. Use mean rank, the family columns, and per-task scores to
interpret close rows. +400 Elo is a 10-to-1 expected win rate. The scale does not
move when a new model is submitted, so ratings stay comparable over time — but
only within one board, since each board is fitted on its own tasks.
