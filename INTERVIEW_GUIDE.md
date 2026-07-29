# Interview guide

## Thirty-second summary

I built an end-to-end credit-risk classification project using a reproducible
synthetic dataset of 10,000 applicants and eight financial features. I compared
standardized logistic regression with a cross-validated random forest. Logistic
regression achieved the better held-out ROC-AUC at 0.854, while the
class-balanced forest had higher recall at the default threshold. I then used
permutation importance and training-only threshold selection to connect model
performance to a cost-sensitive lending scenario.

## Decisions to be ready to explain

### Why use a stratified split?

Only 17.86% of applicants default. Stratification keeps approximately that
class balance in both the training and test sets.

### Why is accuracy insufficient?

A classifier that predicts no default for every applicant would achieve 82.14%
accuracy while detecting zero defaults. ROC-AUC, average precision, recall, and
the confusion matrix reveal performance that accuracy hides.

### Why scale logistic regression but not random forest?

Logistic regression optimization and coefficient comparison benefit from
similar feature scales. Tree splits depend on order and thresholds, so scaling
does not materially change a random forest.

### Why did logistic regression outperform the forest on ROC-AUC?

The synthetic risk process contains strong smooth, additive relationships.
Logistic regression matches that structure well. The forest captured
thresholds and interactions and improved recall through balanced class weights,
but it did not rank applicants as well overall.

### What does ROC-AUC mean here?

It estimates how often the model ranks a randomly chosen defaulter as riskier
than a randomly chosen non-defaulter across all possible thresholds.

### Why use permutation importance?

Impurity-based random-forest importance can favor continuous variables and
split importance across correlated predictors. Permutation importance measures
the loss in held-out ROC-AUC after shuffling a feature.

### How was leakage prevented?

The test set was separated before modeling. Scaling was fitted inside a
training pipeline. Random-forest parameters and business thresholds were
selected using training-only cross-validation or out-of-fold predictions.

### Why change the probability threshold?

The standard 0.50 cutoff assumes false positives and false negatives have equal
consequences. In lending, missing a likely default can be more expensive. Under
an illustrative 5:1 cost ratio, training-only selection lowered the logistic
threshold to 0.16 and raised held-out recall from 43.1% to 79.3%.

## Limitations to volunteer

- The dataset is synthetic, so the metrics cannot be generalized to real
  borrowers.
- The project is not a deployable lending system.
- No time dimension or economic regime change is modeled.
- Protected attributes are absent, so fairness cannot be evaluated.
- The cost ratio is illustrative rather than derived from actual loan losses.

## Strong next steps

- Replace synthetic observations with governed historical loan data.
- Use an out-of-time validation set.
- Evaluate calibration and expected monetary loss.
- Audit subgroup error rates and proxy discrimination.
- Add drift monitoring and scheduled model review.

