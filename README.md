# IMDb Sentiment: Naive Bayes vs Logistic Regression

Question 1 trains Multinomial Naive Bayes on bag-of-words and TF-IDF features,
then compares both with Logistic Regression on the same representations. It
uses the official 25,000-review training and 25,000-review test split from the
IMDb Large Movie Review Dataset (Maas et al., 2011).

## Run

```powershell
py -m pip install -r requirements.txt
py assignment.py
```

The script downloads the train and test parquet splits on its first run and
saves them under `data/`. It writes reproducible metrics, confusion matrices,
comparison chart, ten disagreement rows, and detailed explanations of three
cases to `results/`. Review text itself is not stored in the generated reports;
disagreement rows are identified by index in the official test split.

## Reproducible experiment

- Uses the same official IMDb training/test split and TF-IDF representation as
  Q2: lowercase, Unicode accent stripping, word unigrams, `min_df=2`,
  `max_df=0.95`, and sublinear TF.
- Fits vocabulary/statistics only on training data; applies the fitted
  vectorizers to the held-out official test data.
- Models: MultinomialNB on TF-IDF; MultinomialNB on BoW; Logistic Regression
  (`C=1`, `max_iter=2000`) on TF-IDF and BoW.
- Reports accuracy, positive-class precision/recall/F1, and the test-set
  confusion matrix (rows are actual negative/positive, columns predicted
  negative/positive).
- Selects ten test reviews where the TF-IDF NB and TF-IDF Logistic Regression
  predictions disagree. It reports each model's probability for its own
  predicted class and examines feature contributions for three cases.

Cases are referenced by stable row number in the official test parquet
ordering and analyzed using weighted words/features, model probabilities, and
labels. The review dataset itself is not committed.

Dataset citation: Maas, A. L. et al. (2011), *Learning Word Vectors for
Sentiment Analysis*, ACL. Dataset source:
[stanfordnlp/imdb](https://huggingface.co/datasets/stanfordnlp/imdb).
