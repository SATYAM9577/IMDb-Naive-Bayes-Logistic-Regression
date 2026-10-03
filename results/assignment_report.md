# IMDb Sentiment Classification: Multinomial Naive Bayes vs Logistic Regression

## Data and features

Used the official IMDb Large Movie Review Dataset with its 25,000-review training and 25,000-review test split. TF-IDF matches Q2 exactly: lowercasing, Unicode accent stripping, unigrams, `min_df=2`, `max_df=0.95`, and sublinear term frequency. Both vectorizers are fit on training data only. BoW uses `CountVectorizer` with the same token/vocabulary filters. Logistic Regression uses `C=1` and `max_iter=2000`.

## (c)-(e) Test-set model comparison

Precision, recall, and F1 are for the positive class. The comparison includes Logistic Regression on both feature types to make the requested `BoW/TF-IDF` comparison explicit.

| Model | Features | Accuracy | Precision | Recall | F1 |
|---|---|---:|---:|---:|---:|
| Multinomial Naive Bayes | BoW | 0.8139 | 0.8617 | 0.7479 | 0.8008 |
| Logistic Regression | BoW | 0.8653 | 0.8721 | 0.8562 | 0.8640 |
| Multinomial Naive Bayes | TF-IDF | 0.8344 | 0.8766 | 0.7785 | 0.8246 |
| Logistic Regression | TF-IDF | 0.8884 | 0.8878 | 0.8890 | 0.8884 |

### Multinomial Naive Bayes with TF-IDF confusion matrix

Rows are actual class; columns are predicted class.

| Actual \ Predicted | Negative | Positive |
|---|---:|---:|
| Negative | 11,130 | 1,370 |
| Positive | 2,769 | 9,731 |

The color plot is [`confusion_matrix_nb_tfidf.png`](confusion_matrix_nb_tfidf.png).

### Interpretation

On TF-IDF, MultinomialNB reaches F1=0.8246 and Logistic Regression reaches F1=0.8884. MultinomialNB applies a generative, conditionally independent word-feature model; Logistic Regression directly optimizes discriminative class probabilities. NB can be a strong text baseline, but feature dependence and TF-IDF's non-count scaling can affect its probability estimates. BoW counts preserve term frequency, the input form most naturally aligned with MultinomialNB's multinomial model.

## (f)-(g) Ten model disagreements and predicted-class probabilities

Found **3138** test reviews where the TF-IDF MultinomialNB and TF-IDF Logistic Regression predictions differ. Ten stable test-row IDs, actual labels, predictions, each model's probability for its own predicted class, and top weighted evidence are in the generated local `results/disagreements.csv` file. Predictions are compared with the same TF-IDF representation to isolate classifier differences.

## (h) Three disagreement cases

Three disagreement cases are analyzed in `results/deep_dive_cases.md` using model probabilities and feature contributions. The NB feature contributions come from active TF-IDF values multiplied by the positive-vs-negative class log-probability difference. Logistic Regression contributions use active TF-IDF values multiplied by the learned coefficient. Case explanations compare the direction and magnitude of these contributions and discuss independence assumptions, negation, word order, sarcasm, and mixed sentiment. The full review text is not reproduced in the public report.
