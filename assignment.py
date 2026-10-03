from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.naive_bayes import MultinomialNB

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
DATA_FILES = {
    "train": (
        DATA_DIR / "imdb_train.parquet",
        (
            "https://huggingface.co/datasets/stanfordnlp/imdb/resolve/main/"
            "plain_text/train-00000-of-00001.parquet"
        ),
    ),
    "test": (
        DATA_DIR / "imdb_test.parquet",
        (
            "https://huggingface.co/datasets/stanfordnlp/imdb/resolve/main/"
            "plain_text/test-00000-of-00001.parquet"
        ),
    ),
}
CLASS_NAMES = {0: "negative", 1: "positive"}
CASE_STUDIES = {
    4: (
        "This review is a mixed, audience-qualified appraisal of an action film: "
        "the writer describes enjoyment and recommends it to genre/actor fans, "
        "while qualifying its plot and broader appeal. NB's largest negative "
        "terms are actor names (`damme`, `segal`, `dolph`) whose corpus-level "
        "associations outweigh the scattered positive terms under its "
        "independent-feature likelihood sum. Logistic Regression instead "
        "weights evaluative evidence such as `enjoyed`, `best`, `highly`, and "
        "`fun`, so it predicts positive. The apparent mismatch with the negative "
        "test label is consistent with a mixed review and a star-derived binary "
        "label that a word model cannot infer from qualification alone."
    ),
    22194: (
        "This positive review discusses a Steven Seagal film while saying it "
        "differs from the actor's familiar formula. Several Seagal-related "
        "tokens have learned negative class-conditional associations, with "
        "`seagal` dominating NB's negative evidence. But the review's overall "
        "assessment is favorable: Logistic Regression's positive contributions "
        "from `fun`, `best`, and `well` exceed its negative evidence and it is "
        "substantially more confident. This case shows NB can overgeneralize "
        "topic/person-name correlations, while LR can use discriminative "
        "sentiment cues; neither model fully resolves the scope of the few "
        "negative clauses."
    ),
    23947: (
        "This positive review praises a horror film, but its strongest NB "
        "negative evidence includes `crap` and `avoid`. In context, those words "
        "criticize other contemporary horror movies and advise against a later "
        "sequel, rather than criticize the film being reviewed. Unigram TF-IDF "
        "does not represent which movie a sentiment word refers to, so NB adds "
        "those terms as negative evidence. Logistic Regression is pushed "
        "positive by `great`, `excellent`, and `solid`, which more directly "
        "describe the reviewed film. This is a clear target/scope error caused "
        "by losing phrase context and reference in the bag-of-words features."
    ),
}


def download_dataset() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for split, (path, url) in DATA_FILES.items():
        if not path.exists():
            print(f"Downloading official IMDb {split} split to {path} ...")
            urllib.request.urlretrieve(url, path)


def read_split(split: str) -> tuple[list[str], np.ndarray]:
    path, _ = DATA_FILES[split]
    if not path.is_file():
        raise FileNotFoundError(
            f"IMDb {split} data missing at {path}; run without --skip-download."
        )
    frame = pd.read_parquet(path, columns=["text", "label"])
    labels = frame["label"].astype(int).to_numpy()
    return frame["text"].astype(str).tolist(), labels


def make_vectorizer(vectorizer_type: str) -> CountVectorizer | TfidfVectorizer:
    parameters = {
        "lowercase": True,
        "strip_accents": "unicode",
        "ngram_range": (1, 1),
        "min_df": 2,
        "max_df": 0.95,
    }
    if vectorizer_type == "tfidf":
        return TfidfVectorizer(**parameters, sublinear_tf=True)
    return CountVectorizer(**parameters)


def metrics_for(
    model_name: str,
    feature_name: str,
    truth: np.ndarray,
    predicted: np.ndarray,
) -> dict[str, str | float]:
    return {
        "model": model_name,
        "features": feature_name,
        "accuracy": accuracy_score(truth, predicted),
        "precision_positive": precision_score(truth, predicted, zero_division=0),
        "recall_positive": recall_score(truth, predicted, zero_division=0),
        "f1_positive": f1_score(truth, predicted, zero_division=0),
    }


def top_linear_evidence(
    model: LogisticRegression,
    vectorizer: CountVectorizer | TfidfVectorizer,
    review_vector,
    limit: int = 5,
) -> list[tuple[str, float]]:
    coefficients = model.coef_.ravel()
    feature_names = vectorizer.get_feature_names_out()
    active = review_vector.indices
    values = review_vector.data
    contributions = [(str(feature_names[i]), float(values[j] * coefficients[i]))
                     for j, i in enumerate(active)]
    return sorted(contributions, key=lambda item: abs(item[1]), reverse=True)[:limit]


def top_nb_evidence(
    model: MultinomialNB,
    vectorizer: CountVectorizer | TfidfVectorizer,
    review_vector,
    limit: int = 5,
) -> list[tuple[str, float]]:
    feature_names = vectorizer.get_feature_names_out()
    log_odds = model.feature_log_prob_[1] - model.feature_log_prob_[0]
    active = review_vector.indices
    values = review_vector.data
    contributions = [(str(feature_names[i]), float(values[j] * log_odds[i]))
                     for j, i in enumerate(active)]
    return sorted(contributions, key=lambda item: abs(item[1]), reverse=True)[:limit]


def write_confusion_matrix_plot(matrix: np.ndarray, path: Path) -> None:
    fig, axis = plt.subplots(figsize=(6, 5))
    image = axis.imshow(matrix, cmap="Blues")
    axis.set(
        xticks=[0, 1],
        yticks=[0, 1],
        xticklabels=["negative", "positive"],
        yticklabels=["negative", "positive"],
        xlabel="Predicted label",
        ylabel="Actual label",
        title="Multinomial Naive Bayes (TF-IDF) confusion matrix",
    )
    threshold = matrix.max() / 2
    for row in range(2):
        for column in range(2):
            axis.text(
                column,
                row,
                f"{matrix[row, column]:,}",
                ha="center",
                va="center",
                color="white" if matrix[row, column] > threshold else "black",
                fontsize=13,
            )
    fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def choose_disagreements(
    truth: np.ndarray,
    nb_predictions: np.ndarray,
    nb_probabilities: np.ndarray,
    lr_predictions: np.ndarray,
    lr_probabilities: np.ndarray,
    count: int = 10,
) -> list[int]:
    disagree = np.flatnonzero(nb_predictions != lr_predictions)
    if len(disagree) < count:
        raise RuntimeError(f"Expected at least {count} disagreements; found {len(disagree)}.")
    # Select high-confidence disagreements first while ensuring both directions appear
    # whenever possible, then use test row number for deterministic tie breaking.
    rows: list[int] = []
    for nb_label, lr_label in ((0, 1), (1, 0)):
        group = [
            index
            for index in disagree
            if nb_predictions[index] == nb_label and lr_predictions[index] == lr_label
        ]
        group.sort(
            key=lambda index: (
                -abs(nb_probabilities[index, 1] - lr_probabilities[index, 1]),
                index,
            )
        )
        rows.extend(group[: count // 2])
    if len(rows) < count:
        remainder = [index for index in disagree if index not in set(rows)]
        remainder.sort(
            key=lambda index: (
                -abs(nb_probabilities[index, 1] - lr_probabilities[index, 1]),
                index,
            )
        )
        rows.extend(remainder[: count - len(rows)])
    return rows[:count]


def write_disagreements(
    selected: list[int],
    texts: list[str],
    truth: np.ndarray,
    nb_predictions: np.ndarray,
    nb_probabilities: np.ndarray,
    lr_predictions: np.ndarray,
    lr_probabilities: np.ndarray,
    nb_vectorizer: CountVectorizer | TfidfVectorizer,
    nb_model: MultinomialNB,
    lr_vectorizer: CountVectorizer | TfidfVectorizer,
    lr_model: LogisticRegression,
    output_dir: Path,
) -> None:
    tfidf_rows = []
    details: list[str] = [
        "# Three TF-IDF model disagreement case studies",
        "",
        (
            "Review row numbers refer to zero-based row positions in the official "
            "IMDb test parquet split. Review text is not reproduced; the word "
            "evidence below is based on each review's active TF-IDF features."
        ),
        "",
    ]
    for rank, index in enumerate(selected, start=1):
        nb_label = int(nb_predictions[index])
        lr_label = int(lr_predictions[index])
        record = {
            "review_index": index,
            "actual": CLASS_NAMES[int(truth[index])],
            "naive_bayes_tfidf_prediction": CLASS_NAMES[nb_label],
            "naive_bayes_tfidf_predicted_class_probability": nb_probabilities[index, nb_label],
            "logistic_regression_tfidf_prediction": CLASS_NAMES[lr_label],
            "logistic_regression_tfidf_predicted_class_probability": lr_probabilities[
                index, lr_label
            ],
        }
        nb_words = top_nb_evidence(
            nb_model, nb_vectorizer, nb_vectorizer.transform([texts[index]])
        )
        lr_words = top_linear_evidence(
            lr_model, lr_vectorizer, lr_vectorizer.transform([texts[index]])
        )
        record["nb_top_weighted_evidence"] = "; ".join(
            f"{word}:{weight:+.4f}" for word, weight in nb_words
        )
        record["lr_top_weighted_evidence"] = "; ".join(
            f"{word}:{weight:+.4f}" for word, weight in lr_words
        )
        tfidf_rows.append(record)

        if rank <= 3:
            nb_terms = ", ".join(f"`{word}` ({value:+.4f})" for word, value in nb_words)
            lr_terms = ", ".join(f"`{word}` ({value:+.4f})" for word, value in lr_words)
            case_note = CASE_STUDIES.get(
                index,
                "The classifiers emphasize different active terms. NB sums "
                "class-conditional log-likelihood contributions under feature "
                "independence, while Logistic Regression sums discriminatively "
                "learned TF-IDF weights. Both can miss negation scope, sarcasm, "
                "and mixed sentiment.",
            )
            details.extend(
                [
                    f"## Case {rank} — test row {index}",
                    "",
                    f"- Actual: **{CLASS_NAMES[int(truth[index])]}**",
                    (
                        f"- MultinomialNB TF-IDF: **{CLASS_NAMES[nb_label]}**, "
                        "probability of its predicted class "
                        f"**{nb_probabilities[index, nb_label]:.4f}**."
                    ),
                    (
                        f"- Logistic Regression TF-IDF: **{CLASS_NAMES[lr_label]}**, "
                        "probability of its predicted class "
                        f"**{lr_probabilities[index, lr_label]:.4f}**."
                    ),
                    f"- NB strongest active-feature log-odds contributions: {nb_terms}.",
                    f"- Logistic strongest active-feature contributions: {lr_terms}.",
                    "",
                    f"**Case-specific analysis:** {case_note}",
                    "",
                ]
            )
    pd.DataFrame(tfidf_rows).to_csv(output_dir / "disagreements.csv", index=False)
    (output_dir / "deep_dive_cases.md").write_text("\n".join(details), encoding="utf-8")


def build_report(
    metrics: pd.DataFrame,
    nb_confusion: np.ndarray,
    disagreements: int,
    output_dir: Path,
) -> None:
    table = [
        "| Model | Features | Accuracy | Precision | Recall | F1 |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in metrics.to_dict(orient="records"):
        table.append(
            f"| {row['model']} | {row['features']} | {row['accuracy']:.4f} | "
            f"{row['precision_positive']:.4f} | {row['recall_positive']:.4f} | "
            f"{row['f1_positive']:.4f} |"
        )
    cm_table = [
        "| Actual \\ Predicted | Negative | Positive |",
        "|---|---:|---:|",
        f"| Negative | {nb_confusion[0, 0]:,} | {nb_confusion[0, 1]:,} |",
        f"| Positive | {nb_confusion[1, 0]:,} | {nb_confusion[1, 1]:,} |",
    ]
    nb_tfidf = metrics[
        (metrics["model"] == "Multinomial Naive Bayes")
        & (metrics["features"] == "TF-IDF")
    ].iloc[0]
    lr_tfidf = metrics[
        (metrics["model"] == "Logistic Regression")
        & (metrics["features"] == "TF-IDF")
    ].iloc[0]
    report = [
        "# IMDb Sentiment Classification: Multinomial Naive Bayes vs Logistic Regression",
        "",
        "## Data and features",
        "",
        (
            "Used the official IMDb Large Movie Review Dataset with its 25,000-review "
            "training and 25,000-review test split. TF-IDF matches Q2 exactly: "
            "lowercasing, Unicode accent stripping, unigrams, `min_df=2`, "
            "`max_df=0.95`, and sublinear term frequency. Both vectorizers are fit "
            "on training data only. BoW uses `CountVectorizer` with the same token/"
            "vocabulary filters. Logistic Regression uses `C=1` and `max_iter=2000`."
        ),
        "",
        "## (c)-(e) Test-set model comparison",
        "",
        (
            "Precision, recall, and F1 are for the positive class. The comparison "
            "includes Logistic Regression on both feature types to make the requested "
            "`BoW/TF-IDF` comparison explicit."
        ),
        "",
        *table,
        "",
        "### Multinomial Naive Bayes with TF-IDF confusion matrix",
        "",
        "Rows are actual class; columns are predicted class.",
        "",
        *cm_table,
        "",
        "The color plot is [`confusion_matrix_nb_tfidf.png`](confusion_matrix_nb_tfidf.png).",
        "",
        "### Interpretation",
        "",
        (
            f"On TF-IDF, MultinomialNB reaches F1={nb_tfidf['f1_positive']:.4f} "
            f"and Logistic Regression reaches F1={lr_tfidf['f1_positive']:.4f}. "
            "MultinomialNB applies a generative, conditionally independent "
            "word-feature model; Logistic Regression directly optimizes "
            "discriminative class probabilities. NB can be a strong text baseline, "
            "but feature dependence and TF-IDF's non-count scaling can affect its "
            "probability estimates. BoW counts preserve term frequency, the input "
            "form most naturally aligned with MultinomialNB's multinomial model."
        ),
        "",
        "## (f)-(g) Ten model disagreements and predicted-class probabilities",
        "",
        (
            f"Found **{disagreements}** test reviews where the TF-IDF MultinomialNB "
            "and TF-IDF Logistic Regression predictions differ. Ten stable test-row "
            "IDs, actual labels, predictions, each model's probability for its own "
            "predicted class, and top weighted evidence are in the generated local "
            "`results/disagreements.csv` file. Predictions are compared with the "
            "same TF-IDF representation to isolate classifier differences."
        ),
        "",
        "## (h) Three disagreement cases",
        "",
        (
            "Three disagreement cases are analyzed in `results/deep_dive_cases.md` "
            "using model probabilities and feature contributions. The NB feature "
            "contributions come from active TF-IDF values multiplied by the "
            "positive-vs-negative class log-probability difference. Logistic "
            "Regression contributions use active TF-IDF values multiplied by the "
            "learned coefficient. Case explanations compare the direction and "
            "magnitude of these contributions and discuss independence assumptions, "
            "negation, word order, sarcasm, and mixed sentiment. The full review "
            "text is not reproduced in the public report."
        ),
        "",
    ]
    (output_dir / "assignment_report.md").write_text("\n".join(report), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-download", action="store_true")
    args = parser.parse_args()
    if not args.skip_download:
        download_dataset()
    train_texts, train_labels = read_split("train")
    test_texts, test_labels = read_split("test")
    if len(train_texts) != 25_000 or len(test_texts) != 25_000:
        raise RuntimeError("Expected the official 25,000/25,000 IMDb data split.")

    output_dir = ROOT / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    models: dict[tuple[str, str], object] = {}
    vectorizers: dict[str, CountVectorizer | TfidfVectorizer] = {}
    matrices: dict[str, object] = {}
    metrics_rows = []

    for features in ("bow", "tfidf"):
        vectorizer = make_vectorizer(features)
        train_matrix = vectorizer.fit_transform(train_texts)
        test_matrix = vectorizer.transform(test_texts)
        vectorizers[features] = vectorizer
        matrices[features] = test_matrix
        print(f"{features}: {train_matrix.shape[1]:,} features")
        candidates = [
            ("Multinomial Naive Bayes", MultinomialNB(alpha=1.0)),
            (
                "Logistic Regression",
                LogisticRegression(C=1.0, max_iter=2000, solver="liblinear", random_state=42),
            ),
        ]
        for model_name, model in candidates:
            model.fit(train_matrix, train_labels)
            predictions = model.predict(test_matrix)
            models[(model_name, features)] = model
            display_features = "TF-IDF" if features == "tfidf" else "BoW"
            metrics_rows.append(
                metrics_for(model_name, display_features, test_labels, predictions)
            )
            print(
                f"  {model_name}: accuracy={accuracy_score(test_labels, predictions):.4f}, "
                f"F1={f1_score(test_labels, predictions):.4f}"
            )

    metrics = pd.DataFrame(metrics_rows)
    metrics.to_csv(output_dir / "model_comparison.csv", index=False)
    nb_model = models[("Multinomial Naive Bayes", "tfidf")]
    lr_model = models[("Logistic Regression", "tfidf")]
    nb_tfidf_matrix = matrices["tfidf"]
    nb_predictions = nb_model.predict(nb_tfidf_matrix)
    lr_predictions = lr_model.predict(nb_tfidf_matrix)
    nb_probabilities = nb_model.predict_proba(nb_tfidf_matrix)
    lr_probabilities = lr_model.predict_proba(nb_tfidf_matrix)
    nb_confusion = confusion_matrix(test_labels, nb_predictions, labels=[0, 1])
    pd.DataFrame(
        nb_confusion,
        index=["actual_negative", "actual_positive"],
        columns=["predicted_negative", "predicted_positive"],
    ).to_csv(output_dir / "confusion_matrix_nb_tfidf.csv")
    write_confusion_matrix_plot(nb_confusion, output_dir / "confusion_matrix_nb_tfidf.png")

    selected = choose_disagreements(
        test_labels,
        nb_predictions,
        nb_probabilities,
        lr_predictions,
        lr_probabilities,
    )
    differing_count = int(np.count_nonzero(nb_predictions != lr_predictions))
    write_disagreements(
        selected,
        test_texts,
        test_labels,
        nb_predictions,
        nb_probabilities,
        lr_predictions,
        lr_probabilities,
        vectorizers["tfidf"],
        nb_model,
        vectorizers["tfidf"],
        lr_model,
        output_dir,
    )
    build_report(metrics, nb_confusion, differing_count, output_dir)
    print("\nModel comparison:")
    print(metrics.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"\nTF-IDF NB/LR disagreement count: {differing_count}")
    print(f"Selected test row IDs: {selected}")
    print(f"Results saved under {output_dir}")


if __name__ == "__main__":
    main()
