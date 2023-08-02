import json
import re
from typing import List, Tuple

import joblib
import numpy as np
import pandas as pd
from nltk.corpus import stopwords

from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MaxAbsScaler

stop_words = set(stopwords.words("english"))
p = re.compile(r"[^\w\s]+")
model_path = "model.pkl"
with open('total_cols') as f:
    new = f.read().split('\n')
total_cols = list(filter(lambda x: len(x) > 0, new))

def read_input_data(path: str) -> (np.ndarray, np.ndarray):
    """
    Take in an input file of JSON objects separated by newlines and return X and y for training
    """
    with open(path) as f:
        objs = [json.loads(item) for item in f.read().strip().split("\n")]
        df = pd.json_normalize(objs)

    df = (
        df.drop(columns=["title", "release_date", "movie_box_office_revenue", "feature_length"])
        .rename(columns={c: c.split(".")[0] + "." + df[c].dropna().unique()[0] for c in df.columns if "/m/" in c})
        .drop(columns=[c for c in df.columns if "languages" in c or "countries" in c])
    )
    df.columns = [c.split(".")[-1] for c in df.columns]  # remove `genres.` prefix
    df = df.set_index("plot_summary")
    df[~df.isna()] = 1
    df = df.fillna(0).astype(int).reset_index()
    df.plot_summary = df.plot_summary.apply(prep_plot_text)
    X, y = df[["plot_summary"]].to_numpy(), df.drop(columns=["plot_summary"]).to_numpy()
    return X, y


def train_model(path: str):
    X, y = read_input_data(path)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

    sgd = SGDClassifier(loss='log_loss')
    model = OneVsRestClassifier(sgd)
    pipeline = Pipeline(
        [
            (
                "feature_preparation",
                ColumnTransformer(
                    [("text_feature", TfidfVectorizer(), 0), ("drop_text", "drop", 0)]
                ),
            ),
            ("scaling", MaxAbsScaler()),
            ("classifier", model),
        ]
    )
    pipeline.fit(X_train, y_train)
    joblib.dump(pipeline, "model.pkl")


def prep_plot_text(sentence):
    """
    General cleanup of text. Remove stopwords and punctuation.
    """
    return p.sub("", " ".join(w for w in sentence.strip("\n").split() if w.lower() not in stop_words))


def get_genres_from_plot(plot: str, thresh: float = 0.05. pipeline) -> List[Tuple[str, str]]:
    """
    Main inference function. Takes in an existing sklearn pipeline and returns genres and probabilities
    """
    plot = prep_plot_text(plot)
    plot = np.array([plot]).reshape(1, -1)
    genre_probs = pipeline.predict_proba(plot)[0].tolist()
    filtered_pred = [(genre, prob) for (genre, prob) in zip(total_cols, genre_probs) if prob > thresh]
    res = sorted(filtered_pred, key=lambda x: x[1], reverse=True)
    return [(genre, f"{prob:.0%}") for (genre, prob) in res]


if __name__ == "__main__":
    pipeline = joblib.load(model_path)
    plot = "Five teens walk into a creepy forest to be butchered by an evil psychopath"
    print(get_genres_from_plot(plot, 0.1, pipeline))
    # [('Horror', '65%'), ('Slasher', '30%'), ('B-movie', '19%')]
