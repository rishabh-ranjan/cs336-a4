import os

import fasttext

from .fast import preprocess_for_fasttext


def classify(model_path: str | os.PathLike, text: str) -> tuple[str, float]:
    model = fasttext.load_model(str(model_path))
    text = text.replace("\n", " ")
    label, prob = model.predict(text)
    label = label[0].replace("__label__", "")
    prob = prob[0]
    return label, prob


def classify_quality(model_path: str | os.PathLike, text: str) -> tuple[str, float]:
    model = fasttext.load_model(str(model_path))
    text = preprocess_for_fasttext(text)
    label, prob = model.predict(text)
    label = label[0].replace("__label__", "")
    prob = prob[0]
    return label, prob
