import os

import fasttext


def identify_language(model_path: str | os.PathLike, text: str) -> tuple[str, float]:
    model = fasttext.load_model(str(model_path))
    text = text.replace("\n", " ")
    lang_id, prob = model.predict(text)
    lang_id = lang_id[0].replace("__label__", "")
    prob = prob[0]
    return lang_id, prob
