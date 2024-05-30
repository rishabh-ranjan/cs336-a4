from pathlib import Path

import fasttext
import mmh3
import nltk
from tqdm.auto import tqdm

hash_count = None


def is_dup(line_bytes):

    global hash_count

    if hash_count is None:
        with open("/dev/shm/cc/hash_count.bin", "rb") as in_f:
            hash_count = in_f.read()

    line_hash = mmh3.hash(line_bytes)
    return hash_count[line_hash] == 2


def initializer():
    global english_model, toxic_model, nsfw_model

    english_model = fasttext.load_model("/dev/shm/cc/models/lid.176.bin")
    toxic_model = fasttext.load_model(
        "/dev/shm/cc/models/jigsaw_fasttext_bigrams_hatespeech_final.bin"
    )
    nsfw_model = fasttext.load_model(
        "/dev/shm/cc/models/jigsaw_fasttext_bigrams_nsfw_final.bin"
    )


def is_toxic(line):
    labels, scores = toxic_model.predict(line.rstrip())
    return labels[0] == "__label__toxic" and scores[0] > 0.0004


def is_nsfw(line):
    labels, scores = nsfw_model.predict(line.rstrip())
    return labels[0] == "__label__nsfw" and scores[0] > 0.00017


def is_english(doc):
    labels, scores = english_model.predict(doc.replace("\n", " "))
    return labels[0] == "__label__en" and scores[0] > 0.5


def at_least_one_alpha(word) -> bool:
    for char in word:
        if char.isalpha():
            return True
    return False


def is_high_quality(doc):
    lines = [line.strip() for line in doc.split("\n")]

    spl_punct = [".", "?", "!", '"']
    count = 0
    for line in lines:
        if line[-1] not in spl_punct:
            count += 1
    if count / len(lines) > 0.5:
        return False

    bullet_lines = [line for line in lines if line.startswith("•")]
    if len(bullet_lines) / len(lines) > 0.9:
        return False

    ellipsis_lines = [line for line in lines if line.endswith("...")]
    if len(ellipsis_lines) / len(lines) > 0.3:
        return False

    words = nltk.word_tokenize(doc)
    if len(words) < 50 or len(words) > 100_000:
        return False

    word_lengths = sorted(len(word) for word in words)
    median_word_length = word_lengths[len(word_lengths) // 2]
    if median_word_length < 3 or median_word_length > 10:
        return False

    alpha_words = [word for word in words if at_least_one_alpha(word)]
    if len(alpha_words) / len(words) < 0.8:
        return False

    for req in ["the", "be", "to", "of", "and", "that", "have", "with"]:
        if doc.count(req) < 2:
            return False

    return True


def worker(in_file, out_file, tqdm_disable=False):
    in_file_size = Path(in_file).stat().st_size
    with open(in_file, "rb") as in_f, open(out_file, "w") as out_f, tqdm(
        total=in_file_size, unit="B", unit_scale=True, disable=tqdm_disable
    ) as pbar:
        buf = ""
        for line_bytes in in_f:
            pbar.update(in_f.tell() - pbar.n)

            if line_bytes == b"<|endoftext|>\n":
                doc = buf
                buf = ""

                if doc == "":
                    continue

                if not is_english(doc):
                    continue

                out_f.write(doc)
                out_f.write("<|endoftext|>\n")
            else:
                if is_dup(line_bytes):
                    continue

                line = line_bytes.decode("utf-8")

                if is_toxic(line):
                    continue

                if is_nsfw(line):
                    continue

                buf += line
