from collections import defaultdict
import json
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
    return hash_count[line_hash] >= 10


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
    return labels[0] == "__label__toxic" and scores[0] > 0.4


def is_nsfw(line):
    labels, scores = nsfw_model.predict(line.rstrip())
    return labels[0] == "__label__nsfw" and scores[0] > 0.4


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

    spl_punct = [".", "?", "!", '"']
    non_spl_punct_lines = [
        line for line in lines if line == "" or line[-1] not in spl_punct
    ]
    if len(non_spl_punct_lines) / len(lines) > 0.5:
        return False

    return True


def worker(in_file, out_file, tqdm_disable=False):
    reach_count = defaultdict(int)
    reject_count = defaultdict(int)
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

                reach_count["not_english"] += 1
                if not is_english(doc):
                    reject_count["not_english"] += 1
                    with open("/dev/shm/cc/inspect/not_english.txt", "a") as f:
                        f.write(doc)
                    continue

                reach_count["not_high_quality"] += 1
                if not is_high_quality(doc):
                    reject_count["not_high_quality"] += 1
                    with open("/dev/shm/cc/inspect/not_high_quality.txt", "a") as f:
                        f.write(doc)
                    continue

                out_f.write(doc)
                out_f.write("<|endoftext|>\n")
            else:
                reach_count["duplicate"] += 1
                if is_dup(line_bytes):
                    reject_count["duplicate"] += 1
                    with open("/dev/shm/cc/inspect/duplicate.txt", "ab") as f:
                        f.write(line_bytes)
                    continue

                line = line_bytes.decode("utf-8")

                reach_count["toxic"] += 1
                if is_toxic(line):
                    reject_count["toxic"] += 1
                    with open("/dev/shm/cc/inspect/toxic.txt", "a") as f:
                        f.write(line)
                    continue

                reach_count["nsfw"] += 1
                if is_nsfw(line):
                    reject_count["nsfw"] += 1
                    with open("/dev/shm/cc/inspect/nsfw.txt", "a") as f:
                        f.write(line)
                    continue

                if not (len(line) >= 2 and line[0].isspace() and not line[1].isspace()):
                    line = line.lstrip()
                line = line.rstrip() + "\n"

                buf += line

    with open("/dev/shm/cc/inspect/counts.json", "w") as f:
        obj = {"reach": reach_count, "reject": reject_count}
        json.dump(obj, f)
