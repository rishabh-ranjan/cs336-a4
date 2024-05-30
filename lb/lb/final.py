from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
from pathlib import Path

import fasttext
import mmh3
import nltk
from tqdm.auto import tqdm


def initializer():
    global hash_count, english_model, toxic_model, nsfw_model

    with open("/dev/shm/cc/hash_count_255.bin", "rb") as in_f:
        hash_count = in_f.read()

    english_model = fasttext.load_model("/dev/shm/cc/models/lid.176.bin")
    toxic_model = fasttext.load_model(
        "/dev/shm/cc/models/jigsaw_fasttext_bigrams_hatespeech_final.bin"
    )
    nsfw_model = fasttext.load_model(
        "/dev/shm/cc/models/jigsaw_fasttext_bigrams_nsfw_final.bin"
    )


def is_dup(line_bytes):
    line_hash = mmh3.hash(line_bytes)
    count = hash_count[line_hash]
    return count >= 5


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

    ellipsis_lines = [
        line for line in lines if line.endswith("...") or line.endswith("…")
    ]
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

    count = 0
    for word in ["the", "be", "to", "of", "that", "have", "with"]:  # removed "and"
        if word in words:
            count += 1
    if count < 2:
        return False

    return True


def worker(in_file, out_file, tqdm_disable=False):
    Path(out_file).parent.mkdir(parents=True, exist_ok=True)
    in_file_size = Path(in_file).stat().st_size
    with open(in_file, "rb") as in_f, open(out_file, "w") as out_f, tqdm(
        total=in_file_size, unit="B", unit_scale=True, disable=tqdm_disable
    ) as pbar:
        buf = ""
        last_line_bytes = None
        for line_bytes in in_f:
            pbar.update(in_f.tell() - pbar.n)

            if line_bytes == b"<|endoftext|>\n":
                doc = buf
                buf = ""
                last_line_bytes = None

                if doc == "":
                    continue

                if not is_english(doc):
                    continue

                if not is_high_quality(doc):
                    continue

                out_f.write(doc)
                out_f.write("<|endoftext|>\n")
            else:
                if is_dup(line_bytes):
                    continue

                if last_line_bytes is not None and last_line_bytes.strip().startswith(
                    line_bytes.strip()
                ):
                    continue
                last_line_bytes = line_bytes

                line = line_bytes.decode("utf-8")

                if is_toxic(line):
                    continue

                if is_nsfw(line):
                    continue

                if not (len(line) >= 2 and line[0].isspace() and not line[1].isspace()):
                    line = line.lstrip()
                line = line.rstrip() + "\n"

                buf += line


def master(in_dir, out_dir, max_workers=None):
    if max_workers is None:
        max_workers = os.cpu_count()

    mp.set_start_method("forkserver")

    in_files = list(Path(in_dir).iterdir())
    with ProcessPoolExecutor(
        max_workers=max_workers, initializer=initializer
    ) as executor:
        futures = []
        for in_file in tqdm(in_files, "submit"):
            out_file = f"{out_dir}/{in_file.name}"
            future = executor.submit(worker, in_file, out_file, True)
            futures.append(future)

        for future in tqdm(futures, "result"):
            future.result()
