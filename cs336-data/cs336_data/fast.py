from pathlib import Path
import random
import re
import string

from bs4 import BeautifulSoup
import fasttext
from fastwarc.warc import ArchiveIterator, WarcRecordType
from fastwarc.stream_io import GZipStream
from tqdm.auto import tqdm

MIN_CONTENT_LENGTH = 250
MAX_CONTENT_LENGTH = 500_000


def is_not_html(record):
    return record.http_content_type != "text/html"


def warc_gz_to_html_dir(
    warc_gz_path,
    html_dir,
    num_samples=float("inf"),
    min_content_length=-1,
    max_content_length=-1,
    count=0,
):
    """Extract HTML content from .warc.gz file and save to .html files.

    Implements reservoir sampling. If num_samples is set to float('inf'), all
    records are extracted.

    Creates {html_dir}/{i}.html for i = 0, ..., num_samples - 1.

    Use count to reservoir sample across multiple .warc.gz files.
    """
    Path(html_dir).mkdir(parents=True, exist_ok=True)
    with open(warc_gz_path, "rb") as gz:
        with GZipStream(gz) as stream:
            archive_iterator = ArchiveIterator(
                stream,
                record_types=WarcRecordType.response,
                min_content_length=min_content_length,
                max_content_length=max_content_length,
                func_filter=is_not_html,
            )
            for record in archive_iterator:
                count += 1
                if count <= num_samples:
                    # fill the reservoir
                    idx = count - 1
                else:
                    idx = random.randrange(count)
                    if idx >= num_samples:
                        # skip this record
                        continue
                html_bytes = record.reader.read()
                with open(f"{html_dir}/{idx}.html", "wb") as out_f:
                    out_f.write(html_bytes)


def html_bytes_to_text(html_path, text_path=None):
    """Extract text from HTML bytes."""
    with open(html_path, "rb") as in_f:
        html_bytes = in_f.read()
    soup = BeautifulSoup(html_bytes, "lxml")
    text = soup.get_text()
    if text_path is None:
        return text
    else:
        with open(text_path, "w") as out_f:
            out_f.write(text)


punctuation_pat = re.compile(f"([{re.escape(string.punctuation)}])")
whitespace_pat = re.compile(r"\s+")


def preprocess_for_fasttext(text):
    """Preprocess text for fastText classification."""
    text = text.lower()
    text = punctuation_pat.sub(r" \1 ", text)
    text = whitespace_pat.sub(" ", text)
    text = text.strip()
    return text


def html_to_fasttext(html_path):
    """Extract text from HTML bytes and preprocess for fastText classification."""
    text = html_bytes_to_text(html_path)
    text = preprocess_for_fasttext(text)
    return text


def shuffle_balance_and_mix(in_paths, out_path):
    """Shuffle, balance, and mix text files."""
    text_lists = []
    for in_path in in_paths:
        with open(in_path, "r") as in_f:
            text_list = in_f.readlines()
            text_lists.append(text_list)
    min_len = min(len(text_list) for text_list in text_lists)
    text_lists = [text_list[:min_len] for text_list in text_lists]
    out_text_list = []
    for i in range(min_len):
        for text_list in text_lists:
            out_text_list.append(text_list[i])
    random.shuffle(out_text_list)
    with open(out_path, "w") as out_f:
        for text in out_text_list:
            out_f.write(text)


# text classification with fasttext


def split_train_val(in_path, val_frac=0.1):
    """Split text file into train and validation files."""
    with open(in_path, "r") as in_f:
        lines = in_f.readlines()
    random.shuffle(lines)
    val_len = int(val_frac * len(lines))
    val_lines = lines[:val_len]
    train_lines = lines[val_len:]
    in_path_stem = Path(in_path).stem
    parent = Path(in_path).parent
    train_path = f"{parent}/{in_path_stem}.train"
    val_path = f"{parent}/{in_path_stem}.valid"
    with open(train_path, "w") as train_f:
        for line in train_lines:
            train_f.write(line)
    with open(val_path, "w") as val_f:
        for line in val_lines:
            val_f.write(line)
    return train_path, val_path


def train_fasttext(train_path, valid_path, model_path, **kwargs):
    """Train fastText model."""
    model = fasttext.train_supervised(input=train_path, **kwargs)
    model.save_model(model_path)
    num_samples, precision_at_1, recall_at_1 = model.test(valid_path)
    return num_samples, precision_at_1, recall_at_1
