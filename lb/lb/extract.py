from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

import mmh3
from bs4 import BeautifulSoup
from fastwarc.stream_io import GZipStream
from fastwarc.warc import ArchiveIterator, WarcRecordType
from tqdm.auto import tqdm


def content_not_html(record):
    return record.http_content_type != "text/html"


def html_bytes_to_text(html_bytes):
    soup = BeautifulSoup(html_bytes, "lxml")
    text = soup.get_text()
    return text


def warc_gz_to_text_iter(warc_gz_file):
    with open(warc_gz_file, "rb") as gz:
        with GZipStream(gz) as stream:
            archive_iterator = ArchiveIterator(
                stream,
                record_types=WarcRecordType.response,
                func_filter=content_not_html,
            )
            for record in archive_iterator:
                html_bytes = record.reader.read()
                text = html_bytes_to_text(html_bytes)
                yield text


def warc_gz_to_text_file(warc_gz_file, text_file):
    with open(text_file, "w") as out_f:
        with tqdm(total=100_000) as pbar:
            for text in warc_gz_to_text_iter(warc_gz_file):
                out_f.write(text)
                out_f.write("\n<|endoftext|>")
                pbar.update(1)


def get_line_hash_freq_dict(warc_gz_file):
    freq = defaultdict(int)
    for text in warc_gz_to_text_iter(warc_gz_file):
        for line in text.split("\n"):
            h = mmh3.hash(line)
            freq[h] += 1
    return freq


def get_duplicate_line_hashes(warc_gz_files, max_workers):
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for warc_gz_file in warc_gz_files:
            future = executor.submit(get_line_hash_freq_dict, warc_gz_file)
            futures.append(future)

        global_freq = defaultdict(int)
        for future in tqdm(futures):
            local_freq = future.result()
            for k, v in local_freq.items():
                global_freq[k] += v

    dups = [k for k, v in global_freq.items() if v > 1]
    return dups


def exact_line_dedup(text, duplicate_line_hashes):
    lines = text.split("\n")
    dedup_lines = []
    for line in lines:
        h = mmh3.hash(line)
        if h in duplicate_line_hashes:
            continue
        dedup_lines.append(line)
    dedup_text = "\n".join(dedup_lines)
    return dedup_text
