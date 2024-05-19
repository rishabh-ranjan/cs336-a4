import os
from pathlib import Path

from fastwarc.warc import ArchiveIterator, WarcRecordType
from fastwarc.stream_io import GZipStream
from resiliparse.extract.html2text import extract_plain_text
from resiliparse.parse.encoding import detect_encoding


def extract_text_from_html_bytes(html_bytes: bytes) -> str | None:
    encoding = detect_encoding(html_bytes)
    html_str = html_bytes.decode(encoding)
    text = extract_plain_text(html_str)
    return text


def extract_html_from_warc_gz(
    warc_gz_path: str | os.PathLike, out_dir: str | os.PathLike
) -> None:
    Path(out_dir).mkdir(parents=True)
    count = 0
    with open(warc_gz_path, "rb") as gz:
        with GZipStream(gz) as stream:
            for record in ArchiveIterator(stream, record_types=WarcRecordType.response):
                if record.http_content_type != "text/html":
                    continue
                html_bytes = record.reader.read()
                count += 1
                with open(f"{out_dir}/{count}.html", "wb") as f:
                    f.write(html_bytes)
