from resiliparse.extract.html2text import extract_plain_text
from resiliparse.parse.encoding import detect_encoding


def extract_text_from_html_bytes(html_bytes: bytes) -> str | None:
    encoding = detect_encoding(html_bytes)
    html_str = html_bytes.decode(encoding)
    text = extract_plain_text(html_str)
    return text
