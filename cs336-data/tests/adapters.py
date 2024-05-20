#!/usr/bin/env python3
from __future__ import annotations

import os
from typing import Any

import cs336_data.classify
import cs336_data.extract
import cs336_data.gopher
import cs336_data.identify
import cs336_data.mask


def run_extract_text_from_html_bytes(html_bytes: bytes) -> str | None:
    return cs336_data.extract.extract_text_from_html_bytes(html_bytes)


def run_identify_language(text: str) -> tuple[Any, float]:
    # model_path = "/home/shared/lid.176.bin"
    model_path = "/lfs/ampere2/0/ranjanr/cs336-a4/data/lid.176.bin"
    return cs336_data.identify.identify_language(model_path, text)


def run_mask_emails(text: str) -> tuple[str, int]:
    return cs336_data.mask.mask_emails(text)


def run_mask_phone_numbers(text: str) -> tuple[str, int]:
    return cs336_data.mask.mask_phone_numbers(text)


def run_mask_ips(text: str) -> tuple[str, int]:
    return cs336_data.mask.mask_ips(text)


def run_classify_nsfw(text: str) -> tuple[Any, float]:
    # model_path = "/home/shared/jigsaw_fasttext_bigrams_nsfw_final.bin"
    model_path = (
        "/lfs/ampere2/0/ranjanr/cs336-a4/data/jigsaw_fasttext_bigrams_nsfw_final.bin"
    )
    return cs336_data.classify.classify_nsfw(model_path, text)


def run_classify_toxic_speech(text: str) -> tuple[Any, float]:
    # model_path = "/home/shared/jigsaw_fasttext_bigrams_hatespeech_final.bin"
    model_path = "/lfs/ampere2/0/ranjanr/cs336-a4/data/jigsaw_fasttext_bigrams_hatespeech_final.bin"
    return cs336_data.classify.classify_toxic_speech(model_path, text)


def run_classify_quality(text: str) -> tuple[Any, float]:
    raise NotImplementedError


def run_gopher_quality_filter(text: str) -> bool:
    return cs336_data.gopher.gopher_quality_filter(text)


def run_exact_line_deduplication(
    input_files: list[os.PathLike], output_directory: os.PathLike
):
    raise NotImplementedError


def run_minhash_deduplication(
    input_files: list[os.PathLike],
    num_hashes: int,
    num_bands: int,
    ngrams: int,
    jaccard_threshold: float,
    output_directory: os.PathLike,
):
    raise NotImplementedError
