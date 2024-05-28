from concurrent.futures import ProcessPoolExecutor
import gzip
import multiprocessing as mp
from pathlib import Path
import os
import shutil
from tqdm.auto import tqdm


def compress(in_file, out_file):
    with open(in_file, "rb") as in_f, gzip.open(out_file, "wb") as out_f:
        shutil.copyfileobj(in_f, out_f)


def decompress(in_file, out_file):
    with gzip.open(in_file, "rb") as in_f, open(out_file, "wb") as out_f:
        shutil.copyfileobj(in_f, out_f)


def concurrent_compress(in_dir, out_dir, max_workers=None):
    if max_workers is None:
        max_workers = os.cpu_count()

    Path(out_dir).mkdir(parents=True, exist_ok=True)

    in_files = sorted(Path(in_dir).iterdir())

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for in_file in tqdm(in_files, "submit"):
            name = in_file.name
            out_file = f"{out_dir}/{name}.gz"
            future = executor.submit(compress, in_file, out_file)
            futures.append(future)

        for future in tqdm(futures, "result"):
            future.result()


def concurrent_decompress(in_dir, out_dir, max_workers=None):
    if max_workers is None:
        max_workers = os.cpu_count()

    Path(out_dir).mkdir(parents=True, exist_ok=True)

    in_files = sorted(Path(in_dir).iterdir())

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for in_file in tqdm(in_files, "submit"):
            name = in_file.stem
            out_file = f"{out_dir}/{name}"
            future = executor.submit(decompress, in_file, out_file)
            futures.append(future)

        for future in tqdm(futures, "result"):
            future.result()
