import multiprocessing as mp
from pathlib import Path

import mmh3
import numpy as np
import os
from tqdm.auto import tqdm


def worker(in_file, out_dir):
    pid = os.getpid()
    out_file = f"{out_dir}/{pid}.bin"
    Path(out_file).parent.mkdir(parents=True, exist_ok=True)

    size = 2**32
    np.zeros(size, dtype=np.uint8)

    in_file_size = Path(in_file).stat().st_size
    hash_count = np.memmap(hash_count_file, dtype=np.uint8, mode="r+")
    with (
        open(in_file, "rb") as in_f,
        open(hash_count_file, "r+b") as hash_count_f,
        open(log_file, "ab") as log_f,
    ):
        with tqdm(total=in_file_size, unit="B", unit_scale=True) as pbar:
            hash_count = np.memmap(hash_count_f, dtype=np.uint8, mode="r+")
            for line in in_f:
                pbar.update(in_f.tell() - pbar.n)
                line_hash = mmh3.hash(line, signed=False)
                if hash_count[line_hash] == 1:
                    log_f.write(line)
                hash_count[line_hash] += 1


def init_hash_count(hash_count_file):
    size = 2**32
    hash_count = np.zeros(size, dtype=np.uint8)
    hash_count.tofile(hash_count_file)


def master(in_dir, out_file, max_workers=None):
    if max_workers is None:
        max_workers = os.cpu_count()

    mp.set_start_method("forkserver")

    init_hash_count(hash_count_file)

    in_files = sorted(Path(in_dir).iterdir())
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for in_file in tqdm(in_files, "submit"):
            future = executor.submit(worker, in_file, hash_count_file, log_file)
            futures.append(future)

        for future in tqdm(futures, "result"):
            future.result()
