from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
from multiprocessing.shared_memory import SharedMemory
from pathlib import Path

import mmh3
import os
from tqdm.auto import tqdm


def worker(in_file, shm_name, tqdm_disable=False):
    shm = SharedMemory(shm_name)
    hash_count = shm.buf

    in_file_size = Path(in_file).stat().st_size
    with open(in_file, "rb") as in_f, tqdm(
        total=in_file_size, unit="B", unit_scale=True, disable=tqdm_disable
    ) as pbar:
        for line in in_f:
            pbar.update(in_f.tell() - pbar.n)

            line_hash = mmh3.hash(line, signed=False)
            old_count = hash_count[line_hash]
            if old_count < 255:
                hash_count[line_hash] = old_count + 1

    shm.close()


def master(in_dir, out_file, max_workers=None):
    if max_workers is None:
        max_workers = os.cpu_count()

    mp.set_start_method("forkserver")

    shm = SharedMemory(create=True, size=2**32)

    in_files = list(Path(in_dir).iterdir())
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for in_file in tqdm(in_files, "submit"):
            future = executor.submit(worker, in_file, shm.name, True)
            futures.append(future)

        for future in tqdm(futures, "result"):
            future.result()

    hash_count = shm.buf
    with open(out_file, "wb") as out_f:
        out_f.write(hash_count)

    shm.close()
    shm.unlink()
