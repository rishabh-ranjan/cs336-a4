from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
import os
from pathlib import Path
import tempfile

import numpy as np
import tiktoken
from tqdm.auto import tqdm


def worker(in_file, temp_dir, tqdm_disable=False):
    tokenizer = tiktoken.get_encoding("gpt2")
    tokens = []
    in_file_size = Path(in_file).stat().st_size
    with open(in_file, "rb") as in_f, tqdm(
        total=in_file_size, unit="B", unit_scale=True, disable=tqdm_disable
    ) as pbar:
        buf = b""
        for line_bytes in in_f:
            pbar.update(in_f.tell() - pbar.n)

            if line_bytes == b"<|endoftext|>\n":
                text = buf.decode("utf-8")
                buf = b""
                new_tokens = tokenizer.encode(text, disallowed_special=())
                tokens.extend(new_tokens)
                tokens.append(tokenizer.eot_token)
            else:
                buf += line_bytes

    arr = np.array(tokens, dtype=np.uint16)
    pid = os.getpid()
    out_file = f"{temp_dir}/{pid}.bin"
    with open(out_file, "ab") as out_f:
        out_f.write(arr)


def master(in_dir, out_file, max_workers=None, tqdm_disable=True):
    if max_workers is None:
        max_workers = os.cpu_count()

    try:
        mp.set_start_method("forkserver")
    except RuntimeError:
        # context has already been set
        pass

    in_files = list(Path(in_dir).iterdir())
    with tempfile.TemporaryDirectory() as temp_dir:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for in_file in tqdm(in_files, "submit"):
                future = executor.submit(worker, in_file, temp_dir, tqdm_disable)
                futures.append(future)

            for future in tqdm(futures, "result"):
                future.result()

        temp_files = list(Path(temp_dir).iterdir())
        Path(out_file).parent.mkdir(parents=True, exist_ok=True)
        with open(out_file, "wb") as out_f:
            for temp_file in tqdm(temp_files, "cat"):
                with open(temp_file, "rb") as in_f:
                    buf = in_f.read()
                    out_f.write(buf)
