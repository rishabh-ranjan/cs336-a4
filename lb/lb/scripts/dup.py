import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
import os
from pathlib import Path

import mmh3
from tqdm.auto import tqdm


def get_line_hash_freq_dict(in_file):
    freq = defaultdict(int)
    with open(in_file, "r") as in_f:
        for line in in_f:
            if line == "\n":
                continue
            if line == "<|endoftext|>\n":
                continue
            h = mmh3.hash(line)
            freq[h] += 1
    return freq


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_dir", type=str, required=True)
    parser.add_argument("--out_file", type=str, required=True)
    parser.add_argument("--max_workers", type=int, default=os.cpu_count())
    args = parser.parse_args()

    Path(args.out_file).parent.mkdir(parents=True, exist_ok=True)

    mp.set_start_method("forkserver")

    in_files = list(Path(args.in_dir).iterdir())
    with ProcessPoolExecutor() as executor:
        futures = []
        for in_file in tqdm(in_files, "submit"):
            future = executor.submit(get_line_hash_freq_dict, in_file)
            futures.append(future)

        global_freq = defaultdict(int)
        for future in tqdm(futures, "result"):
            local_freq = future.result()
            for k, v in local_freq.items():
                global_freq[k] += v

        with open(args.out_file, "w") as out_f:
            for k, v in global_freq.items():
                if v == 1:
                    continue
                out_f.write(f"{k}\n")


if __name__ == "__main__":
    main()
