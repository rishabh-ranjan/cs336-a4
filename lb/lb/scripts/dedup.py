import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
import os
from pathlib import Path

import mmh3
from tqdm.auto import tqdm


def exact_line_dedup(dup_file, in_file, out_file):
    dups = []
    with open(dup_file, "r") as in_f:
        for line in in_f:
            h = int(line)
            dups.append(h)

    dups = set(dups)

    with open(in_file, "r") as in_f, open(out_file, "w") as out_f:
        with tqdm(total=Path(in_file).stat().st_size) as pbar:
            for line in in_f:
                pbar.update(len(line.encode())
                if line == "\n":
                    continue
                if line == "<|endoftext|>\n":
                    out_f.write(line)
                    continue
                h = mmh3.hash(line)
                if h in dups:
                    continue
                out_f.write(line)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dup_file", type=str, required=True)
    parser.add_argument("--in_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--max_workers", type=int, default=os.cpu_count())
    args = parser.parse_args()

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    mp.set_start_method("forkserver")

    in_files = list(Path(args.in_dir).iterdir())
    with ProcessPoolExecutor() as executor:
        futures = []
        for in_file in tqdm(in_files, "submit"):
            out_file = f"{args.out_dir}/{in_file.name}"
            future = executor.submit(exact_line_dedup, args.dup_file, in_file, out_file)
            futures.append(future)

        for future in tqdm(futures, "result"):
            future.result()


if __name__ == "__main__":
    main()
