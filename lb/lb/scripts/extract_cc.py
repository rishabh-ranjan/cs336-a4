import argparse
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from lb import extract
from tqdm.auto import tqdm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--warc_gz_dir", type=str, required=True)
    parser.add_argument("--text_dir", type=str, required=True)
    parser.add_argument("--max_workers", type=int, default=os.cpu_count())
    args = parser.parse_args()

    warc_gz_files = list(Path(args.warc_gz_dir).iterdir())

    with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
        futures = []
        for warc_gz_file in tqdm(warc_gz_files, "submit"):
            text_file = f"{args.text_dir}/{warc_gz_file.stem}.txt"
            future = executor.submit(
                extract.warc_gz_to_text_file,
                warc_gz_file,
                text_file,
                tqdm_disable=True,
            )
            futures.append(future)

        for future in tqdm(futures, "result"):
            future.result()


if __name__ == "__main__":
    main()
