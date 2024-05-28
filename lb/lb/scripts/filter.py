import argparse
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import multiprocessing as mp

import fasttext
import mmh3
from tqdm.auto import tqdm
import torch


def filter_setup(dup_file, model_file):
    dups = torch.load(dup_file)

    model = fasttext.load_model(model_file)

    return dups, model


def filter_core(dups, model, in_file, out_file):
    with open(in_file, "r") as in_f, open(out_file, "w") as out_f:
        buf = ""
        with tqdm(total=Path(in_file).stat().st_size) as pbar:
            for line in in_f:
                pbar.update(len(line.encode()))
                if line == "<|endoftext|>\n":
                    input_text = buf.replace("\n", " ")
                    label, prob = model.predict(input_text)
                    if label[0] == "__label__eng_Latn" and prob[0] > 0.9:
                        out_f.write(buf)
                        out_f.write(line)
                        buf = ""
                else:
                    if line == "\n":
                        continue
                    h = mmh3.hash(line)
                    if h in dups:
                        continue
                    buf += line


def worker_filter(args):
    dups, model = filter_setup(args.dup_file, args.model_file)
    filter_core(dups, model, args.in_file, args.out_file)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dup_file", type=str, required=True)
    parser.add_argument("--model_file", type=str, required=True)
    parser.add_argument("--in_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--max_workers", type=int, default=os.cpu_count())
    args = parser.parse_args()

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    mp.set_start_method("forkserver")

    model_path = hf_hub_download(
        repo_id="facebook/fasttext-language-identification",
        filename="model.bin",
    )

    in_files = list(Path(args.in_dir).iterdir())
    with ProcessPoolExecutor() as executor:
        futures = []
        for in_file in tqdm(in_files, "submit"):
            out_file = f"{args.out_dir}/{in_file.name}"
            future = executor.submit(filter, args.dup_file, in_file, out_file)
            futures.append(future)

        for future in tqdm(futures, "result"):
            future.result()


if __name__ == "__main__":
    main()
