import argparse
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import multiprocessing as mp

import fasttext
from huggingface_hub import hf_hub_download
from tqdm.auto import tqdm


def filter_english(in_file, out_file):
    model_path = hf_hub_download(
        repo_id="facebook/fasttext-language-identification",
        filename="model.bin",
    )
    model = fasttext.load_model(model_path)

    with open(in_file, "r") as in_f, open(out_file, "w") as out_f:
        buf = ""
        for line in in_f:
            if line == "<|endoftext|>\n":
                input_text = buf[:100].replace("\n", " ")
                label, prob = model.predict(input_text)
                if label[0] == "__label__eng_Latn" and prob[0] > 0.8:
                    out_f.write(buf)
                    out_f.write(line)
            else:
                buf += line


def main():
    parser = argparse.ArgumentParser()
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
            future = executor.submit(filter_english, in_file, out_file)
            futures.append(future)

        for future in tqdm(futures, "result"):
            future.result()


if __name__ == "__main__":
    main()
