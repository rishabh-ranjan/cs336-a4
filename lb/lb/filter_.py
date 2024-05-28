from pathlib import Path
import re
import string

import fasttext
import mmh3
import torch
from tqdm.auto import tqdm

punctuation_pat = re.compile(f"([{re.escape(string.punctuation)}])")
whitespace_pat = re.compile(r"\s+")


def normalize(text):
    text = text.lower()
    text = punctuation_pat.sub(r" \1 ", text)
    text = whitespace_pat.sub(" ", text)
    return text


def core(dups, model, in_file, out_file, tqdm_disable=False):
    with open(in_file, "rb") as in_f, open(out_file, "w") as out_f:
        buf = ""
        in_file_size = Path(in_file).stat().st_size
        with tqdm(
            total=in_file_size, unit="B", unit_scale=True, disable=tqdm_disable
        ) as pbar:
            for line in in_f:
                pbar.update(in_f.tell() - pbar.n)
                line = line.decode("utf-8")
                if line == "<|endoftext|>\n":
                    text = buf[:100].replace("\n", " ")
                    labels, probs = model.predict(text)
                    label = labels[0]
                    prob = probs[0]
                    if label == "__label__en" and prob > 0.8:
                        out_f.write(buf)
                        out_f.write("<|endoftext|>\n")
                    buf = ""
                else:
                    if line == "\n":
                        continue
                    h = mmh3.hash(line)
                    if h in dups:
                        continue
                    buf += line


def worker(dup_file, model_file, in_file, out_file):
    dups = torch.load(dup_file)
    model = fasttext.load_model(model_file)
    Path(out_file).parent.mkdir(parents=True, exist_ok=True)
    core(dups, model, in_file, out_file, tqdm_disable=True)


def master(dup_file, model_file, in_dir, out_dir, max_workers):
    in_files = sorted(Path(in_dir).iterdir())
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for in_file in tqdm(in_files, "submit"):
            name = in_file.name
            out_file = f"{out_dir}/{name}"
            future = executor.submit(worker, dup_file, model_file, in_file, out_file)
            futures.append(future)

        for future in tqdm(futures, "result"):
            future.result()
