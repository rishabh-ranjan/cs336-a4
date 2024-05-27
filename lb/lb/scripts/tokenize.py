import argparse
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm
from transformers import GPT2Tokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--text_dir", type=str, required=True)
    parser.add_argument("--tokens_file", type=str, required=True)
    args = parser.parse_args()

    Path(args.tokens_file).parent.mkdir(parents=True, exist_ok=True)

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    end_of_text = tokenizer.encode("<|endoftext|>")[0]

    text_file_list = list(Path(args.text_dir).iterdir())

    with open(args.tokens_file, "wb") as out_f:
        for text_file in tqdm(text_file_list, "text_files"):
            with open(text_file, "r") as in_f:
                text = in_f.read()
            tokens = tokenizer.encode(text)
            tokens.append(end_of_text)
            arr = np.array(tokens, dtype=np.uint16)
            out_f.write(arr)


if __name__ == "__main__":
    main()
