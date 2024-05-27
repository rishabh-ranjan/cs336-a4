import argparse
import json
from pathlib import Path

from tqdm.auto import tqdm
from xopen import xopen


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--paloma_dir", type=str, required=True)
    parser.add_argument("--text_dir", type=str, required=True)
    args = parser.parse_args()

    Path(args.text_dir).mkdir(parents=True, exist_ok=True)

    paloma_file_list = list(Path(args.paloma_dir).iterdir())
    count = 0
    for paloma_file in tqdm(paloma_file_list, "paloma"):
        with xopen(paloma_file, "r") as in_f:
            for line in in_f:
                obj = json.loads(line)
                text = obj["text"]
                text_file = f"{args.text_dir}/{count}.txt"
                count += 1
                with open(text_file, "w") as out_f:
                    out_f.write(text)


if __name__ == "__main__":
    main()
