from collections import defaultdict
from pathlib import Path


def exact_line_dedup(path_list, out_dir):
    hash_freq = defaultdict(int)
    for path in path_list:
        with open(path, "r") as f:
            for line in f:
                line_hash = hash(line)
                hash_freq[line_hash] += 1

    Path(out_dir).mkdir(parents=True, exist_ok=True)

    for in_path in path_list:
        name = Path(in_path).name
        out_path = f"{out_dir}/{name}"
        with open(in_path, "r") as in_f, open(out_path, "w") as out_f:
            for line in in_f:
                line_hash = hash(line)
                if hash_freq[line_hash] == 1:
                    out_f.write(line)
