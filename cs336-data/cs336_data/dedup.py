from collections import defaultdict
from pathlib import Path
from tqdm.auto import tqdm


def exact_line_dedup(path_list, out_dir, log_path="data/exact_line_dups.log"):
    hash_freq = defaultdict(int)
    for path in tqdm(path_list, "first pass"):
        with open(path, "r") as f:
            for line in f:
                line_hash = hash(line.strip())
                hash_freq[line_hash] += 1

    Path(out_dir).mkdir(parents=True, exist_ok=True)

    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w") as log_f:
        for in_path in tqdm(path_list, "second pass"):
            name = Path(in_path).name
            out_path = f"{out_dir}/{name}"
            with open(in_path, "r") as in_f, open(out_path, "w") as out_f:
                for line in in_f:
                    line_hash = hash(line.strip())
                    line_freq = hash_freq[line_hash]
                    if line_freq == 1:
                        out_f.write(line)
                    else:
                        log_f.write(f"{line_freq}: {line}")
