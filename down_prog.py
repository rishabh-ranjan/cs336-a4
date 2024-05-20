import os
import time
import argparse
from tqdm import tqdm


def count_files_in_directory(directory):
    return len(
        [
            name
            for name in os.listdir(directory)
            if os.path.isfile(os.path.join(directory, name))
        ]
    )


def main(directory, total_files=100000, sleep_time=1):
    with tqdm(total=total_files) as pbar:
        while True:
            current_file_count = count_files_in_directory(directory)
            pbar.n = current_file_count
            pbar.refresh()
            if current_file_count >= total_files:
                break
            time.sleep(sleep_time)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Track download progress.")
    parser.add_argument(
        "directory", type=str, help="The directory to monitor for downloaded files"
    )
    args = parser.parse_args()

    main(args.directory)
