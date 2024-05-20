import argparse
import requests


def main():
    parser = argparse.ArgumentParser(description="Scrape HTML")
    parser.add_argument("--url", type=str, required=True, help="URL to scrape")
    parser.add_argument(
        "--out_path", type=str, required=True, help="Path to output file"
    )
    args = parser.parse_args()

    head = requests.head(args.url)
    head.raise_for_status()

    # check if content-type is text/html
    if "text/html" not in head.headers["Content-Type"]:
        print("Content-Type is not text/html")
        return

    response = requests.get(args.url)
    response.raise_for_status()

    with open(args.out_path, "w") as f:
        f.write(response.text)


if __name__ == "__main__":
    main()
