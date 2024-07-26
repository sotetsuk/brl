import sys
from urllib.request import urlopen


def _download(url, filename):
    try:
        print(f"Downloading from {url} ...", file=sys.stderr)
        data = urlopen(url).read()
        with open(filename, mode="wb") as f:
            f.write(data)
    except Exception as e:
        print(f"Failed to downalod the data from {url}", file=sys.stderr)
        print(e, file=sys.stderr)
        sys.exit(1)

URL = "https://huggingface.co/datasets/sotetsuk/dds_dataset/resolve/main/dds_results_100M.npy"
_download(URL, "dds_results_100M.npy")
