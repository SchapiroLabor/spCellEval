"""Download deepcell-types training data to a custom destination."""

import argparse
from pathlib import Path
import deepcell_types.utils._auth as auth


def main():
    parser = argparse.ArgumentParser(
        description="Download deepcell-types training data to a specified directory."
    )
    parser.add_argument(
        "--dest",
        type=Path,
        required=True,
        help="Destination directory for the downloaded data (e.g. /scratch/user/deepcell).",
    )
    args = parser.parse_args()

    dest = args.dest.resolve()
    dest.mkdir(parents=True, exist_ok=True)

    # Redirect download location before fetch_data resolves the path
    auth._asset_location = dest

    auth.fetch_data("data/deepcell-types/deepcell-types-data.tar.gz", cache_subdir="data")

    print(f"Done. Data saved to: {dest / 'data'}")


if __name__ == "__main__":
    main()
