#!/usr/bin/env python3
import argparse
import pathlib

import semantic_inference
import ultralytics


def main():
    parser = argparse.ArgumentParser(description="Downloading model weights")
    parser.add_argument("models", nargs="*")
    parser.add_argument("-d", "--directory", default=None)
    args = parser.parse_args()

    target_path = args.directory or semantic_inference.models_path()
    target_path = pathlib.Path(target_path).expanduser().absolute()
    target_path.mkdir(exist_ok=True, parents=True)
    for model in args.models:
        output_path = target_path / f"{model}"
        if output_path.exists():
            continue

        print(f"Downloading {model} to {output_path}")
        ultralytics.utils.downloads.attempt_download_asset(str(output_path))


if __name__ == "__main__":
    main()
