import numpy as np
import tifffile as tiff
import os
from tqdm import tqdm
import argparse

def create_multistack_tiff(input_path, output_path, output_name, sort):
    if sort:
        files = sorted(
            [
                f
                for f in os.listdir(input_path)
                if (f.endswith(".tiff") or f.endswith("tif")) and not (f.startswith(".") or "segmentation" in f.lower())
            ]
        )
    else:
        files = [
            f
            for f in os.listdir(input_path)
            if (f.endswith(".tiff") or f.endswith("tif")) and not (f.startswith(".") or "segmentation" in f.lower())
        ]
    if files[0].endswith(".tiff"):
        channel_names = [f.split(".tiff")[0] for f in files]
    else:
        channel_names = [f.split(".tif")[0] for f in files]
    with tiff.TiffFile(os.path.join(input_path, files[0])) as tif:
        data = tif.asarray()
        shape = data.shape
        dtype = data.dtype

    stack = np.zeros((len(files), *shape), dtype=dtype)
    for i, file in enumerate(tqdm(files, desc="Stacking images")):
        with tiff.TiffFile(os.path.join(input_path, file)) as tif:
            stack[i] = tif.asarray()
    metadata = {"axes": "CYX", "Channel": {"Name": channel_names}}

    with tiff.TiffWriter(os.path.join(output_path, f"{output_name}.ome.tif"), bigtiff=True) as tif:
        tif.write(stack, metadata=metadata, photometric="minisblack", )


def main():
    parser = argparse.ArgumentParser(description="Create a multistack tiff file")
    parser.add_argument(
        "--input_path",
        type=str,
        help="Path to the folder containing the single channel tiff files",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        help="Path to the output folder for the multistack tiff file"
    )
    parser.add_argument(
        "--output_name",
        type=str,
        help="Name of the output file. Example: 'multistack_image'. The file format will be automatically added"
    )
    parser.add_argument(
        "--sort",
        type=bool,
        default=False,
        help="Boolean, Sort the files in the input directory before stacking them. Default: False"
    )
    args = parser.parse_args()
    create_multistack_tiff(args.input_path, args.output_path, args.output_name, args.sort)

if __name__ == "__main__":
    main()
    