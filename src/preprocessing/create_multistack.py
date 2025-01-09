import numpy as np
import tifffile as tiff
import os
from tqdm import tqdm


def create_multistack_tiff(input_path, output_path, output_name):
    files = [
        f
        for f in os.listdir(input_path)
        if (f.endswith(".tiff") or f.endswith("tif")) and not f.startswith(".")
    ]
    channel_names = [f.split(".")[0] for f in files]
    with tiff.TiffFile(os.path.join(input_path, files[0])) as tif:
        data = tif.asarray()
        shape = data.shape
        dtype = data.dtype

    stack = np.zeros((len(files), *shape), dtype=dtype)
    for i, file in enumerate(tqdm(files, desc="Stacking images")):
        with tiff.TiffFile(os.path.join(input_path, file)) as tif:
            stack[i] = tif.asarray()
    metadata = {"axes": "CYX", "Channel": {"Name": channel_names}}

    with tiff.TiffWriter(os.path.join(output_path, f"{output_name}.ome.tif")) as tif:
        tif.write(stack, metadata=metadata, photometric="minisblack")