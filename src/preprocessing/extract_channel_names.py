import numpy as np
import tifffile as tiff
import xml.etree.ElementTree as ET
from io import StringIO
import os
import argparse


def extract_channels(tif_path, output_path, metadata_attribute, attribute_substring):
    with tiff.TiffFile(tif_path) as tif:
        md = tif.ome_metadata
        root = ET.parse(StringIO(md)).getroot()
        ns = {'ome': 'http://www.openmicroscopy.org/Schemas/OME/2016-06'}
        path = f'ome:Image/ome:Pixels/ome:{metadata_attribute}'
        channels = root.findall(path, ns)
        channel_names = [c.attrib.get(attribute_substring) for c in channels]
        with open(os.path.join(output_path, 'markers.txt'), 'w', encoding='utf-8') as f:
            f.writelines([str(i) + '\n' for i in channel_names])


def main():
    parser = argparse.ArgumentParser(description="Extract channel names from an OME-TIFF file")
    parser.add_argument(
        "--tif_path",
        type=str,
        help="Path to the OME-TIFF file",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        help="Path to the output folder for the channel names. The output file will be named 'markers.txt'"
    )
    parser.add_argument(
        "--metadata_attribute",
        type=str,
        default='Channel',
        help="Name of the metadata attribute containing the channel names. Default: 'Channel'"
    )
    parser.add_argument(
        "--attribute_substring",
        type=str,
        default='Name',
        help="Substring of the metadata attribute to extract. Default: 'Name'"
    )
    args = parser.parse_args()
    extract_channels(
        args.tif_path,
        args.output_path,
        args.metadata_attribute,
        args.attribute_substring
        )
if __name__ == '__main__':
    main()