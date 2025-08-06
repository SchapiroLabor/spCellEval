import os
import numpy as np
import tifffile
import argparse
from skimage.io import imread

#parse arguments from command line
parser = argparse.ArgumentParser(description='Max project all images in a folder')
parser.add_argument('-i', '--input', dest='input', required=True, help='Input directory to search.')
parser.add_argument("-o", "--output", dest="output", action="store", required=False, help="Path to output directory. Required unless --overwrite is set to True")
parser.add_argument("-n", "--nuclear-channels", dest="nuc_channels", nargs="+", action="store", required=True, help="Channels to be used for max projection for nucleus")
parser.add_argument("-m", "--membrane-channels", dest="mem_channels", nargs="+", action="store", required=True, help="Channels to be used for max projection for membrane")
parser.add_argument("--overwrite", dest="overwrite", required=False, default= False ,help="Overwrite input file with maxprojection")
args = parser.parse_args()

args.nuc_channels = [int(channel) for channel in args.nuc_channels]
args.mem_channels = [int(channel) for channel in args.mem_channels]

def max_projection(inputfolder, outputfolder, nuc_channels, mem_channels, overwrite):
    for filename in os.listdir(inputfolder):
        img_path = os.path.join(inputfolder, filename)
        #read image
        img = imread(img_path)
        #create output image
        #reshape image to put channels in first dimension
        #check position of channels
        if img.shape[2] < img.shape[0] and img.shape[2] < img.shape[1]:
            #channels are in last dimension
            img = np.transpose(img, (2, 0, 1))
        output_img = np.zeros((2, img.shape[1], img.shape[2]), dtype=img.dtype)
        ### max project nuclear channels
        output_img[0] = np.max(img[nuc_channels], axis=0)
        ### max project membrane channels
        output_img[1] = np.max(img[mem_channels], axis=0)
        ### save the output image
        #create output folder if non existent
        if not os.path.exists(outputfolder):
            os.makedirs(outputfolder)
        #check if output folder is same as input folder
        if overwrite:
            output_img_path = os.path.join(inputfolder, f"{filename}")
            print(f"WARNING:Input image will be overwritten with max projection")
        elif not overwrite and inputfolder == outputfolder:
            output_img_path = os.path.join(outputfolder, f"maxproject_{filename}")
            print(f"WARNING:Output folder same as input folder. Output image will be saved with prefix 'maxproject_'")
        else:
            output_img_path = os.path.join(outputfolder, f"{filename}")
        tifffile.imwrite(output_img_path, output_img)

# main function
def main():
    max_projection(args.input, args.output, args.nuc_channels, args.mem_channels, args.overwrite)

#main
if __name__ == '__main__':
    main()

#example usage: python3 max_projection_simple.py -i data/multistack_tiffs -o data/max_projections --nuclear-channels 39 --membrane-channels 5 7 11 23 27 34