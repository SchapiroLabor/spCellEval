import os
import pandas as pd
import argparse

def construct_csv(image_dir, mask_dir, output_dir):
    image_dirs = []
    mask_dirs = []
    for image in os.listdir(image_dir):
        if image.endswith('.tif') or image.endswith('.tiff'):    
            image_dirs.append(os.path.join(image_dir, image))
            mask_dirs.append(os.path.join(mask_dir, image))
    if len(image_dirs) != len(mask_dirs):
        raise ValueError("The number of images and masks do not match. Please check the directories.")

    df = pd.DataFrame({
        'image_path': image_dirs,
        'mask_path': mask_dirs
    })

    df.to_csv(os.path.join(output_dir, "batch_processing.csv"), index=False)

def main():
    parser = argparse.ArgumentParser(description="Construct a CSV file from image and mask directories. Care this only works if the images and masks have the same names.")
    parser.add_argument("--image_dir", type=str, required=True, help="Path to the directory containing images.")
    parser.add_argument("--mask_dir", type=str, required=True, help="Path to the directory containing masks.")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to the output directory.")
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    construct_csv(args.image_dir, args.mask_dir, args.output_dir)

if __name__ == "__main__":
    main()
