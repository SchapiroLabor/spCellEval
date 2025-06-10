import os
import pandas as pd
import argparse

def construct_csv(image_dir, mask_dir, output_dir):
    image_dirs = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if (f.endswith('.tif') or f.endswith('.tiff'))]
    mask_dirs = [os.path.join(mask_dir, f) for f in os.listdir(mask_dir) if (f.endswith('.tif') or f.endswith('.tiff'))]

    df = pd.DataFrame({
        'image_path': image_dirs,
        'mask_path': mask_dirs
    })

    df.to_csv(os.path.join(output_dir, "batch_processing.csv"), index=False)

def main():
    parser = argparse.ArgumentParser(description="Construct a CSV file from image and mask directories.")
    parser.add_argument("--image_dir", type=str, required=True, help="Path to the directory containing images.")
    parser.add_argument("--mask_dir", type=str, required=True, help="Path to the directory containing masks.")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to the output directory.")
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    construct_csv(args.image_dir, args.mask_dir, args.output_dir)

if __name__ == "__main__":
    main()
