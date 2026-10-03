import argparse
import torch
from torch.utils.data import DataLoader
from pathlib import Path
from utils.utils import *

def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--content_dir",
        type=str,
        default=r"C:\Users\bhavy\OneDrive\Desktop\Folders\Code\Some Projects\NST\content_data",
        help="Location of dataset"
    )

    parser.add_argument(
        "--style_dir",
        type=str,
        default=r"C:\Users\bhavy\OneDrive\Desktop\Folders\Code\Some Projects\NST\style_data",
        help="Directory containing style dataset")

    parser.add_argument(
        "--vgg",
        type=str,
        default=r"C:\Users\bhavy\OneDrive\Desktop\Folders\Code\Some Projects\NST\vgg_normalised.pth",
        help="Location of pre-trained VGG model"
    )

    parser.add_argument(
        "--experiment",
        type=str,
        default="experiment1",
        help="Name of experiment to store the results"
    )

    parser.add_argument("--content_size", type=int, default=512, help="Final size of the image")

    parser.add_argument("--style_size", type=int, default=512, help="Size of the style image")

    parser.add_argument("--crop", action="store_true", help="Whether to crop the image to final size")


    return parser.parse_args()

def main():
    args = parse_arguments()

    # Check if GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    save_dir = Path("experiment") / args.experiment
    save_dir.mkdir(parents=True, exist_ok=True)


    with open(save_dir / "args.txt", "w") as args_file:
        for arg, value in vars(args).items():
            args_file.write(f"{arg}: {value}\n")

    content_transform = get_transform(args.content_size, args.crop)
    style_transform = get_transform(args.style_size, args.crop)

    content_dataset = ImageFolderDataset(args.content_dir, content_transform)
    style_dataset = ImageFolderDataset(args.style_dir, style_transform)

    content_loader = DataLoader(content_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True,drop_last=True)
    style_loader = DataLoader(style_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True,drop_last=True)

if __name__ == "__main__":
    main()
