import argparse
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from pathlib import Path
from utils.utils import *
from utils.models import VGGEncoder, Decoder

from torchvision.utils import save_image

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

    parser.add_argument('--final_size', type=int, default=256,
                        help='Size of final image')

    parser.add_argument("--content_size", type=int, default=512, help="Final size of the image")

    parser.add_argument("--style_size", type=int, default=512, help="Size of the style image")

    parser.add_argument("--crop", action="store_true", help="Whether to crop the image to final size")

    parser.add_argument(
    "--batch_size",
    type=int,
    default=4,
    help="Batch size"
    )

    parser.add_argument(
    "--lr",
    type=float,
    default=1e-4,
    help="Learning rate"
    )

    parser.add_argument(
    "--lr_decay",
    type=float,
    default=0.01,
    help="Learning rate decay"
    )

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

    content_transform = get_transform(args.content_size, args.crop, args.final_size)
    style_transform = get_transform(args.style_size, args.crop, args.final_size)

    content_dataset = ImageFolderDataset(args.content_dir, content_transform)
    style_dataset = ImageFolderDataset(args.style_dir, style_transform)

    content_loader = DataLoader(content_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True,drop_last=True)
    style_loader = DataLoader(style_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True,drop_last=True)

    print('Number of batches in content dataset: ', len(content_loader))
    print('Number of batches in style dataset: ', len(style_loader))
    
    encoder = VGGEncoder(args.vgg).to(device)
    decoder = Decoder().to(device)

    optimizer = optim.Adam(decoder.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.LambdaLR(
        optimizer, 
        lr_lambda = lambda epoch: 1.0/ (1.0 + args.lr_decay* epoch)
    )

print("Starting training...")
if __name__ == "__main__":
    main()
