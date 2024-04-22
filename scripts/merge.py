import argparse

from PIL import Image
import torch
from torchvision import transforms
import os


def main(args):
    os.makedirs(args.output_dir, exist_ok=True)

    for i in range(40):
        img1 = Image.open(os.path.join(args.fslsd, f'{i}.jpg'))
        img2 = Image.open(os.path.join(args.e4s, f'{i}.jpg'))
        img3 = Image.open(os.path.join(args.my_e4s, f'{i}.jpg'))

        img1 = transforms.ToTensor()(img1)
        img2 = transforms.ToTensor()(img2)
        img3 = transforms.ToTensor()(img3)

        main_part = img1[..., :1024 * 2]
        fslsd = img1[..., 1024 * 2:]
        e4s = img2[..., 1024 * 2:]
        my_e4s = img3[..., 1024 * 2:]

        final = torch.cat((main_part, fslsd, e4s, my_e4s), dim=2)

        final = transforms.ToPILImage()(final)

        final.save(os.path.join(args.output_dir, f'{i}.jpg'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, default='all_concat')
    parser.add_argument('--fslsd_dir', type=str, default='fslsd')
    parser.add_argument('--e4s_dir', type=str, default='e4s')
    parser.add_argument('--my_e4s_dir', type=str, default='my_e4s')
    args = parser.parse_args()
    main(args)
