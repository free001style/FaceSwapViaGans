import argparse
import os
import warnings

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image
from scipy.linalg import sqrtm
from torch.autograd import Variable

from metrics import utils, hopenet
from metrics.arcface import iresnet100
from metrics.detect import detect_landmarks
from metrics.FAN import FAN

warnings.filterwarnings("ignore")

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


class EvalMetric(object):
    def __init__(self):
        self._metric = 0
        self._len = 0

    def update(self, **kwargs):
        pass

    def reset(self):
        self._metric = 0
        self._len = 0

    def get(self):
        return self._metric / self._len


class Identity(EvalMetric):
    def __init__(self):
        super().__init__()
        self.model = iresnet100(fp16=False)
        self.model.load_state_dict(torch.load('pretrained_ckpts/arcface.pt', map_location=device))
        self.model.to(device).eval()

    def update(self, source, swap):
        source = self.model(
            torch.nn.functional.interpolate(source.unsqueeze(0), [112, 112], mode='bilinear', align_corners=False))
        swap = self.model(
            torch.nn.functional.interpolate(swap.unsqueeze(0), [112, 112], mode='bilinear', align_corners=False))
        self._metric += torch.cosine_similarity(source, swap, dim=1)
        self._len += 1


class PoseMetric(EvalMetric):
    def __init__(self):
        super().__init__()
        self.model = hopenet.Hopenet(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], 66)
        self.model.load_state_dict(torch.load('pretrained_ckpts/hopenet_robust_alpha1.pkl', map_location=device))
        self.model.to(device).eval()

    def get_face_points(self, img):
        transformations = transforms.Compose([transforms.Resize(224),
                                              transforms.CenterCrop(224),
                                              transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                   std=[0.229, 0.224, 0.225])])

        idx_tensor = [idx for idx in range(66)]
        idx_tensor = torch.FloatTensor(idx_tensor).to(device)

        img = transformations(img)

        images = Variable(img)

        yaw, pitch, roll = self.model(images)

        # Continuous predictions
        yaw_predicted = utils.softmax_temperature(yaw.data, 1)
        pitch_predicted = utils.softmax_temperature(pitch.data, 1)
        roll_predicted = utils.softmax_temperature(roll.data, 1)

        yaw_predicted = torch.sum(yaw_predicted * idx_tensor, 1).cpu() * 3 - 99
        pitch_predicted = torch.sum(pitch_predicted * idx_tensor, 1).cpu() * 3 - 99
        roll_predicted = torch.sum(roll_predicted * idx_tensor, 1).cpu() * 3 - 99

        pitch = pitch_predicted[0]
        yaw = -yaw_predicted[0]
        roll = roll_predicted[0]
        return torch.tensor([pitch.item(), yaw.item(), roll.item()])

    def update(self, target, swap):
        target = self.get_face_points(target.unsqueeze(0))
        swap = self.get_face_points(swap.unsqueeze(0))
        self._metric += (target - swap).pow(2).sum().pow(0.5)
        self._len += 1


class Expression(EvalMetric):
    def __init__(self):
        super().__init__()
        self.model = FAN(4, "False", "False", 98)
        self.setup_model('pretrained_ckpts/WFLW_4HG.pth')
        self._mse = torch.nn.MSELoss(reduction='none')

    def setup_model(self, path_to_model: str):
        checkpoint = torch.load(path_to_model, map_location='cpu')
        if 'state_dict' not in checkpoint:
            self.model.load_state_dict(checkpoint)
        else:
            pretrained_weights = checkpoint['state_dict']
            model_weights = self.model.state_dict()
            pretrained_weights = {k: v for k, v in pretrained_weights.items() \
                                  if k in model_weights}
            model_weights.update(pretrained_weights)
            self.model.load_state_dict(model_weights)
        self.model.eval().to(device)

    def update(self, target, swap):
        swap = torch.nn.functional.interpolate(swap.unsqueeze(0), [256, 256], mode='bilinear', align_corners=False)
        target = torch.nn.functional.interpolate(target.unsqueeze(0), [256, 256], mode='bilinear', align_corners=False)

        swap_lmk, _ = detect_landmarks(swap, self.model, normalize=True)
        target_lmk, _ = detect_landmarks(target, self.model, normalize=True)

        self._metric += torch.norm(swap_lmk - target_lmk, 2)
        self._len += 1


@torch.no_grad()
def main(args):
    identity = Identity()
    pose = PoseMetric()
    expression = Expression()
    fid = os.popen(f"export PYTHONPATH='.' && python3 -m pytorch_fid {args.target} {args.swap}").read()[6:-2]
    for i, (source_path, target_path, swap_path) in enumerate(
            zip(os.listdir(args.source), os.listdir(args.target), os.listdir(args.swap))):
        print(i)
        source = Image.open(os.path.join(args.source, source_path)).convert('RGB')
        target = Image.open(os.path.join(args.target, target_path)).convert('RGB')
        swap = Image.open(os.path.join(args.swap, swap_path)).convert('RGB')
        source = transforms.ToTensor()(source).to(device)
        target = transforms.ToTensor()(target).to(device)
        swap = transforms.ToTensor()(swap).to(device)
        identity.update(source, swap)
        pose.update(target, swap)
        expression.update(target, swap)
    with open(args.output, "a") as file:
        file.write(f'cos_id -- {np.round(identity.get().item(), 2)}' + "\n")
        file.write(f'pose -- {np.round(pose.get().item(), 2)}' + "\n")
        file.write(f'expression -- {np.round(expression.get().item(), 2)}' + "\n")
        file.write(f'fid -- {np.round(float(fid), 2)}' + "\n")


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--source', type=str)
    args.add_argument('--target', type=str)
    args.add_argument('--swap', type=str)
    args.add_argument('--output', type=str, default='output_metrics.txt')
    args = args.parse_args()
    main(args)
