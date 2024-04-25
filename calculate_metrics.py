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


def get_concat(args, source, target, swap, name):
    sample = torch.cat([source, target], dim=2)
    sample = torch.cat([sample, swap], dim=2)
    sample = transforms.ToPILImage()(sample)
    sample.save(os.path.join(args.concat_dir, name))


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
        self.model.load_state_dict(torch.load('./pretrained_ckpts/arcface.pt'), map_location=device)
        self.model.to(device).eval()

    def update(self, source, swap):
        source = self.model(torch.nn.functional.interpolate(source, [112, 112], mode='bilinear',
                                                            align_corners=False))
        swap = self.model(torch.nn.functional.interpolate(swap, [112, 112], mode='bilinear',
                                                          align_corners=False))
        self._metric += torch.cosine_similarity(source, swap, dim=1)
        self._len += 1


class PoseMetric(EvalMetric):
    def __init__(self):
        super().__init__()
        self.model = hopenet.Hopenet(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], 66)
        self.model.load_state_dict(torch.load('./pretrained_ckpts/hopenet_robust_alpha1.pkl', map_location=device))
        self.model.eval()

    def get_face_points(self, img):
        idx_tensor = [idx for idx in range(66)]
        idx_tensor = torch.FloatTensor(idx_tensor)

        img = img.unsqueeze(0)
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
        target = self.get_face_points(
            torch.nn.functional.interpolate(target, [224, 224], mode='bilinear', align_corners=False))
        swap = self.get_face_points(
            torch.nn.functional.interpolate(swap, [224, 224], mode='bilinear', align_corners=False))
        self._metric += (target - swap).pow(2).sum().pow(0.5)
        self._len += 1


class Expression(EvalMetric):
    def __init__(self):
        super().__init__()
        self.model = FAN(4, "False", "False", 98)
        self.setup_model('./pretrained_models/WFLW_4HG.pth')
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

    def update(self, target, prediction):
        prediction = torch.nn.functional.interpolate(prediction, [256, 256], mode='bilinear', align_corners=False)
        target = torch.nn.functional.interpolate(target, [256, 256], mode='bilinear', align_corners=False)

        prediction_lmk, _ = detect_landmarks(prediction, self.model, normalize=True)
        target_lmk, _ = detect_landmarks(target, self.model, normalize=True)

        self._metric += torch.norm(prediction_lmk - target_lmk, 2)
        self._len += 1


class Fid(EvalMetric):
    def __init__(self):
        super().__init__()
        self.model = torchvision.models.inception_v3(pretrained=True).to(device)
        self.model.fc = torch.nn.Identity()
        self.model.eval()
        self.s_feat, self.sw_feat = [], []

    def get_features(self, source, swap):
        img = torch.stack((source, swap))

        img = img.to(device)
        features = self.model(img).detach().cpu()
        source_features = features[0, ...].view(1, -1)
        swap_features = features[1, ...].view(1, -1)
        self.s_feat.append(source_features)
        self.sw_feat.append(swap_features)

    def update(self):
        source_features = torch.stack(self.s_feat).squeeze().numpy()
        swap_features = torch.stack(self.sw_feat).squeeze().numpy()
        eps = 1e-6
        mu1 = np.mean(source_features, axis=0)
        mu2 = np.mean(swap_features, axis=0)
        cov1 = np.cov(source_features, rowvar=False)
        cov2 = np.cov(swap_features, rowvar=False)
        assert mu1.shape == mu2.shape, \
            'Training and test mean vectors have different lengths'
        assert cov1.shape == cov2.shape, \
            'Training and test covariances have different dimensions'
        diff = mu1 - mu2
        # Product might be almost singular
        covmean, _ = sqrtm(cov1.dot(cov2), disp=False)
        if not np.isfinite(covmean).all():
            offset = np.eye(cov1.shape[0]) * eps
            covmean = sqrtm((cov1 + offset).dot(cov2 + offset))

        # Numerical error might give slight imaginary component
        if np.iscomplexobj(covmean):
            if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
                m = np.max(np.abs(covmean.imag))
                raise ValueError('Imaginary component {}'.format(m))
            covmean = covmean.real

        tr_covmean = np.trace(covmean)

        self._metric += diff.dot(diff) + np.trace(cov1) + np.trace(cov2) - 2 * tr_covmean
        self._len += 1


@torch.no_grad()
def main(args):
    identity = Identity()
    pose = PoseMetric()
    expression = Expression()
    fid = Fid()
    for i, (source_path, target_path, swap_path) in enumerate(
            zip(os.listdir(args.source), os.listdir(args.target), os.listdir(args.swap))):
        source = Image.open(os.path.join(args.source, source_path)).convert('RGB')
        target = Image.open(os.path.join(args.target, target_path)).convert('RGB')
        swap = Image.open(os.path.join(args.swap, swap_path)).convert('RGB')
        source = transforms.ToTensor()(source).to(device)
        target = transforms.ToTensor()(target).to(device)
        swap = transforms.ToTensor()(swap).to(device)
        identity.update(source, swap)
        pose.update(target, swap)
        expression.update(target, swap)
        fid.get_features(source, swap)
        if int(source_path[:-4]) < 40:
            os.makedirs(args.concat_dir, exist_ok=True)
            get_concat(args, source, target, swap, source_path)
    with open(args.output, "a") as file:
        file.write(f'{args.model}: cos_id -- {np.round(identity.get().item(), 2)}' + "\n")
        file.write(f'{args.model}: pose -- {np.round(pose.get().item(), 2)}' + "\n")
        file.write(f'{args.model}: expression -- {np.round(expression.get().item(), 2)}' + "\n")
        file.write(f'{args.model}: fid -- {np.round(fid.get(), 2)}' + "\n")


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--source', type=str)
    args.add_argument('--target', type=str)
    args.add_argument('--swap', type=str)
    args.add_argument('--output', type=str)
    args.add_argument('--model', type=str)
    args.add_argument('--concat_dir', type=str)
    args = args.parse_args()
    main(args)
