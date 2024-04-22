import argparse
import os
import warnings

import cv2
import dlib
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image
from scipy.linalg import sqrtm
from torch.autograd import Variable

from src.metrics import utils, hopenet
from src.metrics.face_recog_model import Backbone

warnings.filterwarnings("ignore")

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def get_face_points(path, model):
    transformations = transforms.Compose([transforms.Resize(224),
                                          transforms.CenterCrop(224), transforms.ToTensor(),
                                          transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    idx_tensor = [idx for idx in range(66)]
    idx_tensor = torch.FloatTensor(idx_tensor)

    img = Image.open(path)
    img = img.convert('RGB')
    img = transformations(img)
    img = img.unsqueeze(0)
    images = Variable(img)

    yaw, pitch, roll = model(images)

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


def calculate_cos_id(model, source, swap):
    source = Image.open(source)
    source = transforms.ToTensor()(source)

    swap = Image.open(swap)
    swap = transforms.ToTensor()(swap)

    source = (source * 2 - 1).unsqueeze(0).to(device)
    swap = (swap * 2 - 1).unsqueeze(0).to(device)
    source = model(source)
    swap = model(swap)
    return torch.dot(source.squeeze(), swap.squeeze()).item()


def calculate_pose(model, target, swap):
    trg = get_face_points(target, model)
    swp = get_face_points(swap, model)
    return (trg - swp).pow(2).sum().pow(0.5).item()


def calculate_expression(frontalFaceDetector, faceLandmarkDetector, target, swap):
    target = cv2.imread(target)
    swap = cv2.imread(swap)

    target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)
    swap = cv2.cvtColor(swap, cv2.COLOR_BGR2RGB)

    try:
        target_face = frontalFaceDetector(target, 0)[0]
        swap_face = frontalFaceDetector(swap, 0)[0]
    except:
        return None

    target_face = dlib.rectangle(int(target_face.left()), int(target_face.top()),
                                 int(target_face.right()), int(target_face.bottom()))
    swap_face = dlib.rectangle(int(swap_face.left()), int(swap_face.top()),
                               int(swap_face.right()), int(swap_face.bottom()))

    land_target = faceLandmarkDetector(target, target_face)
    land_swap = faceLandmarkDetector(swap, swap_face)

    target = np.zeros((68, 2))
    swap = np.zeros((68, 2))

    for i in range(68):
        target[i, :] = [land_target.part(i).x, land_target.part(i).y]
        swap[i, :] = [land_swap.part(i).x, land_swap.part(i).y]

    return np.sqrt(np.sum((target - swap) ** 2, axis=1)).mean()


def get_features(model, source, swap):
    source = Image.open(source)
    source = transforms.ToTensor()(source)

    swap = Image.open(swap)
    swap = transforms.ToTensor()(swap)

    img = torch.stack((source, swap))

    img = img.to(device)
    features = model(img).detach().cpu()
    source_features = features[0, ...].view(1, -1)
    swap_features = features[1, ...].view(1, -1)
    return source_features, swap_features


def calculate_fid(source_features, swap_features):
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

    return (diff.dot(diff) + np.trace(cov1) +
            np.trace(cov2) - 2 * tr_covmean)


def get_concat(source, target, swap, args, name):
    source = Image.open(source).convert('RGB')
    target = Image.open(target).convert('RGB')
    swap = Image.open(swap).convert('RGB')
    source = transforms.ToTensor()(source)
    target = transforms.ToTensor()(target)
    swap = transforms.ToTensor()(swap)
    sample = torch.cat([source, target], dim=2)
    sample = torch.cat([sample, swap], dim=2)
    sample = transforms.ToPILImage()(sample)
    sample.save(os.path.join(args.concat_dir, name))


def main(args):
    model_id_cos = Backbone().eval().requires_grad_(False).to(device)
    model_id_cos.load_state_dict(torch.load('pretrained_ckpts/auxiliray/model_ir_se50.pth', map_location=device))
    model_pose = hopenet.Hopenet(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], 66)
    model_pose.load_state_dict(torch.load('pretrained_ckpts/hopenet_robust_alpha1.pkl', map_location=device))
    model_pose.eval()
    frontalFaceDetector = dlib.get_frontal_face_detector()
    faceLandmarkDetector = dlib.shape_predictor("pretrained_ckpts/shape_predictor_68_face_landmarks.dat")
    model_features = torchvision.models.inception_v3(pretrained=True).to(device)
    model_features.fc = torch.nn.Identity()
    model_features.eval()
    pose, cos_id, exp = 0, 0, 0
    exp_count = 0
    source_features, swap_features = [], []
    for i, (source_path, target_path, swap_path) in enumerate(
            zip(os.listdir(args.source), os.listdir(args.target), os.listdir(args.swap))):
        source = os.path.join(args.source, source_path)
        target = os.path.join(args.target, target_path)
        swap = os.path.join(args.swap, swap_path)
        cos_id += calculate_cos_id(model_id_cos, source, swap)
        pose += calculate_pose(model_pose, target, swap)
        return_expression = calculate_expression(frontalFaceDetector, faceLandmarkDetector, target, swap)
        if return_expression is not None:
            exp += return_expression
            exp_count += 1
        s_feat, sw_feat = get_features(model_features, source, swap)
        source_features.append(s_feat)
        swap_features.append(sw_feat)
        if int(source_path[:-4]) < 40:
            os.makedirs(args.concat_dir, exist_ok=True)
            get_concat(source, target, swap, args, source_path)

    pose /= (i + 1)
    cos_id /= (i + 1)
    exp /= exp_count
    source_features = torch.stack(source_features).squeeze().numpy()
    swap_features = torch.stack(swap_features).squeeze().numpy()
    fid = calculate_fid(source_features, swap_features)
    with open(args.output, "a") as file:
        file.write(f'{args.model}: cos_id -- {np.round(cos_id, 2)}' + "\n")
        file.write(f'{args.model}: pose -- {np.round(pose, 2)}' + "\n")
        file.write(f'{args.model}: expression -- {np.round(exp, 2)}' + "\n")
        file.write(f'{args.model}: fid -- {np.round(fid, 2)}' + "\n")


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
