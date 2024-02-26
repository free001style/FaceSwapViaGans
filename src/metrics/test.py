from PIL import Image
import torch
import torchvision.transforms.functional as tv_f
from src.metrics.face_recog_model import Backbone
from sklearn.metrics.pairwise import cosine_similarity
import dlib, cv2
import os, argparse
import numpy as np
from torch.autograd import Variable
from torchvision import transforms
import torchvision

import utils, hopenet


def append_to_file(file_path, text):
    if not os.path.exists(file_path):
        with open(file_path, "w"):
            pass
    with open(file_path, "a") as file:
        file.write(text + "\n")


def id_cos(args):
    device = 'cpu'

    facenet = Backbone().eval().requires_grad_(False).to(device)
    facenet.load_state_dict(torch.load('model_ir_se50.pth', map_location='cpu'))
    ans = 0
    j = 0

    for source_path, swap_path in zip(os.listdir(args.source), os.listdir(args.swap)):
        source = os.path.join(args.source, source_path)
        swap = os.path.join(args.swap, swap_path)

        source = Image.open(source)
        source = tv_f.to_tensor(source)

        swap = Image.open(swap)
        swap = tv_f.to_tensor(swap)

        source = (source * 2 - 1).unsqueeze(0).to(device)
        swap = (swap * 2 - 1).unsqueeze(0).to(device)
        source = facenet(source)
        swap = facenet(swap)
        ans += cosine_similarity(source, swap)[0, 0]
        j += 1

    append_to_file(args.output, f'{args.model}: ID cos -- {np.round(ans / j, 2)}')


def expression(args):
    Model_PATH = "shape_predictor_68_face_landmarks.dat"
    frontalFaceDetector = dlib.get_frontal_face_detector()
    faceLandmarkDetector = dlib.shape_predictor(Model_PATH)

    target_dir = args.target
    swap_dir = args.swap

    j = 0
    ans = 0

    for target_path, swap_path in zip(os.listdir(target_dir), os.listdir(swap_dir)):
        target = os.path.join(target_dir, target_path)
        swap = os.path.join(swap_dir, swap_path)

        target = cv2.imread(target)
        swap = cv2.imread(swap)

        target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)
        swap = cv2.cvtColor(swap, cv2.COLOR_BGR2RGB)

        try:
            target_face = frontalFaceDetector(target, 0)[0]
            swap_face = frontalFaceDetector(swap, 0)[0]
        except:
            continue

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

        ans += np.sqrt(np.sum((target - swap) ** 2, axis=1)).mean()
        j += 1
    append_to_file(args.output, f'{args.model}: Expr. -- {np.round(ans / j, 2)}')


def _calculate(path, model):
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


def calculate_pose(target, swap, model):
    trg = _calculate(target, model)
    swp = _calculate(swap, model)
    return (trg - swp).pow(2).sum().pow(0.5).item()


def pose(args):
    snapshot_path = "hopenet_robust_alpha1.pkl"
    model = hopenet.Hopenet(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], 66)

    saved_state_dict = torch.load(snapshot_path, map_location='cpu')
    model.load_state_dict(saved_state_dict)

    model.eval()

    ans = 0
    j = 0
    for target, swap in zip(os.listdir(args.target), os.listdir(args.swap)):
        target_path = os.path.join(args.target, target)
        swap_path = os.path.join(args.swap, swap)
        ans += calculate_pose(target_path, swap_path, model)
        j += 1
    append_to_file(args.output, f'{args.model}: Pose -- {np.round(ans / j, 2)}')


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--source', type=str)
    args.add_argument('--target', type=str)
    args.add_argument('--swap', type=str)
    args.add_argument('--output', type=str)
    args.add_argument('--model', type=str)
    args = args.parse_args()
    id_cos(args)
    expression(args)
    pose(args)
