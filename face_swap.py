import os
import copy
import cv2
from PIL import Image
import torch
import warnings
import numpy as np
import torchvision.transforms as transforms
from torch.nn import functional as F
from skimage.transform import resize

from pretrained.face_vid2vid.driven_demo import init_facevid2vid_pretrained_model, drive_source_demo
from pretrained.gpen.gpen_demo import init_gpen_pretrained_model, GPEN_demo
from pretrained.face_parsing.face_parsing_demo import init_faceParsing_pretrained_model, faceParsing_demo, \
    vis_parsing_maps
from utils.swap_face_mask import swap_head_mask_revisit_considerGlass

from utils import torch_utils
from utils.alignment import crop_faces, calc_alignment_coefficients
from utils.morphology import dilation, erosion
from utils.multi_band_blending import blending
from utils.skin_color_transfer import skin_color_transfer

from options.swap_options import SwapFacePipelineOptions
from models.networks import Net3
from datasets.dataset import TO_TENSOR, NORMALIZE, __celebAHQ_masks_to_faceParser_mask_detailed

warnings.filterwarnings("ignore")


def create_masks(mask, outer_dilation=0, operation='dilation'):
    radius = outer_dilation
    temp = copy.deepcopy(mask)
    if operation == 'dilation':
        full_mask = dilation(temp, torch.ones(2 * radius + 1, 2 * radius + 1, device=mask.device), engine='convolution')
        border_mask = full_mask - temp
    elif operation == 'erosion':
        full_mask = erosion(temp, torch.ones(2 * radius + 1, 2 * radius + 1, device=mask.device), engine='convolution')
        border_mask = temp - full_mask
    # 'expansion' means to obtain a boundary that expands to both sides
    elif operation == 'expansion':
        full_mask = dilation(temp, torch.ones(2 * radius + 1, 2 * radius + 1, device=mask.device), engine='convolution')
        erosion_mask = erosion(temp, torch.ones(2 * radius + 1, 2 * radius + 1, device=mask.device),
                               engine='convolution')
        border_mask = full_mask - erosion_mask

    border_mask = border_mask.clip(0, 1)
    content_mask = mask

    return content_mask, border_mask, full_mask


def logical_or_reduce(*tensors):
    return torch.stack(tensors, dim=0).any(dim=0)


def logical_and_reduce(*tensors):
    return torch.stack(tensors, dim=0).all(dim=0)


def paste_image_mask(inverse_transform, image, dst_image, mask, radius=0, sigma=0.0):
    image_masked = image.copy().convert('RGBA')
    pasted_image = dst_image.copy().convert('RGBA')
    if radius != 0:
        mask_np = np.array(mask)
        kernel_size = (radius * 2 + 1, radius * 2 + 1)
        kernel = np.ones(kernel_size)
        eroded = cv2.erode(mask_np, kernel, borderType=cv2.BORDER_CONSTANT, borderValue=255)
        blurred_mask = cv2.GaussianBlur(eroded, kernel_size, sigmaX=sigma)
        blurred_mask = Image.fromarray(blurred_mask)
        image_masked.putalpha(blurred_mask)
    else:
        image_masked.putalpha(mask)

    projected = image_masked.transform(dst_image.size, Image.PERSPECTIVE, inverse_transform, Image.BILINEAR)
    pasted_image.alpha_composite(projected)
    return pasted_image


def paste_image(coeffs, img, orig_image):
    pasted_image = orig_image.copy().convert('RGBA')
    projected = img.convert('RGBA').transform(orig_image.size, Image.PERSPECTIVE, coeffs, Image.BILINEAR)
    pasted_image.paste(projected, (0, 0), mask=projected)
    return pasted_image


def smooth_face_boundry(image, dst_image, mask, radius=0, sigma=0.0):
    image_masked = image.copy().convert('RGBA')
    pasted_image = dst_image.copy().convert('RGBA')
    if radius != 0:
        mask_np = np.array(mask)
        kernel_size = (radius * 2 + 1, radius * 2 + 1)
        kernel = np.ones(kernel_size)
        eroded = cv2.erode(mask_np, kernel, borderType=cv2.BORDER_CONSTANT, borderValue=255)
        blurred_mask = cv2.GaussianBlur(eroded, kernel_size, sigmaX=sigma)
        blurred_mask = Image.fromarray(blurred_mask)
        image_masked.putalpha(blurred_mask)
    else:
        image_masked.putalpha(mask)

    pasted_image.alpha_composite(image_masked)
    return pasted_image


# ===================================
def crop_and_align_face(target_files):
    image_size = 1024
    scale = 1.0
    center_sigma = 0
    xy_sigma = 0
    use_fa = False

    print('Aligning images')
    crops, orig_images, quads = crop_faces(image_size, target_files, scale, center_sigma=center_sigma,
                                           xy_sigma=xy_sigma, use_fa=use_fa)

    inv_transforms = [
        calc_alignment_coefficients(quad + 0.5, [[0, 0], [0, image_size], [image_size, image_size], [image_size, 0]])
        for quad in quads
    ]

    return crops, orig_images, quads, inv_transforms


def swap_comp_style_vector(style_vectors1, style_vectors2, comp_indices=[]):
    """Replace the style_vectors1 with style_vectors2

    Args:
        style_vectors1 (Tensor): with shape [1,#comp,512], style vectors of target image
        style_vectors2 (Tensor): with shape [1,#comp,512], style vectors of source image
        comp_indices (List):j regions with source attributes
    """
    assert comp_indices is not None

    style_vectors = copy.deepcopy(style_vectors1)

    for comp_idx in comp_indices:
        style_vectors[:, comp_idx, :] = style_vectors2[:, comp_idx, :]

    # if no ear(7) region for source
    if torch.sum(style_vectors2[:, 7, :]) == 0:
        style_vectors[:, 7, :] = (style_vectors1[:, 7, :] + style_vectors2[:, 7, :]) / 2

    # if no teeth(9) region for source
    if torch.sum(style_vectors2[:, 9, :]) == 0:
        style_vectors[:, 9, :] = style_vectors1[:, 9, :]

    return style_vectors


@torch.no_grad()
def faceSwapping_pipeline(opts, source, target, name, target_mask=None, need_crop=False, only_target_crop=False):
    """
    The overall pipeline of face swapping:

        Input: source image, target image

        (1) Crop the faces from the source and target and align them, obtaining S and T ; (cropping is optional)
        (2) Use faceVid2Vid & GPEN to re-enact S, resulting in driven face D, and then parsing the mask of D
        (3) Extract the texture vectors of D and T using RGI
        (4) Texture and shape swapping between face D and face T
        (5) Feed the swapped mask and texture vectors to the generator, obtaining swapped face I
        (6) Re-color I, using T, and enhance I after that
        (7) Stitch I back to the target image

    Args:
        opts (): args
        source (str): path to source
        target (str): path to target
        name (str): file name for saving
        target_mask (ndarray): 12-class segmentation map, will be estimated if not provided
        need_crop (bool): crop all images
        only_target_crop (bool): only crop target image
    """

    dirs = {'swap': os.path.join(opts.output_dir, 'swap')}

    if not opts.save_only_swap:
        for path in ['source', 'target']:
            dirs[path] = os.path.join(opts.output_dir, path)
    if opts.save_concat:
        dirs['fuse'] = os.path.join(opts.output_dir, 'fuse')
    if opts.verbose:
        dirs['T_crop'] = os.path.join(opts.output_dir, 'T_crop')
        dirs['S_crop'] = os.path.join(opts.output_dir, 'S_crop')
        dirs['T_mask'] = os.path.join(opts.output_dir, 'T_mask')
        dirs['T_mask_vis'] = os.path.join(opts.output_dir, 'T_mask_vis')
        dirs['Drive'] = os.path.join(opts.output_dir, 'Drive')
        dirs['D_mask'] = os.path.join(opts.output_dir, 'D_mask')
        dirs['D_mask_vis'] = os.path.join(opts.output_dir, 'D_mask_vis')
        dirs['D_style_vec'] = os.path.join(opts.output_dir, 'D_style_vec')
        dirs['D_recon'] = os.path.join(opts.output_dir, 'D_recon')
        dirs['T_style_vec'] = os.path.join(opts.output_dir, 'T_style_vec')
        dirs['T_recon'] = os.path.join(opts.output_dir, 'T_recon')
        dirs['swappedMask'] = os.path.join(opts.output_dir, 'swappedMask')
        dirs['swappedMaskVis'] = os.path.join(opts.output_dir, 'swappedMaskVis')
        dirs['swapped_style_vec'] = os.path.join(opts.output_dir, 'swapped_style_vec')
        dirs['dirty_swap'] = os.path.join(opts.output_dir, 'dirty_swap')
        dirs['recolor_img'] = os.path.join(opts.output_dir, 'recolor_img')
        dirs['enhance_img'] = os.path.join(opts.output_dir, 'enhance_img')

    for key, dir_ in dirs.items():
        os.makedirs(dir_, exist_ok=True)

    source_and_target_files = [source, target]
    source_and_target_files = [(os.path.basename(f).split('.')[0], f) for f in source_and_target_files]

    # (1) Crop the faces from the source and target and align them, obtaining S and T
    if only_target_crop:
        crops, orig_images, quads, inv_transforms = crop_and_align_face(source_and_target_files[1:])
        crops = [crop.convert("RGB") for crop in crops]
        T = crops[0]
        S = Image.open(source).convert("RGB").resize((1024, 1024))
        T.save(os.path.join(dirs['T_crop'], f'{name}.jpg'))
    elif need_crop:
        crops, orig_images, quads, inv_transforms = crop_and_align_face(source_and_target_files)
        crops = [crop.convert("RGB") for crop in crops]
        S, T = crops
        T.save(os.path.join(dirs['T_crop'], f'{name}.jpg'))
        S.save(os.path.join(dirs['S_crop'], f'{name}.jpg'))
    else:
        S = Image.open(source).convert("RGB").resize((1024, 1024))
        T = Image.open(target).convert("RGB").resize((1024, 1024))
        S.save(os.path.join(dirs['source'], f'{name}.jpg'))
        T.save(os.path.join(dirs['target'], f'{name}.jpg'))

    S_256, T_256 = [resize(np.array(im) / 255.0, (256, 256)) for im in [S, T]]  # 256,[0,1] range
    T_mask = faceParsing_demo(faceParsing_model, T, convert_to_seg12=True,
                              model_name=opts.faceParser_name) if target_mask is None else target_mask
    if opts.verbose:
        Image.fromarray(T_mask).save(os.path.join(dirs['T_mask'], f"{name}.png"))
        T_mask_vis = vis_parsing_maps(T, T_mask)
        Image.fromarray(T_mask_vis).save(os.path.join(dirs['T_mask_vis'], f"{name}.png"))

    # (2) faceVid2Vid  input & output [0,1] range with RGB
    predictions = drive_source_demo(S_256, [T_256], generator, kp_detector, he_estimator, estimate_jacobian)
    predictions = [(pred * 255).astype(np.uint8) for pred in predictions]
    # del generator, kp_detector, he_estimator

    # (2) GPEN input & output [0,255] range with BGR
    drives = [GPEN_demo(pred[:, :, ::-1], GPEN_model, aligned=False) for pred in predictions]
    D = Image.fromarray(drives[0][:, :, ::-1])  # to PIL.Image
    if opts.verbose:
        D.save(os.path.join(dirs['Drive'], f"{name}.png"))

    # (2) mask of D
    D_mask = faceParsing_demo(faceParsing_model, D, convert_to_seg12=True, model_name=opts.faceParser_name)
    if opts.verbose:
        Image.fromarray(D_mask).save(os.path.join(dirs['D_mask'], f"{name}.png"))
        D_mask_vis = vis_parsing_maps(D, D_mask)
        Image.fromarray(D_mask_vis).save(os.path.join(dirs['D_mask_vis'], f"{name}.png"))

    # wrap data
    driven = transforms.Compose([TO_TENSOR, NORMALIZE])(D)
    driven = driven.to(opts.device).float().unsqueeze(0)
    driven_mask = transforms.Compose([TO_TENSOR])(Image.fromarray(D_mask))
    driven_mask = (driven_mask * 255).long().to(opts.device).unsqueeze(0)
    driven_onehot = torch_utils.labelMap2OneHot(driven_mask, num_cls=opts.num_seg_cls)

    target = transforms.Compose([TO_TENSOR, NORMALIZE])(T)
    target = target.to(opts.device).float().unsqueeze(0)
    target_mask = transforms.Compose([TO_TENSOR])(Image.fromarray(T_mask))
    target_mask = (target_mask * 255).long().to(opts.device).unsqueeze(0)
    target_onehot = torch_utils.labelMap2OneHot(target_mask, num_cls=opts.num_seg_cls)

    # (3) Extract the texture vectors of D and T using RGI
    driven_style_vector, _ = net.get_style_vectors(driven, driven_onehot)
    target_style_vector, _ = net.get_style_vectors(target, target_onehot)
    if opts.verbose:
        torch.save(driven_style_vector, os.path.join(dirs['D_style_vec'], f"{name}.pt"))
        driven_style_codes = net.cal_style_codes(driven_style_vector)
        driven_face, _, structure_feats = net.gen_img(torch.zeros(1, 512, 32, 32).to(opts.device), driven_style_codes,
                                                      driven_onehot)
        driven_face_image = torch_utils.tensor2im(driven_face[0])
        driven_face_image.save(os.path.join(dirs['D_recon'], f"{name}.png"))

        torch.save(target_style_vector, os.path.join(dirs['T_style_vec'], f"{name}.pt"))
        target_style_codes = net.cal_style_codes(target_style_vector)
        target_face, _, structure_feats = net.gen_img(torch.zeros(1, 512, 32, 32).to(opts.device), target_style_codes,
                                                      target_onehot)
        target_face_image = torch_utils.tensor2im(target_face[0])
        target_face_image.save(os.path.join(dirs['T_recon'], f"{name}.png"))

    # (4) shape swapping between face D and face T
    swapped_msk, hole_map = swap_head_mask_revisit_considerGlass(D_mask, T_mask)

    if opts.verbose:
        cv2.imwrite(os.path.join(dirs['swappedMask'], f"{name}.png"), swapped_msk)
        swappped_one_hot = torch_utils.labelMap2OneHot(torch.from_numpy(swapped_msk).unsqueeze(0).unsqueeze(0).long(),
                                                       num_cls=12)
        torch_utils.tensor2map(swappped_one_hot[0]).save(os.path.join(dirs['swappedMaskVis'], f"{name}.png"))

    # Texture swapping between face D and face T. Retain the style_vectors of background(0), hair(4), ear_rings(11), eye_glass(10) from target
    comp_indices = set(range(opts.num_seg_cls)) - {0, 4, 11, 10}
    swapped_style_vectors = swap_comp_style_vector(target_style_vector, driven_style_vector, list(comp_indices))
    if opts.verbose:
        torch.save(swapped_style_vectors, os.path.join(dirs['swapped_style_vec'], f"{name}.pt"))

    # (5) Feed the swapped mask and texture vectors to the generator, obtaining swapped face I;
    swapped_msk = Image.fromarray(swapped_msk).convert('L')
    swapped_msk = transforms.Compose([TO_TENSOR])(swapped_msk)
    swapped_msk = (swapped_msk * 255).long().to(opts.device).unsqueeze(0)
    swapped_onehot = torch_utils.labelMap2OneHot(swapped_msk, num_cls=opts.num_seg_cls)
    #
    swapped_style_codes = net.cal_style_codes(swapped_style_vectors)
    swapped_face, _, structure_feats = net.gen_img(torch.zeros(1, 512, 32, 32).to(opts.device), swapped_style_codes,
                                                   swapped_onehot)
    swapped_face_image = torch_utils.tensor2im(swapped_face[0])
    if opts.verbose:
        swapped_face_image.save(os.path.join(dirs['dirty_swap'], f"{name}.png"))

    # (6) Re-color I, using T

    # For re-color I will re-color face by mask with nose(5), skin(6), ears(7) and neck(8)
    mask_face_swapped = logical_or_reduce(*[swapped_msk == item for item in [5, 6, 7, 8]]).float()
    mask_face_target = logical_or_reduce(*[target_mask == item for item in [5, 6, 7, 8]]).float()
    mask_face_swapped = F.interpolate(mask_face_swapped, (1024, 1024), mode='bilinear', align_corners=False)
    mask_face_target = F.interpolate(mask_face_target, (1024, 1024), mode='bilinear', align_corners=False)

    _, face_border, _ = create_masks(mask_face_swapped, operation='expansion', outer_dilation=5)
    face_border = face_border[0, 0, :, :, None].cpu().numpy()
    face_border = np.repeat(face_border, 3, axis=-1)

    mask_face_swapped = mask_face_swapped[0, 0, :, :, None].cpu().numpy()
    mask_face_target = mask_face_target[0, 0, :, :, None].cpu().numpy()

    T = np.asarray(T)
    SW = np.asarray(swapped_face_image)

    swapped_face_inner = SW * mask_face_swapped
    target_face_inner = T * mask_face_target
    # re-color
    swapped_face_inner = skin_color_transfer(swapped_face_inner / 255., target_face_inner / 255.)
    # blend re-color face with no-face part of swapped_image
    swapped_face_image_ = SW * (1 - mask_face_swapped) + swapped_face_inner * 255 * mask_face_swapped
    color_transfer_image = Image.fromarray(blending(SW, swapped_face_image_, mask=face_border))
    if opts.verbose:
        color_transfer_image.save(os.path.join(dirs['recolor_img'], f"{name}.png"))

    # enhance I
    mask_softer = torch_utils.SoftErosion().to(opts.device)
    blending_mask = torch_utils.get_facial_mask_from_seg19(
        swapped_msk,
        target_size=color_transfer_image.size,
        edge_softer=mask_softer
    )
    edge_img = torch_utils.get_edge(swapped_face_image)
    edge_img = np.array(edge_img).astype(np.float32) / 255.
    blending_mask = (blending_mask - edge_img).clip(0., 1.)  # remove high-frequency parts
    swapped_face_image = torch_utils.blending_two_images_with_mask(
        swapped_face_image, color_transfer_image,
        up_ratio=0.75, up_mask=blending_mask.copy()
    )
    if opts.verbose:
        swapped_face_image.save(os.path.join(dirs['enhance_img'], f"{name}.png"))

    # (7) Stitch I back to the target image
    #
    # Gaussian blending with mask
    outer_dilation = 5
    # for more consistent swap make mask using bg(0), hair(4), ears(7), neck(8), glass(10) and ear rings(11)
    mask_bg = logical_or_reduce(*[swapped_msk == clz for clz in [0, 4, 7, 8, 10, 11]])
    is_foreground = torch.logical_not(mask_bg)
    hole_index = hole_map[None][None] == 255
    is_foreground[hole_index[None]] = True
    foreground_mask = is_foreground.float()

    if opts.lap_bld:
        content_mask, border_mask, full_mask = create_masks(foreground_mask, outer_dilation=outer_dilation,
                                                            operation='expansion')
    else:
        content_mask, border_mask, full_mask = create_masks(foreground_mask, outer_dilation=outer_dilation)

    content_mask = F.interpolate(content_mask, (1024, 1024), mode='bilinear', align_corners=False)
    content_mask_image = Image.fromarray(255 * content_mask[0, 0, :, :].cpu().numpy().astype(np.uint8))
    full_mask = F.interpolate(full_mask, (1024, 1024), mode='bilinear', align_corners=False)
    full_mask_image = Image.fromarray(255 * full_mask[0, 0, :, :].cpu().numpy().astype(np.uint8))

    # Paste swapped face onto the target's face
    if opts.lap_bld:
        content_mask = content_mask[0, 0, :, :, None].cpu().numpy()
        border_mask = F.interpolate(border_mask, (1024, 1024), mode='bilinear', align_corners=False)
        border_mask = border_mask[0, 0, :, :, None].cpu().numpy()
        border_mask = np.repeat(border_mask, 3, axis=-1)

        swapped_and_pasted = swapped_face_image * content_mask + T * (1 - content_mask)
        swapped_and_pasted = Image.fromarray(np.uint8(swapped_and_pasted))
        swapped_and_pasted = Image.fromarray(blending(np.array(T), np.array(swapped_and_pasted), mask=border_mask))
    else:
        if outer_dilation == 0:
            swapped_and_pasted = smooth_face_boundry(swapped_face_image, T, content_mask_image, radius=outer_dilation)
        else:
            swapped_and_pasted = smooth_face_boundry(swapped_face_image, T, full_mask_image, radius=outer_dilation)

    # Restore to original image from cropped area
    if only_target_crop:
        inv_trans_coeffs, orig_image = inv_transforms[0], orig_images[0]
        swapped_and_pasted = swapped_and_pasted.convert('RGBA')
        pasted_image = orig_image.convert('RGBA')
        swapped_and_pasted.putalpha(255)
        projected = swapped_and_pasted.transform(orig_image.size, Image.PERSPECTIVE, inv_trans_coeffs, Image.BILINEAR)
        pasted_image.alpha_composite(projected)
    elif need_crop:
        inv_trans_coeffs, orig_image = inv_transforms[1], orig_images[1]
        swapped_and_pasted = swapped_and_pasted.convert('RGBA')
        pasted_image = orig_image.convert('RGBA')
        swapped_and_pasted.putalpha(255)
        projected = swapped_and_pasted.transform(orig_image.size, Image.PERSPECTIVE, inv_trans_coeffs, Image.BILINEAR)
        pasted_image.alpha_composite(projected)
    else:
        pasted_image = swapped_and_pasted

    pasted_image = pasted_image.convert('RGB')
    pasted_image.save(os.path.join(dirs['swap'], f'{name}.jpg'))
    if opts.save_concat:  # TODO добавить сохранения конкатов для фоток с разными размерами
        source = Image.open(os.path.join(dirs['source'], f'{name}.jpg'))
        target = Image.open(os.path.join(dirs['target'], f'{name}.jpg'))

        source = transforms.ToTensor()(source)
        target = transforms.ToTensor()(target)
        swap = transforms.ToTensor()(pasted_image)

        sample = torch.cat([source, target], dim=2)
        sample = torch.cat([sample, swap], dim=2)
        sample = transforms.ToPILImage()(sample)
        sample.save(os.path.join(dirs['fuse'], f'{name}.jpg'))


if __name__ == "__main__":
    opts = SwapFacePipelineOptions().parse()
    # ================= Pre-trained models initialization =========================
    # face_vid2vid
    face_vid2vid_cfg = "./pretrained_ckpts/facevid2vid/vox-256.yaml"
    face_vid2vid_ckpt = "./pretrained_ckpts/facevid2vid/00000189-checkpoint.pth.tar"
    generator, kp_detector, he_estimator, estimate_jacobian = init_facevid2vid_pretrained_model(face_vid2vid_cfg,
                                                                                                face_vid2vid_ckpt)

    # GPEN
    gpen_model_params = {
        "base_dir": "./pretrained_ckpts/gpen/",  # a sub-folder named <weights> should exist
        "in_size": 512,
        "model": "GPEN-BFR-512",
        "use_sr": True,
        "sr_model": "realesrnet",
        "sr_scale": 4,
        "channel_multiplier": 2,
        "narrow": 1,
    }
    GPEN_model = init_gpen_pretrained_model(model_params=gpen_model_params)

    # face parser
    if opts.faceParser_name == "default":
        faceParser_ckpt = "./pretrained_ckpts/face_parsing/79999_iter.pth"
        config_path = ""
    elif opts.faceParser_name == "segnext":
        faceParser_ckpt = "./pretrained_ckpts/face_parsing/best_mIoU_iter_150000.pth"
        config_path = "./pretrained_ckpts/face_parsing/segnext.large.512x512.celebamaskhq.160k.py"
    else:
        raise NotImplementedError("Please choose a valid face parser,"
                                  "the current supported models are [ default | segnext ], but %s is given." % opts.faceParser_name)

    faceParsing_model = init_faceParsing_pretrained_model(opts.faceParser_name, faceParser_ckpt, config_path)
    print("Load pre-trained face parsing models success!")

    # E4S model
    net = Net3(opts)
    net = net.to(opts.device)
    save_dict = torch.load(opts.checkpoint_path)
    net.load_state_dict(torch_utils.remove_module_prefix(save_dict["state_dict"], prefix="module."))
    net.latent_avg = save_dict['latent_avg'].to(opts.device)
    print("Load E4S pre-trained model success!")
    # ========================================================

    if len(opts.target_mask) != 0:
        target_mask = Image.open(opts.target_mask).convert("L")
        target_mask_seg12 = __celebAHQ_masks_to_faceParser_mask_detailed(target_mask)
    else:
        target_mask_seg12 = None

    if opts.evaluate:
        source_dir = sorted(
            [os.path.join(opts.celeba_dataset_root, "CelebA-HQ-img", "%d.jpg" % idx) for idx in range(28000, 29000)])
        target_dir = sorted(
            [os.path.join(opts.celeba_dataset_root, "CelebA-HQ-img", "%d.jpg" % idx) for idx in range(29000, 30000)])
    else:
        source_dir = os.listdir(opts.source)
        target_dir = os.listdir(opts.target)

    for i, (source, target) in enumerate(zip(source_dir, target_dir)):
        if opts.evaluate:
            source_path = source
            target_path = target
        else:
            source_path = os.path.join(opts.source, source)
            target_path = os.path.join(opts.target, target)
        faceSwapping_pipeline(opts, source_path, target_path, str(i), target_mask=target_mask_seg12,
                              need_crop=opts.need_crop)
