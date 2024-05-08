import gdown
import os


def download():
    gdown.download(id="1bBmt21B5d-X29cGTqIdjkzcy3zh3wWTS")  # download model_ir_se50.pth
    gdown.download(id="1W-BxYgNOAgZvCuPTsJczAzglQwuU6Lo1")  # download iteration_100000.pt
    gdown.download(id="11cP8ceL2t3A2XoYqIBPIY5fIT94EosE5")  # download best_mIoU_iter_150000.pth
    gdown.download(id="1tnWOrdwAhvtMxbDYSCnUIAi-j5H4u3HV")  # download RetinaFace-R50.pth
    gdown.download(id="1VOOVi5ZOjU5how64j6QW3ENmhDgg7Yu3")  # download realesrnet_x4.pth
    gdown.download(id="1dRBVwYwtWpH0bYsFcvnCewve2NjTqiJ7")  # download ParseNet-latest.pth
    gdown.download(id="1Oidc5g9170RKvuWtHGOsqrPziAKvbeIk")  # download GPEN-BFR-512.pth
    gdown.download(id="1XPtbPTRWCqZuH_p-Yw6UHd3NcWQahPnl")  # download stylegan2-ffhq-config-f.pt
    gdown.download(id="1Wg0VC0rJ3w1BIHOzwBPHfxYa2yXzMSHX")  # download arcface.pt
    gdown.download(id="1AEFt0jgfziK3bkLQjSLIcP28GSQS59Ki")  # download hopenet_robust_alpha1.pkl
    gdown.download(id="1XQvlw3JqfeKCv7Or9dKSCzfRYen9ADcb")  # download shape_predictor_68_face_landmarks.dat
    gdown.download(id="1uivmaQEETh0pEJvH21CYv7e6u9RJW-PV")  # download WFLW_4HG.pth
    gdown.download(id="1uXUsxEDdziuOL_hEsi4EBzIre4h91pel")  # download vox.pt

    os.rename('model_ir_se50.pth', 'pretrained_ckpts/auxiliary/model_ir_se50.pth')
    os.makedirs('pretrained_ckpts/e4s', exist_ok=True)
    os.rename('iteration_100000.pt', 'pretrained_ckpts/e4s/iteration_100000.pt')
    os.rename('best_mIoU_iter_150000.pth', 'pretrained_ckpts/face_parsing/best_mIoU_iter_150000.pth')
    os.makedirs('pretrained_ckpts/gpen/weights', exist_ok=True)
    os.rename('RetinaFace-R50.pth', 'pretrained_ckpts/gpen/weights/RetinaFace-R50.pth')
    os.rename('realesrnet_x4.pth', 'pretrained_ckpts/gpen/weights/realesrnet_x4.pth')
    os.rename('ParseNet-latest.pth', 'pretrained_ckpts/gpen/weights/ParseNet-latest.pth')
    os.rename('GPEN-BFR-512.pth', 'pretrained_ckpts/gpen/weights/GPEN-BFR-512.pth')
    os.makedirs('pretrained_ckpts/stylegan2', exist_ok=True)
    os.rename('stylegan2-ffhq-config-f.pt', 'pretrained_ckpts/stylegan2/stylegan2-ffhq-config-f.pt')
    os.rename('arcface.pt', 'pretrained_ckpts/arcface.pt')
    os.rename('hopenet_robust_alpha1.pkl', 'pretrained_ckpts/hopenet_robust_alpha1.pkl')
    os.rename('shape_predictor_68_face_landmarks.dat', 'pretrained_ckpts/shape_predictor_68_face_landmarks.dat')
    os.rename('WFLW_4HG.pth', 'pretrained_ckpts/WFLW_4HG.pth')
    os.rename('vox.pt', 'pretrained_ckpts/vox.pt')


if __name__ == "__main__":
    download()
