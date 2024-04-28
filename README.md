# FaceSwap Via GANs

This is my 3rd year course work in the HSE University.

## Start
Weights can be downloaded using `download_weights.py` or [here](https://drive.google.com/drive/folders/1C5qIjiIsaswRGvXEB5HaeJhYVXPJ0Ozd).
In case using link, make sure that `pretrained_ckpts` directory looks like below.

```
    pretrained_ckpts/
    ├── auxiliray (need for training)
    │   ├── model_ir_se50.pth
    │   └── model.pth
    ├── e4s
    │   └── iteration_100000.pt
    ├── face_parsing
    │   ├── segnext.large.512x512.celebamaskhq.160k.py
    │   └── best_mIoU_iter_150000.pth
    ├── facevid2vid
    │   ├── 00000189-checkpoint.pth.tar
    │   └── vox-256.yaml
    ├── gpen
    │   ├── fetch_gepn_models.sh
    │   └── weights
    │       ├── GPEN-BFR-512.pth
    │       ├── ParseNet-latest.pth
    │       ├── realesrnet_x4.pth
    │       └── RetinaFace-R50.pth
    ├── stylegan2 (need for training)
    │   └── stylegan2-ffhq-config-f.pt
    ├── shape_predictor_68_face_landmarks.dat
    ├── hopenet_robust_alpha1.pkl (need for metric calculation)
    ├── WFLW_4HG.pth (need for metric calculation)
    └── arcface.pt (need for metric calculation)
  ```
## Inference
   ```sh
   python face_swap.py --source=path_to_photos/source --target=path_to_photos/target
   ``` 
   For more information please check `options/swap_options.py` or `face_swap.py -h`.
## Training 
```sh
python  -m torch.distributed.launch \
        --nproc_per_node=4 \
        --nnodes=1 \
        --node_rank=0 \
        --master_addr=localhost \
        --master_port=22221 \
        train.py --exp_dir='name of your experiment'
```
For more information please check `options/train_options.py` or `train.py -h`.