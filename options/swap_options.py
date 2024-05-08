from argparse import ArgumentParser


class SwapFacePipelineOptions:

    def __init__(self):
        self.parser = ArgumentParser()
        self.initialize()

    def initialize(self):
        self.parser.add_argument('--num_seg_cls', type=int, default=12, help='Segmentation mask class number')
        self.parser.add_argument('--train_G', default=True, type=bool, help='Whether to train the model')
        self.parser.add_argument('--device', default='cuda:0', type=str, help='Which GPU(s) to use')
        self.parser.add_argument('--lap_bld', default=True, help='Whether to use Laplacian multi-band blending')
        # ================= Model =====================
        self.parser.add_argument('--out_size', type=int, default=1024, help='output image size')
        self.parser.add_argument('--fsencoder_type', type=str, default="psp", help='FS Encode type')
        self.parser.add_argument('--remaining_layer_idx', type=int, default=13,
                                 help='mask-guided style injection, i.e., K in paper')
        self.parser.add_argument('--outer_dilation', type=int, default=15, help='dilation width')
        self.parser.add_argument('--erode_radius', type=int, default=3, help='erode width')

        # ================== Pre-trained Models ==================
        self.parser.add_argument('--learn_in_w', default=False, help='Whether to learn in w space instead of w+')
        self.parser.add_argument('--start_from_latent_avg', action='store_true', default=True,
                                 help='Whether to add average latent vector to generate codes from encoder.')
        self.parser.add_argument('--output_size', default=1024, type=int, help='Output size of generator')
        self.parser.add_argument('--n_styles', default=18, type=int, help='StyleGAN layers')
        self.parser.add_argument('--checkpoint_path', default='./pretrained_ckpts/e4s/iteration_100000.pt', type=str,
                                 help='Path to E4S pre-trained model checkpoint')
        self.parser.add_argument('--faceParser_name', default='segnext', type=str,
                                 help='face parser name, [ default | segnext] is currently supported.')

        # ================== input & output ==================
        self.parser.add_argument('--source', type=str, default="input/source",
                                 help='Path to the source images')
        self.parser.add_argument('--target', type=str, default="input/target",
                                 help='Path to the target images')
        self.parser.add_argument('--output_dir', type=str, default="./output",
                                 help='Path to the output directory')
        self.parser.add_argument('--need_crop', default=True,
                                 help='Whether to do cropping and aligning source and target photos')
        self.parser.add_argument('--target_mask', type=str, default="", help='Path to the target mask')

        self.parser.add_argument('--verbose', default=False, type=bool, help='Whether to show the intermediate results')

        self.parser.add_argument('--save_only_swap', type=str, default=True,
                                 help='Whether to save only generated photo without source and target')
        self.parser.add_argument('--save_concat', type=str, default=False,
                                 help='Whether to save fused images i.e. source | target | swap')
        self.parser.add_argument('--evaluate', type=bool, default=False, help='Generate photos for evaluation')
        self.parser.add_argument('--celeba_dataset_root', default='../datasets/CelebAMask-HQ', type=str,
                                 help='CelebAMask-HQ dataset root path')

    def parse(self):
        opts = self.parser.parse_args()
        return opts
