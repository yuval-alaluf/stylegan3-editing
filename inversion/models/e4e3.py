import torch
from torch import nn

from configs.paths_config import model_paths
from inversion.models.encoders import restyle_e4e_encoders
from models.stylegan3.model import SG3Generator
from utils import common


class e4e(nn.Module):

    def __init__(self, opts):
        super(e4e, self).__init__()
        self.set_opts(opts)
        # Define architecture
        self.n_styles = 16
        self.encoder = self.set_encoder()
        self.face_pool = torch.nn.AdaptiveAvgPool2d((256, 256))
        # Load weights if needed
        self.load_weights()

    def set_encoder(self):
        if self.opts.encoder_type == 'ProgressiveBackboneEncoder':
            encoder = restyle_e4e_encoders.ProgressiveBackboneEncoder(50, 'ir_se', self.n_styles, self.opts)
        elif self.opts.encoder_type == 'ResNetProgressiveBackboneEncoder':
            encoder = restyle_e4e_encoders.ResNetProgressiveBackboneEncoder(self.n_styles, self.opts)
        else:
            raise Exception(f'{self.opts.encoder_type} is not a valid encoders')
        return encoder

    def load_weights(self):
        if self.opts.checkpoint_path is not None:
            print(f'Loading ReStyle e4e from checkpoint: {self.opts.checkpoint_path}')
            ckpt = torch.load(self.opts.checkpoint_path, map_location='cpu')
            self.encoder.load_state_dict(self._get_keys(ckpt, 'encoder'), strict=True)
            self.decoder = SG3Generator(checkpoint_path=None).decoder
            self.decoder.load_state_dict(self._get_keys(ckpt, 'decoder', remove=["synthesis.input.transform"]), strict=False)
            self._load_latent_avg(ckpt)
        else:
            encoder_ckpt = self._get_encoder_checkpoint()
            self.encoder.load_state_dict(encoder_ckpt, strict=False)
            self.decoder = SG3Generator(checkpoint_path=self.opts.stylegan_weights).decoder.cuda()
            self.latent_avg = self.decoder.mapping.w_avg

    def forward(self, x, latent=None, resize=True, input_code=False, landmarks_transform=None,
                return_latents=False, return_aligned_and_unaligned=False):

        images, unaligned_images = None, None

        if input_code:
            codes = x
        else:
            codes = self.encoder(x)
            # residual step
            if x.shape[1] == 6 and latent is not None:
                # learn error with respect to previous iteration
                codes = codes + latent
            else:
                # first iteration is with respect to the avg latent code
                codes = codes + self.latent_avg.repeat(codes.shape[0], 1, 1)

        # generate the aligned images
        identity_transform = common.get_identity_transform()
        identity_transform = torch.from_numpy(identity_transform).unsqueeze(0).repeat(x.shape[0], 1, 1).cuda().float()
        self.decoder.synthesis.input.transform = identity_transform
        images = self.decoder.synthesis(codes, noise_mode='const', force_fp32=True)

        if resize:
            images = self.face_pool(images)

        # generate the unaligned image using the user-specified transforms
        if landmarks_transform is not None:
            self.decoder.synthesis.input.transform = landmarks_transform.float()   # size: [batch_size, 3, 3]
            unaligned_images = self.decoder.synthesis(codes, noise_mode='const', force_fp32=True)
            if resize:
                unaligned_images = self.face_pool(unaligned_images)

        if landmarks_transform is not None and return_aligned_and_unaligned:
            return images, unaligned_images, codes

        if return_latents:
            return images, codes
        else:
            return images

    def set_opts(self, opts):
        self.opts = opts

    def _load_latent_avg(self, ckpt, repeat=None):
        if 'latent_avg' in ckpt:
            self.latent_avg = ckpt['latent_avg'].to("cuda")
            if repeat is not None:
                self.latent_avg = self.latent_avg.repeat(repeat, 1)
        else:
            self.latent_avg = None

    def _get_encoder_checkpoint(self):
        print('Loading encoders weights from irse50!')
        encoder_ckpt = torch.load(model_paths['ir_se50'])
        # Transfer the RGB input of the irse50 network to the first 3 input channels of pSp's encoder
        if self.opts.input_nc != 3:
            shape = encoder_ckpt['input_layer.0.weight'].shape
            altered_input_layer = torch.randn(shape[0], self.opts.input_nc, shape[2], shape[3], dtype=torch.float32)
            altered_input_layer[:, :3, :, :] = encoder_ckpt['input_layer.0.weight']
            encoder_ckpt['input_layer.0.weight'] = altered_input_layer
        return encoder_ckpt

    @staticmethod
    def _get_keys(d, name, remove=[]):
        if 'state_dict' in d:
            d = d['state_dict']
        d_filt = {k[len(name) + 1:]: v for k, v in d.items()
                  if k[:len(name)] == name and k[len(name) + 1:] not in remove}
        return d_filt
