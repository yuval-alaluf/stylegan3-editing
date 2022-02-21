import torch
from torch import nn

from editing.styleclip_mapper import latent_mappers
from models.stylegan3.model import SG3Generator


def get_keys(d, name):
	if 'state_dict' in d:
		d = d['state_dict']
	d_filt = {k[len(name) + 1:]: v for k, v in d.items() if k[:len(name)] == name}
	return d_filt


class StyleCLIPMapper(nn.Module):

	def __init__(self, opts):
		super(StyleCLIPMapper, self).__init__()
		self.opts = opts
		# Define architecture
		self.mapper = self.set_mapper()
		self.decoder = SG3Generator(opts.stylegan_weights, res=opts.stylegan_size).decoder
		self.face_pool = torch.nn.AdaptiveAvgPool2d((256, 256))
		# Load weights if needed
		self.load_weights()

	def set_mapper(self):
		if self.opts.mapper_type == 'SingleMapper':
			mapper = latent_mappers.SingleMapper(self.opts)
		elif self.opts.mapper_type == 'LevelsMapper':
			mapper = latent_mappers.LevelsMapper(self.opts)
		else:
			raise Exception('{} is not a valid mapper'.format(self.opts.mapper_type))
		return mapper

	def load_weights(self):
		if self.opts.checkpoint_path is not None:
			print('Loading from checkpoint: {}'.format(self.opts.checkpoint_path))
			ckpt = torch.load(self.opts.checkpoint_path, map_location='cpu')
			self.mapper.load_state_dict(get_keys(ckpt, 'mapper'), strict=True)

	def forward(self, x, input_code=False):
		if input_code:
			codes = x
		else:
			codes = self.mapper(x)

		return codes
