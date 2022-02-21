from pathlib import Path
from typing import Dict, Any, List, Optional
import dataclasses
import matplotlib
import matplotlib.pyplot as plt
import torch
from torch import nn, autograd
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F

matplotlib.use('Agg')

from configs import data_configs
from criteria import id_loss, moco_loss
from criteria.lpips.lpips import LPIPS
from inversion.datasets.images_dataset import ImagesDataset
from inversion.models.e4e_modules.latent_codes_pool import LatentCodesPool
from inversion.models.e4e_modules.discriminator import LatentCodesDiscriminator
from inversion.models.encoders.restyle_e4e_encoders import ProgressiveStage
from inversion.options.e4e_train_options import e4eTrainOptions
from inversion.models.e4e3 import e4e
from utils.ranger import Ranger
from utils import common, train_utils


class Coach:
	def __init__(self, opts: e4eTrainOptions, prev_train_checkpoint: bool = None):
		self.opts = opts

		self.global_step = 0

		self.device = 'cuda:0'
		self.opts.device = self.device

		# Initialize network
		self.net = e4e(self.opts).to(self.device)

		if self.net.latent_avg is None:
			self.net.latent_avg = self.net.decoder.mapping.w_avg

		# get the image corresponding to the latent average
		self.avg_image = self.net(self.net.latent_avg.repeat(16, 1).unsqueeze(0),
								  input_code=True,
								  return_latents=False)[0]
		self.avg_image = self.avg_image.to(self.device).float().detach()
		common.tensor2im(self.avg_image).save(self.opts.exp_dir / 'avg_image.jpg')

		# Initialize loss
		if self.opts.id_lambda > 0 and self.opts.moco_lambda > 0:
			raise ValueError('Both ID and MoCo loss have lambdas > 0! Please select only one to have non-zero lambda!')
		self.mse_loss = nn.MSELoss().to(self.device).eval()
		if self.opts.lpips_lambda > 0:
			self.lpips_loss = LPIPS(net_type='alex').to(self.device).eval()
		if self.opts.id_lambda > 0:
			self.id_loss = id_loss.IDLoss().to(self.device).eval()
		if self.opts.moco_lambda > 0:
			self.moco_loss = moco_loss.MocoLoss()

		# Initialize optimizer
		self.optimizer = self.configure_optimizers()

		# Initialize discriminator
		if self.opts.w_discriminator_lambda > 0:
			self.discriminator = LatentCodesDiscriminator(512, 4).to(self.device)
			self.discriminator_optimizer = torch.optim.Adam(list(self.discriminator.parameters()),
															lr=opts.w_discriminator_lr)
			self.real_w_pool = LatentCodesPool(self.opts.w_pool_size)
			self.fake_w_pool = LatentCodesPool(self.opts.w_pool_size)

		# Initialize dataset
		self.train_dataset, self.test_dataset = self.configure_datasets()
		self.train_dataloader = DataLoader(self.train_dataset,
										   batch_size=self.opts.batch_size,
										   shuffle=True,
										   num_workers=int(self.opts.workers),
										   drop_last=True)
		self.test_dataloader = DataLoader(self.test_dataset,
										  batch_size=self.opts.test_batch_size,
										  shuffle=False,
										  num_workers=int(self.opts.test_workers),
										  drop_last=False)

		# Initialize logger
		log_dir = opts.exp_dir / 'logs'
		log_dir.mkdir(exist_ok=True, parents=True)
		self.logger = SummaryWriter(log_dir=log_dir)

		# Initialize checkpoint dir
		self.checkpoint_dir = opts.exp_dir / 'checkpoints'
		self.checkpoint_dir.mkdir(exist_ok=True, parents=True)
		self.best_val_loss = None
		if self.opts.save_interval is None:
			self.opts.save_interval = self.opts.max_steps

		if prev_train_checkpoint is not None:
			self.load_from_train_checkpoint(prev_train_checkpoint)

	def load_from_train_checkpoint(self, ckpt: Dict[str, Any]):
		print('Loading previous training data...')
		self.global_step = ckpt['global_step'] + 1
		self.best_val_loss = ckpt['best_val_loss']
		self.net.load_state_dict(ckpt['state_dict'])
		if self.opts.w_discriminator_lambda > 0:
			self.discriminator.load_state_dict(ckpt['discriminator_state_dict'])
			self.discriminator_optimizer.load_state_dict(ckpt['discriminator_optimizer_state_dict'])
		if self.opts.progressive_steps:
			self.check_for_progressive_training_update(is_resume_from_ckpt=True)
		print(f'Resuming training from step {self.global_step}')

	def compute_discriminator_loss(self, x: torch.tensor):
		avg_image_for_batch = self.avg_image.unsqueeze(0).repeat(x.shape[0], 1, 1, 1)
		avg_image_for_batch.clone().detach().requires_grad_(True)
		x_input = torch.cat([x, avg_image_for_batch], dim=1)
		disc_loss_dict = {}
		if self.is_training_discriminator():
			disc_loss_dict = self.train_discriminator(x_input)
		return disc_loss_dict

	def perform_train_iteration_on_batch(self, x: torch.tensor, y: torch.tensor):
		y_hat, latent = None, None
		loss_dict, id_logs = None, None
		y_hats = {idx: [] for idx in range(x.shape[0])}
		for iter in range(self.opts.n_iters_per_batch):
			if iter == 0:
				avg_image_for_batch = self.avg_image.unsqueeze(0).repeat(x.shape[0], 1, 1, 1)
				x_input = torch.cat([x, avg_image_for_batch], dim=1)
				y_hat, latent = self.net.forward(x_input, latent=None, return_latents=True)
			else:
				y_hat_clone = y_hat.clone().detach().requires_grad_(True)
				latent_clone = latent.clone().detach().requires_grad_(True)
				x_input = torch.cat([x, y_hat_clone], dim=1)
				y_hat, latent = self.net.forward(x_input, latent=latent_clone, return_latents=True)

			loss, loss_dict, id_logs = self.calc_loss(x, y, y_hat, latent)
			loss.backward()
			# store intermediate outputs
			for idx in range(x.shape[0]):
				y_hats[idx].append([y_hat[idx], id_logs[idx]['diff_target']])

		return y_hats, loss_dict, id_logs

	def train(self):
		self.net.train()
		if self.opts.progressive_steps:
			self.check_for_progressive_training_update()

		while self.global_step < self.opts.max_steps:

			self.optimizer.zero_grad()

			for batch_idx, batch in enumerate(self.train_dataloader):
				x, y = batch
				x, y = x.to(self.device).float(), y.to(self.device).float()

				disc_loss_dict = self.compute_discriminator_loss(x)

				y_hats, encoder_loss_dict, id_logs = self.perform_train_iteration_on_batch(x, y)

				# only update the optimizer after we've seen 8 examples
				if batch_idx % int(8 / self.opts.batch_size) == 0:
					self.optimizer.step()
					self.optimizer.zero_grad()

				loss_dict = {**disc_loss_dict, **encoder_loss_dict}

				# Logging related
				if self.global_step % self.opts.image_interval == 0 or (self.global_step < 1000 and self.global_step % 25 == 0):
					self.parse_and_log_images(id_logs, x, y, y_hats, title='images/train')

				if self.global_step % self.opts.board_interval == 0:
					self.print_metrics(loss_dict, prefix='train')
					self.log_metrics(loss_dict, prefix='train')

				# Validation related
				val_loss_dict = None
				if self.global_step % self.opts.val_interval == 0 or self.global_step == self.opts.max_steps:
					val_loss_dict = self.validate()
					if val_loss_dict and (self.best_val_loss is None or val_loss_dict['loss'] < self.best_val_loss):
						self.best_val_loss = val_loss_dict['loss']
						self.checkpoint_me(val_loss_dict, is_best=True)

				if self.global_step % self.opts.save_interval == 0 or self.global_step == self.opts.max_steps:
					if val_loss_dict is not None:
						self.checkpoint_me(val_loss_dict, is_best=False)
					else:
						self.checkpoint_me(loss_dict, is_best=False)

				if self.global_step == self.opts.max_steps:
					print('OMG, finished training!')
					break

				self.global_step += 1
				if self.opts.progressive_steps:
					self.check_for_progressive_training_update()

	def perform_val_iteration_on_batch(self, x: torch.tensor, y: torch.tensor):
		y_hat, latent = None, None
		cur_loss_dict, id_logs = None, None
		y_hats = {idx: [] for idx in range(x.shape[0])}
		for iter in range(self.opts.n_iters_per_batch):
			if iter == 0:
				avg_image_for_batch = self.avg_image.unsqueeze(0).repeat(x.shape[0], 1, 1, 1)
				x_input = torch.cat([x, avg_image_for_batch], dim=1)
			else:
				x_input = torch.cat([x, y_hat], dim=1)

			y_hat, latent = self.net.forward(x_input, latent=latent, return_latents=True)

			loss, cur_loss_dict, id_logs = self.calc_loss(x, y, y_hat, latent)
			# store intermediate outputs
			for idx in range(x.shape[0]):
				y_hats[idx].append([y_hat[idx], id_logs[idx]['diff_target']])

		return y_hats, cur_loss_dict, id_logs

	def validate(self):
		self.net.eval()
		agg_loss_dict = []
		for batch_idx, batch in enumerate(self.test_dataloader):
			if self.opts.max_val_batches is not None and batch_idx > self.opts.max_val_batches:
				break

			x, y = batch
			x, y = x.to(self.device).float(), y.to(self.device).float()

			# validate discriminator on batch
			avg_image_for_batch = self.avg_image.unsqueeze(0).repeat(x.shape[0], 1, 1, 1)
			x_input = torch.cat([x, avg_image_for_batch], dim=1)
			cur_disc_loss_dict = {}
			if self.is_training_discriminator():
				cur_disc_loss_dict = self.validate_discriminator(x_input)

			# validate encoder on batch
			with torch.no_grad():
				y_hats, cur_enc_loss_dict, id_logs = self.perform_val_iteration_on_batch(x, y)

			cur_loss_dict = {**cur_disc_loss_dict, **cur_enc_loss_dict}
			agg_loss_dict.append(cur_loss_dict)

			# Logging related
			self.parse_and_log_images(id_logs, x, y, y_hats, title='images/test', subscript=f'{batch_idx:04d}')

			# For first step just do sanity test on small amount of data
			if self.global_step == 0 and batch_idx >= 4:
				self.net.train()
				return None  # Do not log, inaccurate in first batch

		loss_dict = train_utils.aggregate_loss_dict(agg_loss_dict)
		self.log_metrics(loss_dict, prefix='test')
		self.print_metrics(loss_dict, prefix='test')

		self.net.train()
		return loss_dict

	def checkpoint_me(self, loss_dict: Dict[str, float], is_best: bool):
		save_name = 'best_model.pt' if is_best else f'iteration_{self.global_step}.pt'
		save_dict = self.__get_save_dict()
		checkpoint_path = self.checkpoint_dir / save_name
		torch.save(save_dict, checkpoint_path)
		with open(self.checkpoint_dir / 'timestamp.txt', 'a') as f:
			if is_best:
				f.write(f'**Best**: Step - {self.global_step}, '
						f'Loss - {self.best_val_loss:.3f} \n{loss_dict}\n')
			else:
				f.write(f'Step - {self.global_step}, \n{loss_dict}\n')

	def configure_optimizers(self):
		params = list(self.net.encoder.parameters())
		if self.opts.train_decoder:
			params += list(self.net.decoder.parameters())
		else:
			self.requires_grad(self.net.decoder, False)
		if self.opts.optim_name == 'adam':
			optimizer = torch.optim.Adam(params, lr=self.opts.learning_rate)
		else:
			optimizer = Ranger(params, lr=self.opts.learning_rate)
		return optimizer

	def configure_datasets(self):
		if self.opts.dataset_type not in data_configs.DATASETS.keys():
			raise Exception(f'{self.opts.dataset_type} is not a valid dataset_type')
		print(f'Loading dataset for {self.opts.dataset_type}')
		dataset_args = data_configs.DATASETS[self.opts.dataset_type]
		transforms_dict = dataset_args['transforms'](self.opts).get_transforms()
		train_dataset = ImagesDataset(source_root=dataset_args['train_source_root'],
									  target_root=dataset_args['train_target_root'],
									  source_transform=transforms_dict['transform_source'],
									  target_transform=transforms_dict['transform_gt_train'])
		test_dataset = ImagesDataset(source_root=dataset_args['test_source_root'],
									 target_root=dataset_args['test_target_root'],
									 source_transform=transforms_dict['transform_source'],
									 target_transform=transforms_dict['transform_test'])
		print(f"Number of training samples: {len(train_dataset)}")
		print(f"Number of test samples: {len(test_dataset)}")
		return train_dataset, test_dataset

	def calc_loss(self, x: torch.tensor, y: torch.tensor, y_hat: torch.tensor, latent: torch.tensor):
		loss_dict = {}
		loss = 0.0
		id_logs = None

		# Adversarial loss
		if self.is_training_discriminator():
			loss_disc = self.compute_adversarial_loss(latent, loss_dict)
			loss += self.opts.w_discriminator_lambda * loss_disc

		# delta regularization loss
		if self.opts.progressive_steps and self.net.encoder.progressive_stage.value != 16:
			total_delta_loss = self.compute_delta_regularization_loss(latent, loss_dict)
			loss += self.opts.delta_norm_lambda * total_delta_loss

		# similarity losses
		if self.opts.id_lambda > 0:
			loss_id, sim_improvement, id_logs = self.id_loss(y_hat, y, x)
			loss_dict['loss_id'] = float(loss_id)
			loss_dict['id_improve'] = float(sim_improvement)
			loss += loss_id * self.opts.id_lambda
		if self.opts.l2_lambda > 0:
			loss_l2 = F.mse_loss(y_hat, y)
			loss_dict['loss_l2'] = float(loss_l2)
			loss += loss_l2 * self.opts.l2_lambda
		if self.opts.lpips_lambda > 0:
			loss_lpips = self.lpips_loss(y_hat, y)
			loss_dict['loss_lpips'] = float(loss_lpips)
			loss += loss_lpips * self.opts.lpips_lambda
		if self.opts.moco_lambda > 0:
			loss_moco, sim_improvement, id_logs = self.moco_loss(y_hat, y, x)
			loss_dict['loss_moco'] = float(loss_moco)
			loss_dict['id_improve'] = float(sim_improvement)
			loss += loss_moco * self.opts.moco_lambda

		loss_dict['loss'] = float(loss)
		return loss, loss_dict, id_logs

	def compute_adversarial_loss(self, latent: torch.tensor, loss_dict: Dict[str, float]):
		loss_disc = 0.
		dims_to_discriminate = self.get_dims_to_discriminate() if self.is_progressive_training() else list(range(16))
		for i in dims_to_discriminate:
			w = latent[:, i, :]
			fake_pred = self.discriminator(w)
			loss_disc += F.softplus(-fake_pred).mean()
		loss_disc /= len(dims_to_discriminate)
		loss_dict['encoder_discriminator_loss'] = float(loss_disc)
		return loss_disc

	def compute_delta_regularization_loss(self, latent: torch.tensor, loss_dict: Dict[str, float]):
		total_delta_loss = 0
		deltas_latent_dims = self.net.encoder.get_deltas_starting_dimensions()
		first_w = latent[:, 0, :]
		for i in range(1, self.net.encoder.progressive_stage.value + 1):
			curr_dim = deltas_latent_dims[i]
			delta = latent[:, curr_dim, :] - first_w
			delta_loss = torch.norm(delta, self.opts.delta_norm, dim=1).mean()
			loss_dict[f"delta{i}_loss"] = float(delta_loss)
			total_delta_loss += delta_loss
		loss_dict['total_delta_loss'] = float(total_delta_loss)
		return total_delta_loss

	def log_metrics(self, metrics_dict: Dict[str, float], prefix: str):
		for key, value in metrics_dict.items():
			self.logger.add_scalar(f'{prefix}/{key}', value, self.global_step)

	def print_metrics(self, metrics_dict: Dict[str, float], prefix: str):
		print(f'Metrics for {prefix}, step {self.global_step}')
		for key, value in metrics_dict.items():
			print(f'\t{key} = ', value)

	def parse_and_log_images(self, id_logs: List[Dict], x: torch.tensor, y: torch.tensor, y_hat: torch.tensor,
							 title: str, subscript: Optional[str] = None, display_count: int = 2):
		im_data = []
		for i in range(display_count):
			if type(y_hat) == dict:
				output_face = [
					[common.tensor2im(y_hat[i][iter_idx][0]), y_hat[i][iter_idx][1]]
					for iter_idx in range(len(y_hat[i]))
				]
			else:
				output_face = [common.tensor2im(y_hat[i])]
			cur_im_data = {
				'input_face': common.tensor2im(x[i]),
				'target_face': common.tensor2im(y[i]),
				'output_face': output_face,
			}
			if id_logs is not None:
				for key in id_logs[i]:
					cur_im_data[key] = id_logs[i][key]
			im_data.append(cur_im_data)
		self.log_images(title, im_data=im_data, subscript=subscript)

	def log_images(self, name: str, im_data: List[Dict], subscript: Optional[str] = None, log_latest: bool = False):
		fig = train_utils.vis_faces(im_data)
		step = self.global_step
		if log_latest:
			step = 0
		if subscript:
			path = Path(self.logger.log_dir) / name / f'{subscript}_{step:04d}.jpg'
		else:
			path = Path(self.logger.log_dir) / name / f'{step:04d}.jpg'
		path.parent.mkdir(exist_ok=True, parents=True)
		fig.savefig(path)
		plt.close(fig)

	def __get_save_dict(self):
		save_dict = {
			'state_dict': self.net.state_dict(),
			'opts': dataclasses.asdict(self.opts),
			'global_step': self.global_step,
			'optimizer': self.optimizer.state_dict(),
			'best_val_loss': self.best_val_loss,
			'latent_avg': self.net.latent_avg
		}
		if self.opts.w_discriminator_lambda > 0:
			save_dict['discriminator_state_dict'] = self.discriminator.state_dict()
			save_dict['discriminator_optimizer_state_dict'] = self.discriminator_optimizer.state_dict()
		return save_dict

	# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Util Functions ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

	def get_dims_to_discriminate(self):
		deltas_starting_dimensions = self.net.encoder.get_deltas_starting_dimensions()
		return deltas_starting_dimensions[:self.net.encoder.progressive_stage.value + 1]

	def is_progressive_training(self):
		return self.opts.progressive_steps is not None

	def check_for_progressive_training_update(self, is_resume_from_ckpt: bool = False):
		for i in range(len(self.opts.progressive_steps)):
			if is_resume_from_ckpt and self.global_step >= self.opts.progressive_steps[i]:  # Case checkpoint
				self.net.encoder.set_progressive_stage(ProgressiveStage(i))
			if self.global_step == self.opts.progressive_steps[i]:  # Case training reached progressive step
				self.net.encoder.set_progressive_stage(ProgressiveStage(i))

	@staticmethod
	def requires_grad(model: nn.Module, flag: bool = True):
		for p in model.parameters():
			p.requires_grad = flag
			
	# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Discriminator ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

	def is_training_discriminator(self):
		return self.opts.w_discriminator_lambda > 0

	@staticmethod
	def discriminator_loss(real_pred: torch.tensor, fake_pred: torch.tensor, loss_dict: Dict[str, float]):
		real_loss = F.softplus(-real_pred).mean()
		fake_loss = F.softplus(fake_pred).mean()
		loss_dict['d_real_loss'] = float(real_loss)
		loss_dict['d_fake_loss'] = float(fake_loss)
		return real_loss + fake_loss

	@staticmethod
	def discriminator_r1_loss(real_pred: torch.tensor, real_w: torch.tensor):
		grad_real, = autograd.grad(outputs=real_pred.sum(), inputs=real_w, create_graph=True)
		grad_penalty = grad_real.pow(2).reshape(grad_real.shape[0], -1).sum(1).mean()
		return grad_penalty

	def train_discriminator(self, x: torch.tensor):
		loss_dict = {}
		self.requires_grad(self.discriminator, True)

		with torch.no_grad():
			real_w, fake_w = self.sample_real_and_fake_latents(x)
		real_pred = self.discriminator(real_w)
		fake_pred = self.discriminator(fake_w)
		loss = self.discriminator_loss(real_pred, fake_pred, loss_dict)
		loss_dict['discriminator_loss'] = float(loss)

		self.discriminator_optimizer.zero_grad()
		loss.backward()
		self.discriminator_optimizer.step()

		# r1 regularization
		d_regularize = self.global_step % self.opts.d_reg_every == 0
		if d_regularize:
			real_w = real_w.detach()
			real_w.requires_grad = True
			real_pred = self.discriminator(real_w)
			r1_loss = self.discriminator_r1_loss(real_pred, real_w)

			self.discriminator.zero_grad()
			r1_final_loss = self.opts.r1 / 2 * r1_loss * self.opts.d_reg_every + 0 * real_pred[0]
			r1_final_loss.backward()
			self.discriminator_optimizer.step()
			loss_dict['discriminator_r1_loss'] = float(r1_final_loss)

		# Reset to previous state
		self.requires_grad(self.discriminator, False)

		return loss_dict

	def validate_discriminator(self, x: torch.tensor):
		with torch.no_grad():
			loss_dict = {}
			real_w, fake_w = self.sample_real_and_fake_latents(x)
			real_pred = self.discriminator(real_w)
			fake_pred = self.discriminator(fake_w)
			loss = self.discriminator_loss(real_pred, fake_pred, loss_dict)
			loss_dict['discriminator_loss'] = float(loss)
			return loss_dict

	def sample_real_and_fake_latents(self, x: torch.tensor):
		sample_z = torch.randn(x.shape[0], 512, device=self.device)
		c = torch.zeros([x.shape[0], self.net.decoder.c_dim], device=self.device)  # unconditional setting
		real_w = self.net.decoder.mapping(sample_z, c)
		fake_w = self.net.encoder(x)
		if self.is_progressive_training():  # When progressive training, feed only unique w's
			dims_to_discriminate = self.get_dims_to_discriminate()
			fake_w = fake_w[:, dims_to_discriminate, :]
		if self.opts.use_w_pool:
			real_w = self.real_w_pool.query(real_w)
			fake_w = self.fake_w_pool.query(fake_w)
		if fake_w.ndim == 3:
			fake_w = fake_w[:, 0, :]
		return real_w, fake_w
