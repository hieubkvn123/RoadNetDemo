import os
import cv2
import imageio
import numpy as np
from argparse import ArgumentParser

import torch
import torch.nn as nn
import torch.optim as optim

from models import RoadNet
from dataset import train_data, test_data, load_dataset_from_dir

### Config argument parser ###
parser = ArgumentParser()
parser.add_argument("--epochs", required=False, default=200, 
	help='Number of training iterations')
parser.add_argument("--batch_size", required=False, default=64,
	help='Number of images to feed to the network per batch')
parser.add_argument("--save_steps", required=False, default=15,
	help='Number of training iterations interval to save model')
parser.add_argument("--vis_steps", required=False, default=5,
	help='Number of training iterations interval to visualize the map result')
parser.add_argument("--learning_rate", required=False, default=0.000005,
	help='Learning rate for optimization')
parser.add_argument('--checkpoint_path', required=False, default='./models/roadnet_weights.pt', 
	help='Weights checkpoint path')
parser.add_argument('--vis_dir', required=False, default='./outputs',
		help='Path to output visualization output')
args = vars(parser.parse_args())

epochs = int(args['epochs'])
batch_size = int(args['batch_size'])
save_steps = int(args['save_steps'])
vis_steps = int(args['vis_steps'])
learning_rate = float(args['learning_rate'])
checkpoint_path = args['checkpoint_path']
vis_dir = args['vis_dir']

### Load data in ###
data_dir = os.environ['ROADNET_DATADIR']
train_loader, test_loader = load_dataset_from_dir(data_dir, train_data, test_data, batch_size=64)

class Trainer(object):
	def __init__(self, 
		epochs=1, 
		batch_size=64, 
		save_steps=15, 
		vis_steps=5,
		vis_dir=None,
		checkpoint_path=None,
		learning_rate=0.000005):
		### Some constants ###
		self.epochs = epochs
		self.batch_size = batch_size
		self.save_steps = save_steps
		self.vis_steps = vis_steps
		self.learning_rate = learning_rate
		self.vis_dir = vis_dir
		self.checkpoint_path = checkpoint_path
		self.loss_weights1 = [0.5, 0.75, 1.0, 0.75, 0.5, 1.0]
		self.loss_weights2 = [0.5, 0.75, 1.0, 0.75, 1.0]

		### Check cuda availability ###
		self.device = "cpu"
		if(torch.cuda.is_available()):
			print('[INFO] Using GPU for training ... ')
			self.device = "cuda"

		self.model = RoadNet().to(self.device)
		self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

	def _viz_testing_map(self, epoch, image_file):
		self.model.eval() ### Enter evaluation mode ###
		image = cv2.imread(image_file)

		### Crop the images to pieces ###
		H, W = image.shape[0], image.shape[1]

		### Get the ratio to resize ###
		ratio_h = int(H/self.model.input_shape[0])
		ratio_w = int(W/self.model.input_shape[1])

		resize_dimensions = (self.model.input_shape[1] * ratio_w, self.model.input_shape[0] * ratio_h)
		image = cv2.resize(image, resize_dimensions)

		### Divide the image into grid cells ###
		full_image = None
		for i in range(ratio_h):
			horizontal_image = None
			for j in range(ratio_w):
				crop_image = image[i*self.model.input_shape[0]:(i+1)*self.model.input_shape[0], j*self.model.input_shape[1]:(j+1)*self.model.input_shape[1]]
				image_tensor = np.array([crop_image]).reshape(-1, 3, self.model.input_shape[0], self.model.input_shape[1])
				image_tensor = torch.from_numpy(image_tensor).float().to(self.device)
				segments, centerlines, edges = self.model(image_tensor)

				segment = torch.sigmoid(segments[0][-1])
				segment = segment.cpu().detach().numpy().reshape(self.model.input_shape[0], self.model.input_shape[1], 1)
				segment[segment > 0.5] = 1
				segment[segment <= 0.5] = 0
				
				if(j == 0):
					horizontal_image = segment
				else:
					horizontal_image = cv2.hconcat([horizontal_image, segment])

			if(i == 0):
				full_image = horizontal_image
			else:
				full_image = cv2.vconcat([full_image, horizontal_image])

		full_image = full_image * 255.0
		full_image = full_image.astype(np.uint8)
		file_name = os.path.join(self.vis_dir, 'result_epoch_{}.jpg'.format(epoch))
		# print(full_image)
		cv2.imwrite(file_name, full_image)

	def train(self, train_loader, test_loader):
		### Print a summary of model architecture ###
		print('----------------------- Segmentation module -----------------------')
		print(self.model._segment_net)

		print('----------------------- Centerline module -----------------------')
		print(self.model._centerline_net)

		print('----------------------- Edge module -----------------------')
		print(self.model._edge_net)
		### Check if the checkpoint dir is there ###
		checkpoint_dir  = os.path.dirname(self.checkpoint_path)
		if(not os.path.exists(checkpoint_dir)):
			print('[INFO] Creating checkpoint directory ... ')
			os.mkdir(checkpoint_dir)

		if(self.vis_dir is not None and not os.path.exists(self.vis_dir)):
			print('[INFO] Creating visualization directory ... ')
			os.mkdir(self.vis_dir)

		### Check if there is a checkpoint ###
		if(os.path.exists(self.checkpoint_path)):
			print('[INFO] Checkpoint exists, loading weights into model and optimizer ... ')
			checkpoint = torch.load(self.checkpoint_path)
			self.model.load_state_dict(checkpoint['model_state_dict'])
			self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
		else:
			print('[INFO] Training from scratch ...')

		for i in range(self.epochs):
			self.model.train() # enter training mode
			running_loss = 0
			running_loss_seg = 0
			running_loss_line = 0
			running_loss_edge = 0
			for batch_id, batch in enumerate(train_loader):
				images, segments_gt, centerlines_gt, edges_gt = batch 
				images = images.to(self.device)
				segments_gt = segments_gt.to(self.device)
				centerlines_gt = centerlines_gt.to(self.device)
				edges_gt = edges_gt.to(self.device)

				### Forward ###
				segments, centerlines, edges = self.model(images)
				if(batch_id == 0):
					#print("Centerline : ", centerlines[-1].cpu().detach().numpy())
					#print()
					#print("Edge : ", edges[-1].cpu().detach().numpy())
					#	print(segments[-1].cpu().detach().numpy())
					print("Centerline ground truth : ", centerlines_gt.cpu().detach().numpy())
					print()
					print("Edge ground truth : ", edges_gt.cpu().detach().numpy())

				### optimizer grad to zero (start recording operations) ###
				self.optimizer.zero_grad()

				### Backward - calculate losses ###
				""" Calc segment loss : mse + bce with logits """
				loss_segment = torch.mean((torch.sigmoid(segments[-1]) - segments_gt) ** 2) * 0.5
				for out_seg, w in zip(segments, self.loss_weights1):
					### calculate beta ###
					# print(out_seg.cpu().detach().numpy())
					count_neg = torch.sum(1.0 - out_seg)
					count_pos = torch.sum(out_seg)
					beta = count_neg / (count_neg + count_pos)
					pos_weight = beta / (1 - beta)
					pos_weight = pos_weight.detach() 
					criterion_seg = nn.BCEWithLogitsLoss(reduction='mean', pos_weight=pos_weight)
					loss_segment += criterion_seg(out_seg, segments_gt) * (1 - beta) * w 

				loss_line = torch.mean((torch.sigmoid(centerlines[-1]) - centerlines_gt) ** 2) * 0.5
				for out_line, w in zip(centerlines, self.loss_weights2):
					### calculate beta ###
					count_neg = torch.sum(1.0 - out_line)
					count_pos = torch.sum(out_line)
					beta = count_neg / (count_neg + count_pos)
					pos_weight = beta / (1 - beta)
					print(beta)
					pos_weight = pos_weight.detach()
					criterion_line = nn.BCEWithLogitsLoss(reduction='mean', pos_weight=pos_weight)
					loss_line += criterion_line(out_line, centerlines_gt) * (1 - beta) * w 

				
				loss_edge = torch.mean((torch.sigmoid(edges[-1]) - edges_gt) ** 2) * 0.5
				for out_edge, w in zip(edges, self.loss_weights2):
					### calculate beta ###
					count_neg = torch.sum(1.0 - out_edge)
					count_pos = torch.sum(out_edge)
					beta = count_neg / (count_neg + count_pos)
					pos_weight = beta / (1 - beta)
					pos_weight = pos_weight.detach()
					criterion_edge = nn.BCEWithLogitsLoss(reduction='mean', pos_weight=pos_weight)
					loss_edge += criterion_edge(out_edge, edges_gt) * (1 - beta) * w 

				total_loss = loss_segment + loss_line + loss_edge
				running_loss += total_loss.item()
				running_loss_seg += loss_segment.item()
				running_loss_line += loss_line.item()
				running_loss_edge += loss_edge.item()

				print('[*]\tEpochs #[%d/%d], Batch #%d, Running loss = %.5f' % 
					(i+1,
					self.epochs,
					batch_id + 1, 
					running_loss / (batch_id + 1)))
				total_loss.backward()
				self.optimizer.step()

			### Save the model if checkpoint path is not None ###
			if((i+1) % self.save_steps == 0 and self.checkpoint_path is not None):
				print('[INFO] Saving checkpoint to %s' % self.checkpoint_path)
				torch.save({
					'epoch' : i+1,
					'model_state_dict' : self.model.state_dict(),
					'optimizer_state_dict' : self.optimizer.state_dict(),
					'loss' : running_loss / self.batch_size
				}, self.checkpoint_path)

			if((i+1) % self.vis_steps == 0 and self.vis_dir is not None):
				print('[INFO] Visualizing output to dir %s' % self.vis_dir )
				self._viz_testing_map(i+1, os.path.join(os.environ['ROADNET_DATADIR'], '1', 'Ottawa-1.tif'))

			print('[*] Epoch #[%d/%d], Loss = %.5f, Loss segment = %.5f, Loss line = %.5f, Loss edge = %.5f' % 
				(i+1, self.epochs, 
				running_loss / self.batch_size,
				running_loss_seg / self.batch_size,
				running_loss_line / self.batch_size,
				running_loss_edge / self.batch_size))

trainer = Trainer(epochs=epochs, 
	batch_size=batch_size, 
	save_steps=save_steps, 
	vis_steps=vis_steps, 
	vis_dir=vis_dir,
	checkpoint_path=checkpoint_path,
	learning_rate=learning_rate)
trainer.train(train_loader, test_loader)
