import os
import cv2
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from models import RoadNet
from dataset import train_data, test_data, load_dataset_from_dir


### Load data in ###
data_dir = os.environ['ROADNET_DATADIR']
train_loader, test_loader = load_dataset_from_dir(data_dir, train_data, test_data, batch_size=64)

class Trainer(object):
	def __init__(self, 
		epochs=1, 
		batch_size=64, 
		save_steps=15, 
		learning_rate=0.005):
		### Some constants ###
		self.epochs = epochs
		self.batch_size = batch_size
		self.save_steps = save_steps
		self.learning_rate = learning_rate
		self.loss_weights1 = [0.5, 0.75, 1.0, 0.75, 0.5, 1.0]
		self.loss_weights2 = [0.5, 0.75, 1.0, 0.75, 1.0]

		### Check cuda availability ###
		self.device = "cpu"
		if(torch.cuda.is_available()):
			print('[INFO] Using GPU for training ... ')
			self.device = "cuda"

		self.model = RoadNet()
		self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

	def _get_bce_criterion(self, inputs):
		pass

	def train(self, train_loader, test_loader):
		self.model.train() # enter training mode
		for i in range(self.epochs):
			running_loss = 0
			for batch in train_loader:
				images, segments_gt, centerlines_gt, edges_gt = batch 

				### Forward ###
				segments, centerlines, edges = self.model(images)

				### optimizer grad to zero (start recording operations) ###
				self.optimizer.zero_grad()

				### Backward - calculate losses ###
				""" Calc segment loss : mse + bce with logits """
				loss_segment = torch.mean((torch.sigmoid(segments[-1]) - segments_gt) ** 2) * 0.5
				for out_seg, w in zip(segments, self.loss_weights1):
					### calculate beta ###
					count_neg = torch.sum(1.0 - out_seg)
					count_pos = torch.sum(out_seg)
					beta = count_neg / (count_neg + count_pos)
					pos_weight = beta / (1 - beta)
					criterion_seg = nn.BCEWithLogitsLoss(size_average=True, reduce=True, pos_weight=pos_weight)
					loss_segment += criterion_seg(out_seg, segments_gt) * (1 - beta) * w 

				loss_line = torch.mean((torch.sigmoid(centerlines[-1]) - centerlines_gt) ** 2) * 0.5
				for out_line, w in zip(centerlines, self.loss_weights2):
					### calculate beta ###
					count_neg = torch.sum(1.0 - out_line)
					count_pos = torch.sum(out_line)
					beta = count_neg / (count_neg + count_pos)
					pos_weight = beta / (1 - beta)
					criterion_line = nn.BCEWithLogitsLoss(size_average=True, reduce=True, pos_weight=pos_weight)
					loss_line += criterion_line(out_line, centerlines_gt) * (1 - beta) * w 

				
				loss_edge = torch.mean((torch.sigmoid(edges[-1]) - edges_gt) ** 2) * 0.5
				for out_edge, w in zip(edges, self.loss_weights2):
					### calculate beta ###
					count_neg = torch.sum(1.0 - out_edge)
					count_pos = torch.sum(out_edge)
					beta = count_neg / (count_neg + count_pos)
					pos_weight = beta / (1 - beta)
					criterion_edge = nn.BCEWithLogitsLoss(size_average=True, reduce=True, pos_weight=pos_weight)
					loss_edge += criterion_edge(out_edge, edges_gt) * (1 - beta) * w 

				total_loss = loss_segment + loss_line + loss_edge
				running_loss += total_loss.item()
				total_loss.backward()
				self.optimizer.step()

			print('[*] Epoch #[%d/%d], Loss = %.5f' % (i+1, epochs, running_loss / batch_size))

trainer = Trainer()
trainer.train(train_loader, test_loader)