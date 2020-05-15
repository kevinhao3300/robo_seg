import torch
import torchvision
import torchvision.transforms as transforms
import os
import imageio
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from torch.utils.data import Dataset, DataLoader

classes = ['AG','AH','GA','GH','HA','HG']

def conv2D_output_size(img_size, padding, kernel_size, stride):
	# compute output shape of conv2D
	outshape = (np.floor((img_size[0] + 2 * padding[0] - (kernel_size[0] - 1) - 1) / stride[0] + 1).astype(int),
	            np.floor((img_size[1] + 2 * padding[1] - (kernel_size[1] - 1) - 1) / stride[1] + 1).astype(int))
	return outshape

class CNN(nn.Module):
	def __init__(self):
		super(CNN, self).__init__()
		drop_p = 0.3
		self.img_x = 256
		self.img_y = 256
		self.CNN_embed_dim = 300

		self.ch1, self.ch2, self.ch3, self.ch4 = 32, 64, 128, 256
		self.k1, self.k2, self.k3, self.k4 = (5, 5), (3, 3), (3, 3), (3, 3)      # 2d kernal size
		self.s1, self.s2, self.s3, self.s4 = (2, 2), (2, 2), (2, 2), (2, 2)      # 2d strides
		self.pd1, self.pd2, self.pd3, self.pd4 = (0, 0), (0, 0), (0, 0), (0, 0)  # 2d padding

		self.conv1_outshape = conv2D_output_size((self.img_x, self.img_y), self.pd1, self.k1, self.s1)  # Conv1 output shape
		self.conv2_outshape = conv2D_output_size(self.conv1_outshape, self.pd2, self.k2, self.s2)
		self.conv3_outshape = conv2D_output_size(self.conv2_outshape, self.pd3, self.k3, self.s3)
		self.conv4_outshape = conv2D_output_size(self.conv3_outshape, self.pd4, self.k4, self.s4)

		self.fc_hidden1, self.fc_hidden2 = 512, 512
		self.drop_p = drop_p

		self.conv1 = nn.Sequential(
			nn.Conv2d(in_channels=3, out_channels=self.ch1, kernel_size=self.k1, stride=self.s1, padding=self.pd1),
			nn.BatchNorm2d(self.ch1, momentum=0.01),
			nn.ReLU(inplace=True),                      
			nn.MaxPool2d(kernel_size=2)
		)
		self.conv2 = nn.Sequential(
			nn.Conv2d(in_channels=self.ch1, out_channels=self.ch2, kernel_size=self.k2, stride=self.s2, padding=self.pd2),
			nn.BatchNorm2d(self.ch2, momentum=0.01),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(kernel_size=2)
		)

		self.conv3 = nn.Sequential(
			nn.Conv2d(in_channels=self.ch2, out_channels=self.ch3, kernel_size=self.k3, stride=self.s3, padding=self.pd3),
			nn.BatchNorm2d(self.ch3, momentum=0.01),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(kernel_size=2)
		)

		self.conv4 = nn.Sequential(
			nn.Conv2d(in_channels=self.ch3, out_channels=self.ch4, kernel_size=self.k4, stride=self.s4, padding=self.pd4),
			nn.BatchNorm2d(self.ch4, momentum=0.01),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(kernel_size=2)
		)

		self.drop = nn.Dropout2d(self.drop_p)
		self.pool = nn.MaxPool2d(2)
		self.fc1 = nn.Linear(self.ch4 * self.conv4_outshape[0] * self.conv4_outshape[1], self.fc_hidden1)   # fully connected layer, output k classes
		self.fc2 = nn.Linear(self.fc_hidden1, self.fc_hidden2)
		self.fc3 = nn.Linear(self.fc_hidden2, self.CNN_embed_dim)   # output = CNN embedding latent variables
    
	def forward(self, x_3d):
		cnn_embed_seq = []
		for t in range(x_3d.size(1)):
		    # CNNs
		    x = self.conv1(x_3d[:, t, :, :, :])
		    x = self.conv2(x)
		    x = self.conv3(x)
		    x = self.conv4(x)
		    x = x.view(x.size(0), -1)           # flatten the output of conv

		    # FC layers
		    x = F.relu(self.fc1(x))
		    # x = F.dropout(x, p=self.drop_p, training=self.training)
		    x = F.relu(self.fc2(x))
		    x = F.dropout(x, p=self.drop_p, training=self.training)
		    x = self.fc3(x)
		    cnn_embed_seq.append(x)

		# swap time and sample dim such that (sample dim, time dim, CNN latent dim)
		cnn_embed_seq = torch.stack(cnn_embed_seq, dim=0).transpose_(0, 1)
		# cnn_embed_seq: shape=(batch, time_step, input_size)

		return cnn_embed_seq

class AGH_Dataset(Dataset):
	def __init__(self, dir_name):
		self.dir = dir_name
		self.files = os.listdir(dir_name)

	def __len__(self):
		return len(self.files)

	def __getitem__ (self, idx):
		reader = imageio.get_reader(self.dir + '/' + self.files[idx])
		arr = []
		for i, im in enumerate(reader):
			arr.append(np.array(im))

		sample = {'video': np.array(arr), 'label': classes.index(self.files[idx][:2])}
		return sample


if __name__ == '__main__':
	num_epochs = 2
	trainset = AGH_Dataset('data/small')
	trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)
	model = CNN()
	with torch.no_grad():
		for idx, m in enumerate(trainloader):
			X = m['video']
			y = m['label']
			out = model.forward(X)

	# net = CNN()
	# criterion = nn.CrossEntropyLoss()
	# optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
	# for epoch in range(num_epochs):  

	#     running_loss = 0.0
	#     for i, data in enumerate(trainloader, 0):
	#         # get the inputs; data is a list of [inputs, labels]
	#         inputs, labels = data

	#         # zero the parameter gradients
	#         optimizer.zero_grad()

	#         # forward + backward + optimize
	#         outputs = net(inputs)
	#         loss = criterion(outputs, labels)
	#         loss.backward()
	#         optimizer.step()

	#         # print statistics
	#         running_loss += loss.item()
	#         if i % 2000 == 1999:    # print every 2000 mini-batches
	#             print('[%d, %5d] loss: %.3f' %
	#                   (epoch + 1, i + 1, running_loss / 2000))
	#             running_loss = 0.0

	# print('Finished Training')

