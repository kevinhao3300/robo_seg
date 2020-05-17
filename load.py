import torch
import torchvision
import torchvision.transforms as transforms
import os
from PIL import Image
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from torch.utils.data import Dataset, DataLoader
import time


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
		self.CNN_embed_dim = 30

		self.ch1, self.ch2, self.ch3, self.ch4 = 8, 16, 32, 64
		self.k1, self.k2, self.k3, self.k4 = (5, 5), (3, 3), (3, 3), (3, 3)      # 2d kernal size
		self.s1, self.s2, self.s3, self.s4 = (2, 2), (2, 2), (2, 2), (2, 2)      # 2d strides
		self.pd1, self.pd2, self.pd3, self.pd4 = (0, 0), (0, 0), (0, 0), (0, 0)  # 2d padding

		self.conv1_outshape = conv2D_output_size((self.img_x, self.img_y), self.pd1, self.k1, self.s1)  # Conv1 output shape
		self.conv2_outshape = conv2D_output_size(self.conv1_outshape, self.pd2, self.k2, self.s2)
		self.conv3_outshape = conv2D_output_size(self.conv2_outshape, self.pd3, self.k3, self.s3)
		self.conv4_outshape = conv2D_output_size(self.conv3_outshape, self.pd4, self.k4, self.s4)
		self.fc_hidden1, self.fc_hidden2 = 128, 128
		self.drop_p = drop_p

		self.conv1 = nn.Sequential(
			nn.Conv2d(in_channels=3, out_channels=self.ch1, kernel_size=self.k1, stride=self.s1, padding=self.pd1),
			nn.BatchNorm2d(self.ch1, momentum=0.01),
			nn.ReLU(inplace=True),                      
			# nn.MaxPool2d(kernel_size=2)
		)
		self.conv2 = nn.Sequential(
			nn.Conv2d(in_channels=self.ch1, out_channels=self.ch2, kernel_size=self.k2, stride=self.s2, padding=self.pd2),
			nn.BatchNorm2d(self.ch2, momentum=0.01),
			nn.ReLU(inplace=True),
			# nn.MaxPool2d(kernel_size=2)
		)

		self.conv3 = nn.Sequential(
			nn.Conv2d(in_channels=self.ch2, out_channels=self.ch3, kernel_size=self.k3, stride=self.s3, padding=self.pd3),
			nn.BatchNorm2d(self.ch3, momentum=0.01),
			nn.ReLU(inplace=True),
			# nn.MaxPool2d(kernel_size=2)
		)

		self.conv4 = nn.Sequential(
			nn.Conv2d(in_channels=self.ch3, out_channels=self.ch4, kernel_size=self.k4, stride=self.s4, padding=self.pd4),
			nn.BatchNorm2d(self.ch4, momentum=0.01),
			nn.ReLU(inplace=True),
			# nn.MaxPool2d(kernel_size=2)
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

class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()

        self.RNN_input_size = 30
        self.h_RNN_layers = 3   # RNN hidden layers
        self.h_RNN = 64                 # RNN hidden nodes
        self.h_FC_dim = 32
        self.drop_p = 0.3
        self.num_classes = 6

        self.LSTM = nn.LSTM(
            input_size=self.RNN_input_size,
            hidden_size=self.h_RNN,        
            num_layers=self.h_RNN_layers,       
            batch_first=True,       # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
        )

        self.fc1 = nn.Linear(self.h_RNN, self.h_FC_dim)
        self.fc2 = nn.Linear(self.h_FC_dim, self.num_classes)

    def forward(self, x_RNN):
        
        self.LSTM.flatten_parameters()
        RNN_out, (h_n, h_c) = self.LSTM(x_RNN, None)  
        """ h_n shape (n_layers, batch, hidden_size), h_c shape (n_layers, batch, hidden_size) """ 
        """ None represents zero initial hidden state. RNN_out has shape=(batch, time_step, output_size) """

        # FC layers
        x = self.fc1(RNN_out[:, -1, :])   # choose RNN_out at the last time step
        x = F.relu(x)
        x = F.dropout(x, p=self.drop_p, training=self.training)
        x = self.fc2(x)

        return x


class AGH_Dataset(Dataset):
	def __init__(self, dir_name):
		self.dir = dir_name
		self.folders = [folder for folder in os.listdir(dir_name) if folder[0].isalpha()]

	def __len__(self):
		return len(self.folders)

	def read_images(self, folder):
		X = []
		for i in range(100):
			im = Image.open(self.dir + '/' + folder + '/' + str(i) + '.jpg')
			transform = transforms.Compose([transforms.Resize([256, 256]),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
			im = transform(im)
			X.append(im)
		X = torch.stack(X, dim=0)
		return X

	def __getitem__ (self, idx):
		folder = self.folders[idx]
		X = self.read_images(folder)
		y = classes.index(folder[:2])

		return X,y

def train(cnn, lstm, train_loader, optimizer, criterion, epoch, device):
	cnn.train()
	lstm.train()
	losses = []
	accs = []
	for batch_idx, (X, y) in enumerate(trainloader):
		X, y = X.to(device), y.to(device).view(-1, )
		optimizer.zero_grad()
		output = lstm(cnn((X)))
		loss = criterion(output,y)
		losses.append(loss.item())

		y_pred = torch.max(output,1)[1]
		acc = 0
		for i in range(len(y_pred)):
			if y_pred[i] == y[i]:
				acc += 1
		accs.append(acc/len(y_pred))

		loss.backward()
		optimizer.step()

	return losses, accs

if __name__ == '__main__':
	print('starting')
	start = time.time()
	num_epochs = 100
	trainset = AGH_Dataset('data/mediumjpg')
	trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)
	device = "cuda"
	cnn = CNN().to(device)
	lstm = RNN().to(device)
	cnn = nn.DataParallel(cnn)
	lstm = nn.DataParallel(lstm)

	crnn_params = list(cnn.parameters()) + list(lstm.parameters())
	optimizer = torch.optim.Adam(crnn_params, lr=1e-4)
	criterion = nn.CrossEntropyLoss()
	epoch_train_losses = []
	accuracies = []
	# epoch_test_losses = []
	for epoch in range(num_epochs):
		train_losses, accs = train(cnn, lstm, trainloader, optimizer, criterion, epoch, device)
		average_loss = 0
		accuracy = 0
		for loss in train_losses:
			average_loss += loss
		for acc in accs:
			accuracy += acc
		accuracy /= len(accs)
		average_loss /= len(train_losses)
		epoch_train_losses.append(average_loss)
		accuracies.append(accuracy)

		print(epoch, average_loss, accuracy)

	print(f'done training after {time.time() - start} seconds')

	#5 epoch, no gpu - 375.50982117652893 seconds
	#5 epoch, gpu -  122.61604070663452 seconds
	

