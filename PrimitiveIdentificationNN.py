#Things to do:
#Set up data using DataLoader
#Set up training and accuracy checking (take from our previous network)
#Fix up _init_ and forward (mostly concerned with LSTM part of forward function)
#Test network and tune hyperparameters/network structure

#imports 
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets
import numpy as np 

#resolution: 512 x 512

#hyperparameters
num_epochs = 100
num_classes = 90
batch_size = 100
learning_rate = 0.001
CNN_layers = 100
LSTM_layers = 1
feature_vector_dim = 1000
hidden_dim = 100

#set up training & testing data: DataLoader

class PINN(nn.Module):
	def _init_(self):
		super(PINN, self)._init()
		self.cnn = nn.Sequential(
			#Conv2d parameters: # of input channels, # of output channels,
			#size of convolutional filter (square or tuple of x/y), how much 
			#filter moves each iteration, padding (from equation)
			nn.Conv2d(3, CNN_layers, kernel_size = 3, stride = 1, padding = 1),
			nn.ReLU(),
			#MaxPool2d parameters: same as Conv2d, except stride is >1 as
			#we want to downsize the image (from equation) 
			nn.MaxPool2d(kernel_size = 2, stride = 2))
		#possible to insert another convolutional layer
		#somehow set up feature vectors: # of nodes in current layer, # 
		#of nodes in next layer
		self.fc1 = nn.Linear(256 * 256 * CNN_layers, feature_vector_dim)
		#avoid overfitting (here, after LSTM, or both?)
		self.drop_out = nn.Dropout()
		#LSTM parameters: input dimension at each time step, size of
		#hidden/cell state at each time step, # of LSTM layers
		self.lstm = nn.LSTM(feature_vector_dim, hidden_dim, LSTM_layers, batch_first = True)
		#go from LSTM output to # of classes: # of nodes in current layer, # 
		#of nodes in next layer 
		self.fc2 = nn.Linear(hidden_dim, num_classes)
		#make class identification between 0 & 1
		self.sigmoid = nn.Sigmoid()


	def forward(self, x):
		out = self.cnn(x)
		#go from some multidimensional input to one-dimensional output
		out = out.reshape(out.size(0), -1)
		out = self.fc1(out)
		out = self.dropout(out)
		out, hidden = self.lstm(x.view(len(x), batch_size, -1))
		#only take last timestep
		out = self.fc2(out[-1].view(batch_size, -1))
		out = self.sigmoid(out)
		return out.view(-1)

	def init_hidden(self):
		#Hidden parameters: # of layers, batch size, size of hidden dimension
		return (torch.zeros(LSTM_layers, batch_size, hidden_dim), torch.zeros(LSTM_layers, batch_size, hidden_dim))