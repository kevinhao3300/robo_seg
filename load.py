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


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

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

	trainset = AGH_Dataset('data/small')
	trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)
	for x in trainloader:
		for lbl in x['label']:
			print(lbl)
		break

	net = CNN()
	criterion = nn.CrossEntropyLoss()
	optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
	for epoch in range(2):  # loop over the dataset multiple times

	    running_loss = 0.0
	    for i, data in enumerate(trainloader, 0):
	        # get the inputs; data is a list of [inputs, labels]
	        inputs, labels = data

	        # zero the parameter gradients
	        optimizer.zero_grad()

	        # forward + backward + optimize
	        outputs = net(inputs)
	        loss = criterion(outputs, labels)
	        loss.backward()
	        optimizer.step()

	        # print statistics
	        running_loss += loss.item()
	        if i % 2000 == 1999:    # print every 2000 mini-batches
	            print('[%d, %5d] loss: %.3f' %
	                  (epoch + 1, i + 1, running_loss / 2000))
	            running_loss = 0.0

	print('Finished Training')

