import torch
import torch.nn as nn
import torch.nn.functional as F


class LCNN(nn.Module):
	def __init__(self):
		super(LCNN, self).__init__()

		self.conv1 = nn.Conv2d(1, 96, kernel_size=5, stride=1, padding=2)
		self.conv2a = nn.Conv2d(48, 96, kernel_size=1, stride=1, padding=0)
		self.conv2 = nn.Conv2d(48, 192, kernel_size=3, stride=1, padding=1)
		self.conv3a = nn.Conv2d(96, 192, kernel_size=1, stride=1, padding=0)
		self.conv3 = nn.Conv2d(96, 384, kernel_size=3, stride=1, padding=1)
		self.conv4a = nn.Conv2d(192, 384, kernel_size=1, stride=1, padding=0)
		self.conv4 = nn.Conv2d(192, 256, kernel_size=3, stride=1, padding=1)
		self.conv5a = nn.Conv2d(128, 256, kernel_size=1, stride=1, padding=0)
		self.conv5 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)

		self.fc1 = nn.Linear(128*8*8, 512)
		self.fc2 = nn.Linear(256, 2000)

		self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
		self.dropout = nn.Dropout(p=0.85)
		self.slice_max = slice_max()



	def forward(self, x):
		x = self.conv1(x)
		x = self.slice_max(x)
		fea1 = x
		x = self.pool(x)

		x = self.conv2a(x)
		x = self.slice_max(x)
		x = self.conv2(x)
		x = self.slice_max(x)
		fea2 = x
		x = self.pool(x)

		x = self.conv3a(x)
		x = self.slice_max(x)
		x = self.conv3(x)
		x = self.slice_max(x)
		fea3 = x
		x = self.pool(x)

		x = self.conv4a(x)
		x = self.slice_max(x)
		x = self.conv4(x)
		x = self.slice_max(x)

		x = self.conv5a(x)
		x = self.slice_max(x)
		x = self.conv5(x)
		x = self.slice_max(x)
		fea4 = x
		x = self.pool(x)

		x = x.view(-1, 128*8*8)

		x = self.fc1(x)
		x = self.slice_max(x)
		x = self.dropout(x)
		x = self.fc2(x)

		return x, [fea1, fea2, fea3, fea4]

	
class slice_max(nn.Module):
	def __init__(self):
		super(slice_max, self).__init__()
		
	def forward(self, x):
		size1 = x.size(1)/2
		x = torch.split(x, size1, 1)
		x = torch.max(x[0], x[1])
		return x	
	




