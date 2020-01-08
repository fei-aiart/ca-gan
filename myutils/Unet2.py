# -*- coding: utf-8 -*-
# @Author: JacobShi777

import torch
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
import os
import cv2
import random
import argparse
import random
import functools
import torch.nn as nn
from torch.autograd import Variable

	
class MyUnetGenerator(nn.Module):
	def __init__(self, input_nc, output_nc, num_downs, ngf=64, \
					norm_layer=nn.BatchNorm2d, use_dropout=False, gpu_ids=[]):
		super(MyUnetGenerator, self).__init__()
		self.gpu_ids = gpu_ids


		self.conv1 = nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1)

		self.relu2 = nn.LeakyReLU(0.2, True)
		self.conv2 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
		self.norm2 = norm_layer(128)

		self.relu3 = nn.LeakyReLU(0.2, True)
		self.conv3 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)
		self.norm3 = norm_layer(256)

		self.relu4 = nn.LeakyReLU(0.2, True)
		self.conv4 = nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1)
		self.norm4 = norm_layer(512)

		self.relu5 = nn.LeakyReLU(0.2, True)
		self.conv5 = nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1)
		self.norm5 = norm_layer(512)

		self.relu6 = nn.LeakyReLU(0.2, True)
		self.conv6 = nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1)
		self.norm6 = norm_layer(512)

		self.relu7 = nn.LeakyReLU(0.2, True)
		self.conv7 = nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1)
		self.norm7 = norm_layer(512)

		self.relu8 = nn.LeakyReLU(0.2, True)
		self.conv8 = nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1)
		self.relu9 = nn.ReLU(True)
		self.conv9 = nn.ConvTranspose2d(1024, 512,kernel_size=4, stride=2,padding=1)
		self.norm9 = norm_layer(512)

		self.relu10 = nn.ReLU(True)
		self.conv10 = nn.ConvTranspose2d(1536, 512,kernel_size=4, stride=2,padding=1)
		self.norm10 = norm_layer(512)

		self.relu11 = nn.ReLU(True)
		self.conv11 = nn.ConvTranspose2d(1536, 512,kernel_size=4, stride=2,padding=1)
		self.norm11 = norm_layer(512)

		self.relu12 = nn.ReLU(True)
		self.conv12 = nn.ConvTranspose2d(1536, 512,kernel_size=4, stride=2,padding=1)
		self.norm12 = norm_layer(512)

		self.relu13 = nn.ReLU(True)
		self.conv13 = nn.ConvTranspose2d(1536, 256,kernel_size=4, stride=2,padding=1)
		self.norm13 = norm_layer(256)

		self.relu14 = nn.ReLU(True)
		self.conv14 = nn.ConvTranspose2d(768, 128,kernel_size=4, stride=2,padding=1)
		self.norm14 = norm_layer(128)

		self.relu15 = nn.ReLU(True)
		self.conv15 = nn.ConvTranspose2d(384, 64,kernel_size=4, stride=2,padding=1)
		self.norm15 = norm_layer(64)

		self.relu16 = nn.ReLU(True)
		self.conv16 = nn.ConvTranspose2d(192, 1,kernel_size=4, stride=2,padding=1)
		self.tanh16 = nn.Tanh()
		


	def forward(self, x, parsing_feature):

		x = self.conv1(x)
		temp1 = x

		x = self.relu2(x)
		x = self.conv2(x)
		x = self.norm2(x)
		temp2 = x

		x = self.relu3(x)
		x = self.conv3(x)
		x = self.norm3(x)
		temp3 = x

		x = self.relu4(x)
		x = self.conv4(x)
		x = self.norm4(x)
		temp4 = x

		x = self.relu5(x)
		x = self.conv5(x)
		x = self.norm5(x)
		temp5 = x

		x = self.relu6(x)
		x = self.conv6(x)
		x = self.norm6(x)
		temp6 = x

		x = self.relu7(x)
		x = self.conv7(x)
		x = self.norm7(x)
		temp7 = x

		x = self.relu8(x)
		x = self.conv8(x)
		x = torch.cat([x, parsing_feature[0]], 1)
		x = self.relu9(x)
		x = self.conv9(x)
		x = self.norm9(x)
		x = torch.cat([x, temp7, parsing_feature[7]], 1)

		x = self.relu10(x)
		x = self.conv10(x)
		x = self.norm10(x)
		x = torch.cat([x, temp6, parsing_feature[6]], 1)

		x = self.relu11(x)
		x = self.conv11(x)
		x = self.norm11(x)
		x = torch.cat([x, temp5, parsing_feature[5]], 1)

		x = self.relu12(x)
		x = self.conv12(x)
		x = self.norm12(x)
		x = torch.cat([x, temp4, parsing_feature[4]], 1)

		x = self.relu13(x)
		x = self.conv13(x)
		x = self.norm13(x)
		x = torch.cat([x, temp3, parsing_feature[3]], 1)

		x = self.relu14(x)
		x = self.conv14(x)
		x = self.norm14(x)
		x = torch.cat([x, temp2, parsing_feature[2]], 1)

		x = self.relu15(x)
		x = self.conv15(x)
		x = self.norm15(x)
		x = torch.cat([x, temp1, parsing_feature[1]], 1)

		x = self.relu16(x)
		x = self.conv16(x)
		x = self.tanh16(x)


		return x



class MyUnetGenerator2(nn.Module):
	def __init__(self, input_nc, output_nc, num_downs, ngf=64, \
					norm_layer=nn.BatchNorm2d, use_dropout=False, gpu_ids=[]):
		super(MyUnetGenerator2, self).__init__()
		self.gpu_ids = gpu_ids


		self.conv1 = nn.Conv2d(4, 64, kernel_size=4, stride=2, padding=1)

		self.relu2 = nn.LeakyReLU(0.2, True)
		self.conv2 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
		self.norm2 = norm_layer(128)

		self.relu3 = nn.LeakyReLU(0.2, True)
		self.conv3 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)
		self.norm3 = norm_layer(256)

		self.relu4 = nn.LeakyReLU(0.2, True)
		self.conv4 = nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1)
		self.norm4 = norm_layer(512)

		self.relu5 = nn.LeakyReLU(0.2, True)
		self.conv5 = nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1)
		self.norm5 = norm_layer(512)

		self.relu6 = nn.LeakyReLU(0.2, True)
		self.conv6 = nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1)
		self.norm6 = norm_layer(512)

		self.relu7 = nn.LeakyReLU(0.2, True)
		self.conv7 = nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1)
		self.norm7 = norm_layer(512)

		self.relu8 = nn.LeakyReLU(0.2, True)
		self.conv8 = nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1)
		self.relu9 = nn.ReLU(True)
		self.conv9 = nn.ConvTranspose2d(1024, 512,kernel_size=4, stride=2,padding=1)
		self.norm9 = norm_layer(512)

		self.relu10 = nn.ReLU(True)
		self.conv10 = nn.ConvTranspose2d(1024, 512,kernel_size=4, stride=2,padding=1)
		self.norm10 = norm_layer(512)

		self.relu11 = nn.ReLU(True)
		self.conv11 = nn.ConvTranspose2d(1024, 512,kernel_size=4, stride=2,padding=1)
		self.norm11 = norm_layer(512)

		self.relu12 = nn.ReLU(True)
		self.conv12 = nn.ConvTranspose2d(1024, 512,kernel_size=4, stride=2,padding=1)
		self.norm12 = norm_layer(512)

		self.relu13 = nn.ReLU(True)
		self.conv13 = nn.ConvTranspose2d(1024, 256,kernel_size=4, stride=2,padding=1)
		self.norm13 = norm_layer(256)

		self.relu14 = nn.ReLU(True)
		self.conv14 = nn.ConvTranspose2d(512, 128,kernel_size=4, stride=2,padding=1)
		self.norm14 = norm_layer(128)

		self.relu15 = nn.ReLU(True)
		self.conv15 = nn.ConvTranspose2d(256, 64,kernel_size=4, stride=2,padding=1)
		self.norm15 = norm_layer(64)

		self.relu16 = nn.ReLU(True)
		self.conv16 = nn.ConvTranspose2d(128, 1,kernel_size=4, stride=2,padding=1)
		self.tanh16 = nn.Tanh()
		


	def forward(self, x, parsing_feature):

		x = self.conv1(x)
		temp1 = x

		x = self.relu2(x)
		x = self.conv2(x)
		x = self.norm2(x)
		temp2 = x

		x = self.relu3(x)
		x = self.conv3(x)
		x = self.norm3(x)
		temp3 = x

		x = self.relu4(x)
		x = self.conv4(x)
		x = self.norm4(x)
		temp4 = x

		x = self.relu5(x)
		x = self.conv5(x)
		x = self.norm5(x)
		temp5 = x

		x = self.relu6(x)
		x = self.conv6(x)
		x = self.norm6(x)
		temp6 = x

		x = self.relu7(x)
		x = self.conv7(x)
		x = self.norm7(x)
		temp7 = x

		x = self.relu8(x)
		x = self.conv8(x)
		x = torch.cat([x, parsing_feature], 1)
		x = self.relu9(x)
		x = self.conv9(x)
		x = self.norm9(x)
		x = torch.cat([x, temp7], 1)

		x = self.relu10(x)
		x = self.conv10(x)
		x = self.norm10(x)
		x = torch.cat([x, temp6], 1)

		x = self.relu11(x)
		x = self.conv11(x)
		x = self.norm11(x)
		x = torch.cat([x, temp5], 1)

		x = self.relu12(x)
		x = self.conv12(x)
		x = self.norm12(x)
		x = torch.cat([x, temp4], 1)

		x = self.relu13(x)
		x = self.conv13(x)
		x = self.norm13(x)
		x = torch.cat([x, temp3], 1)

		x = self.relu14(x)
		x = self.conv14(x)
		x = self.norm14(x)
		x = torch.cat([x, temp2], 1)

		x = self.relu15(x)
		x = self.conv15(x)
		x = self.norm15(x)
		x = torch.cat([x, temp1], 1)

		x = self.relu16(x)
		x = self.conv16(x)
		x = self.tanh16(x)


		return x


class MyEncoder(nn.Module):
	def __init__(self, input_nc, output_nc, num_downs, ngf=64, \
					norm_layer=nn.BatchNorm2d, use_dropout=False, gpu_ids=[]):
		super(MyEncoder, self).__init__()
		self.gpu_ids = gpu_ids


		self.conv1 = nn.Conv2d(8, 64, kernel_size=4, stride=2, padding=1)

		self.relu2 = nn.LeakyReLU(0.2, True)
		self.conv2 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
		self.norm2 = norm_layer(128)

		self.relu3 = nn.LeakyReLU(0.2, True)
		self.conv3 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)
		self.norm3 = norm_layer(256)

		self.relu4 = nn.LeakyReLU(0.2, True)
		self.conv4 = nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1)
		self.norm4 = norm_layer(512)

		self.relu5 = nn.LeakyReLU(0.2, True)
		self.conv5 = nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1)
		self.norm5 = norm_layer(512)

		self.relu6 = nn.LeakyReLU(0.2, True)
		self.conv6 = nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1)
		self.norm6 = norm_layer(512)

		self.relu7 = nn.LeakyReLU(0.2, True)
		self.conv7 = nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1)
		self.norm7 = norm_layer(512)

		self.relu8 = nn.LeakyReLU(0.2, True)
		self.conv8 = nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1)
		

	def forward(self, x):

		x = self.conv1(x)
		temp1 = x

		x = self.relu2(x)
		x = self.conv2(x)
		x = self.norm2(x)
		temp2 = x

		x = self.relu3(x)
		x = self.conv3(x)
		x = self.norm3(x)
		temp3 = x

		x = self.relu4(x)
		x = self.conv4(x)
		x = self.norm4(x)
		temp4 = x

		x = self.relu5(x)
		x = self.conv5(x)
		x = self.norm5(x)
		temp5 = x

		x = self.relu6(x)
		x = self.conv6(x)
		x = self.norm6(x)
		temp6 = x

		x = self.relu7(x)
		x = self.conv7(x)
		x = self.norm7(x)
		temp7 = x

		x = self.relu8(x)
		x = self.conv8(x)

		return [x,temp1,temp2,temp3,temp4,temp5,temp6,temp7]

