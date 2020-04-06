## TODO: define the convolutional neural network architecture

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        # TODO: Define all the layers of this CNN, the only requirements are:

        self.conv1 = nn.Conv2d(1, 32, 5)
        self.bn1 = nn.BatchNorm2d(32)
        
        self.conv2 = nn.Conv2d(32, 64, 5)
        self.bn2 = nn.BatchNorm2d(64)
        
        self.conv3 = nn.Conv2d(64,128, 5)
        self.bn3 = nn.BatchNorm2d(128)
        
        self.fc1 = nn.Linear(128*24*24, 2000)
        self.fc2 = nn.Linear(2000, 136)
        
        self.pool = nn.MaxPool2d(2,2)
        self.dropout = nn.Dropout(p=.5)
        
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        

        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        
        x = x.view(x.size(0), -1)
        x = self.dropout(self.fc1(x))
        x = self.fc2(x) #no activation, it's a regression problem here
        
        # a modified x, having gone through all the layers of your model, should be returned
        return x

class DeeperNet(nn.Module):

    def __init__(self):
        super(DeeperNet, self).__init__()
        
        # Define all the layers of this CNN, the only requirements are:
        self.conv1 = nn.Conv2d(1, 32, 3)
        self.bn1 = nn.BatchNorm2d(32)
        
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.bn2 = nn.BatchNorm2d(64)
        
        self.conv3 = nn.Conv2d(64,128, 5)
        self.bn3 = nn.BatchNorm2d(128)
        
        self.conv4 = nn.Conv2d(128, 256, 5)
        self.bn4 = nn.BatchNorm2d(256)
        
        self.fc1 = nn.Linear(256*10*10, 1024)
        self.fc2 = nn.Linear(1024, 136)
        
        self.pool = nn.MaxPool2d(2,2)
        self.dropout = nn.Dropout(p=.5)
        
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        

        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        
        x = x.view(x.size(0), -1)
        x = self.dropout(self.fc1(x))
        x = self.fc2(x) #no activation, it's a regression problem here
        
        # a modified x, having gone through all the layers of your model, should be returned
        return x

class DeepestNet(nn.Module):

    def __init__(self):
        super(DeeperNet, self).__init__()
        
        # Define all the layers of this CNN, the only requirements are:
        # 224*224*1 -> 110*110*32 (with pool 2x2)
        self.conv1 = nn.Conv2d(1, 32, 5)
        self.bn1 = nn.BatchNorm2d(32)
        
        # 110*110*32 -> 53*53*64 (with pool 2x2)
        self.conv2 = nn.Conv2d(32, 64, 5)
        self.bn2 = nn.BatchNorm2d(64)
        
        # 53*53*64 -> 25*25*128 (with pool 2x2)
        self.conv3 = nn.Conv2d(64,128, 3)
        self.bn3 = nn.BatchNorm2d(128)
        
        # 25*25*128 -> 11*11*256 (with pool 2x2)
        self.conv4 = nn.Conv2d(128, 256, 3)
        self.bn4 = nn.BatchNorm2d(256)
        
        # 11*11*256 -> 4*4*512 (with pool 2x2)
        self.conv5 = nn.Conv2d(256, 512, 3)
        self.bn5 = nn.BatchNorm2d(256)
        
        self.fc1 = nn.Linear(512*4*4, 1024)
        self.fc2 = nn.Linear(1024, 136)
        
        self.pool = nn.MaxPool2d(2,2)
        self.dropout = nn.Dropout(p=.5)
        
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        

        
    def forward(self, x):
        # Define the feedforward behavior of this model
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        x = self.pool(F.relu(self.bn5(self.conv5(x))))
        
        x = x.view(x.size(0), -1)
        x = self.dropout(self.fc1(x))
        x = self.fc2(x) #no activation, it's a regression problem here
        
        return x
    

class ShallowNet(nn.Module):

    def __init__(self):
        super(ShallowNet, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        self.conv1 = nn.Conv2d(1, 32, 3)
        self.bn1 = nn.BatchNorm2d(32)
        
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.bn2 = nn.BatchNorm2d(64)
        
        self.fc1 = nn.Linear(64*54*54, 1024)
        self.fc2 = nn.Linear(1024, 136)
        
        self.pool = nn.MaxPool2d(2,2)
        self.dropout = nn.Dropout(p=.5)
        
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        

        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        
        x = x.view(x.size(0), -1)
        x = self.dropout(self.fc1(x))
        x = self.fc2(x) #no activation, it's a regression problem here
        
        # a modified x, having gone through all the layers of your model, should be returned
        return x