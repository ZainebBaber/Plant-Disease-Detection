import torch.nn as nn
import torch.nn.functional as F

class PlantCNN(nn.Module):
  def __init__(self, num_classes):
    super(PlantCNN, self).__init__()

    # Conv Block 1 rgb=3(input), feature map=16, filter=3
    self.conv1=nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
    self.pool=nn.MaxPool2d(2,2) #pooling layer shrinks the image keeping only the maximum value(important features) of matrix and then strides forward on the feature map

    # Conv Block 2 , inout=16 channels , output=32 channels
    self.conv2 = nn.Conv2d(16, 32, 3, padding=1)



    #  Fully Connected layers
    self.fc1 = nn.Linear(32 * 56 * 56, 64)  # 32 chaneels left, 224(resolution 224x224) after maxpool is half that is 112 then half is 56
    self.dropout = nn.Dropout(p=0.5)
    self.fc2 = nn.Linear(64, num_classes) #

  def forward(self, x): #running phase of the cnn
        # Conv Block 1
        x = F.relu(self.conv1(x)) #apply convolutional and then relu
        x = self.pool(x) #apply pool

        # Conv Block 2
        x = F.relu(self.conv2(x)) #same repition as 1
        x = self.pool(x)



        # Flatten
        x = x.view(x.size(0), -1)  #cnn is 3d so you flatten in

        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)  # output layer (logits)

        return x
