import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import GetCorrectPredCount

    
#-----------------------S10 Assignment----------------
class Block(nn.Module):
    """
    This class defines a convolution layer followed by
    normalization and activation function. Relu is used as activation function.
    """
    expansion = 1
    def __init__(self, input_channels, planes, stride=1):
        """Initialize Block

        Args:
            input_size (int): Input Channel Size
            output_size (int): Output Channel Size
        """
        super(Block, self).__init__()
        self.relu = nn.ReLU()
        self.conv1 = nn.Sequential(nn.Conv2d(input_channels, planes, 3, padding=1, stride = stride, bias=False),
                     nn.BatchNorm2d(planes))
        self.conv2 = nn.Sequential(nn.Conv2d(planes, planes, 3, padding=1, stride = stride, bias=False),
                     nn.BatchNorm2d(planes))
        
        self.shortcut = nn.Sequential()
        if stride != 1 or input_channels != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(input_channels, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )


    def forward(self, x):
        """
        Args:
            x (tensor): Input tensor to this block
        Returns:
            tensor: Return processed tensor
        """
        
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        out += self.shortcut(x)
        out = self.relu(out)

        return out
    
    
class ResNet(nn.Module):
  """ Network Class

  Args:
      nn (nn.Module): Instance of pytorch Module
  """

  def __init__(self, block, num_blocks, classes):
      """Initialize Network
      """
      super(ResNet, self).__init__()
      self.in_planes = 64
      self.classes = classes
      self.prep = self._first_layer()
      self.layer1 = self._make_layer(block, 64, num_blocks[0], stride = 1)
      self.layer2 = self._make_layer(block, 128, num_blocks[1], stride = 1)
      self.layer3 = self._make_layer(block, 256, num_blocks[2], stride = 1)
      self.layer4 = self._make_layer(block, 512, num_blocks[3], stride = 1)
      self.classifier = nn.Sequential(
          nn.AdaptiveAvgPool2d((1, 1)),
          nn.Conv2d(512, classes,1))
      

  def _first_layer(self):
      return nn.Sequential(
          nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=3, stride = 2, bias=False),
          nn.ReLU(),
          nn.BatchNorm2d(64),
          nn.MaxPool2d(kernel_size=3, stride=2)
      ) 
  
  def _make_layer(self, block, planes, num_blocks, stride):
      strides = [stride] + [1] * (num_blocks - 1)
      layers = []
      for stride in strides:
          layers.append(block(self.in_planes, planes, stride))
          self.in_planes = planes * block.expansion
      return nn.Sequential(*layers)
      
      
  def forward(self, x):
      
      out = self.prep(x)
      out = self.layer1(out)
      out = self.layer2(out)
      out = self.layer3(out)
      out = self.layer4(out)
      out = self.classifier(out)
      # Reshape
      out = out.view(-1, self.classes)


      # Output Layer
      return out
    
def ResNet18(classes=10):
  return ResNet(Block, [2, 2, 2, 2], classes=classes)

def ResNet34(classes=10):
    return ResNet(Block, [3, 4, 6, 3], classes=classes)

