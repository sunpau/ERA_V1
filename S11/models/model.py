import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import GetCorrectPredCount

    
#-----------------------S10 Assignment----------------
class ResBlock(nn.Module):
    """
    This class defines a convolution layer followed by
    normalization and activation function. Relu is used as activation function.
    """
    def __init__(self, input_size, output_size):
        """Initialize Block

        Args:
            input_size (int): Input Channel Size
            output_size (int): Output Channel Size
        """
        super(ResBlock, self).__init__()

        self.conv1 = nn.Sequential(nn.Conv2d(input_size, output_size, 3, padding=1, bias=False),
                     nn.BatchNorm2d(output_size),
                     nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv2d(output_size, output_size, 3, padding=1, bias=False),
                     nn.BatchNorm2d(output_size),
                     nn.ReLU())


    def __call__(self, x):
        """
        Args:
            x (tensor): Input tensor to this block
        Returns:
            tensor: Return processed tensor
        """
        x = self.conv1(x)
        x = self.conv2(x)

        return x
    
    
class CustomNet(nn.Module):
    """ Network Class

    Args:
        nn (nn.Module): Instance of pytorch Module
    """

    def __init__(self):
        """Initialize Network
        """
        super(CustomNet, self).__init__()
        self.prep = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(64)
        ) 
        
        self.L1X = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), padding=1, bias=False),
            nn.MaxPool2d(2, 2),
            nn.ReLU(),
            nn.BatchNorm2d(128)
        ) 
        self.L1R1 = ResBlock(128,128)
        
        self.L2 = nn.Sequential(nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), padding=1, bias=False),
            nn.MaxPool2d(2, 2),
            nn.ReLU(),
            nn.BatchNorm2d(256)
        )
        self.L3X = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3, 3), padding=1, bias=False),
            nn.MaxPool2d(2, 2),
            nn.ReLU(),
            nn.BatchNorm2d(512)
        ) 
        self.L3R2 = ResBlock(512, 512)

        # OUTPUT BLOCK
        self.mp = nn.Sequential(
            nn.MaxPool2d(kernel_size=4)
        ) 

        self.FC = nn.Sequential(
            nn.Linear(in_features=512,out_features=10,bias=False)
        ) 
    def forward(self, x):
        
        x = self.prep(x)
        x = self.L1X(x)
        xR1 = self.L1R1(x) 
        x = xR1 + x
        x = self.L2(x)
        x = self.L3X(x)
        xR2 = self.L3R2(x) 
        x = xR2 + x
        x = self.mp(x) 
        x = torch.squeeze(x)  
        x = self.FC(x)


        # Output Layer
        return x
        
