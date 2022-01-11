
from torch import nn
import torch.nn.functional as F


class CNN_MNIST(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(         
            nn.Conv2d(
                in_channels=1,              
                out_channels=8,            
                kernel_size=5,              
                stride=1,
                padding=2,
            ),                              
            nn.ReLU(),                      
            nn.MaxPool2d(kernel_size=2),    
        )
        self.conv2 = nn.Sequential(         
            nn.Conv2d(8, 16, 5, 1, 2),     
            nn.ReLU(),                      
            nn.MaxPool2d(2),
        )

        # fully connected layer, output 10 classes
        self.output_layer = nn.Linear(16 * 7 * 7, 10)

    def forward(self, x):
        x = x.unsqueeze(1)

        x = self.conv1(x)
        x = self.conv2(x)

        # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
        x = x.view(x.size(0), -1)       

        output = F.log_softmax(self.output_layer(x), dim=1)

        return output


class CNN_MNIST_STRIDE(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(         
            nn.Conv2d(
                in_channels=1,              
                out_channels=8,            
                kernel_size=5,              
                stride=2,
                padding=2,
            ),                              
            nn.ReLU(),                      
            #nn.MaxPool2d(kernel_size=2),    
        )
        self.conv2 = nn.Sequential(         
            nn.Conv2d(8, 16, 5, 2, 2),     
            nn.ReLU(),                      
            #nn.MaxPool2d(2),
        )

        # fully connected layer, output 10 classes
        self.output_layer = nn.Linear(16 * 7 * 7, 10)

    def forward(self, x):
        x = x.unsqueeze(1)

        x = self.conv1(x)
        x = self.conv2(x)

        # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
        x = x.view(x.size(0), -1)       

        output = F.log_softmax(self.output_layer(x), dim=1)

        return output


class CNN_MNIST_DILATION(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(         
            nn.Conv2d(
                in_channels=1,              
                out_channels=8,            
                kernel_size=5,              
                stride=2,
                padding=2,
                dilation=2
            ),                              
            nn.ReLU(),                      
            # nn.MaxPool2d(kernel_size=2),    
        )
        self.conv2 = nn.Sequential(         
            nn.Conv2d(8, 16, 5, 1, 2, 1),     
            nn.ReLU(),                      
            nn.MaxPool2d(2),
        )

        # fully connected layer, output 10 classes
        self.output_layer = nn.Linear(16 * 6 * 6, 10)

    def forward(self, x):
        x = x.unsqueeze(1)

        x = self.conv1(x)
        x = self.conv2(x)

        # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
        x = x.view(x.size(0), -1)       

        output = F.log_softmax(self.output_layer(x), dim=1)

        return output