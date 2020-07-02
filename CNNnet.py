from torch import nn
import torch.nn.functional as F


# This python file contains a simple CNN network
class CNNnet(nn.Module):
    def __init__(self, num_classes=7):
        super(CNNnet, self).__init__()
        # Two CNN blocks
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2))
        # Fully connected neural network
        self.fc1 = nn.Linear(32 * 57 * 71, 1024)
        self.fc3 = nn.Linear(1024, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = F.leaky_relu(self.fc1(out))
        out = self.fc3(out)
        return out
