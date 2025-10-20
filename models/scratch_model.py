import torch
import torch.nn as nn
import torch.nn.functional as F

# 모델 
class MLPFromScratch(nn.Module):
    def __init__(self, input_size=784, hidden_sizes=[256, 128], num_classes=10):
        super(MLPFromScratch, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_sizes[0])
        self.fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.fc3 = nn.Linear(hidden_sizes[1], num_classes)
        self.dropout = nn.Dropout(0.3)
        self.batchnorm1 = nn.BatchNorm1d(hidden_sizes[0])
        self.batchnorm2 = nn.BatchNorm1d(hidden_sizes[1])

    def forward(self, x):
        x = x.view(x.size(0), -1)  # flatten 28x28 -> 784
        x = F.relu(self.batchnorm1(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.batchnorm2(self.fc2(x)))
        x = self.dropout(x)
        x = self.fc3(x)
        return x
    
