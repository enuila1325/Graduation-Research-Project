import torch.nn as nn
import torch.nn.functional as F

class image_classificator(nn.Module):
    def __init__(self, out):
        super(image_classificator, self).__init__()
        self.out = out
        self.conv_1 = nn.Conv2d(3, 32, 3)
        self.conv_2 = nn.Conv2d(32, 64, 3)
        self.conv_3 = nn.Conv2d(64, 128, 3)

        self.linear_1 = nn.Linear(128, 16)
        self.linear_2 = nn.Linear(16, self.out)
        
        
        self.pool = nn.MaxPool2d(2,2)
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        x = self.pool(F.relu(self.conv_1(x)))
        x = self.pool(F.relu(self.conv_2(x)))
        x = self.pool(F.relu(self.conv_3(x)))
        
        bs, _, _, _ = x.shape
        x = F.adaptive_avg_pool2d(x, 1).reshape(bs, -1)
        x = self.dropout(F.relu(self.linear_1(x)))
        x = self.linear_2(x)
        return x