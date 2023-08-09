import torch.nn as nn
import torch
class FlowerPowerNet(nn.Module):
    def __init__(self, path=None):
        super().__init__()
        self.path = path
        
        
        self.network = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # output: 64 x 16 x 16

            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # output: 128 x 8 x 8

            nn.Conv2d(16, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # output: 256 x 4 x 4

            nn.Flatten(), 
            nn.Linear(8*28*28, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 5))
        
            
    def load_state_dict(self,device):
        self.network = torch.load(self.path,map_location=torch.device(f'{device}'))
        #load_state_dict(torch.load(self.path))#,map_location=torch.device('cpu')))#,strict=False)
        
    def forward(self, xb):
        return self.network(xb)
        #return torch.sigmoid(self.network(xb))
 