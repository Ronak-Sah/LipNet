import torch
import torch.nn as nn

class CnnEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.net=nn.Sequential(

            nn.Conv3d(1, 64, kernel_size=(3,5,5), padding=(1,2,2)),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.MaxPool3d((1,2,2)),

            nn.Conv3d(64, 128, kernel_size=(3,5,5), padding=(1,2,2)),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            nn.MaxPool3d((1,2,2)),

            nn.Conv3d(128, 256, kernel_size=(3,5,5), padding=(1,2,2)),
            nn.BatchNorm3d(256),
            nn.ReLU(),
            nn.MaxPool3d((1,2,2)),

        )

    def forward(self,x):

        x=self.net(x)
        x = x.mean(dim=[3,4])
        x=x.permute(0,2,1)

        return x
