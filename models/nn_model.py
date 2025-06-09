import torch.nn as nn

class CTFAR_Model(nn.Module):
    def __init__(self):
        super(CTFAR_Model, self).__init__()
        self.model=nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=32,kernel_size=5,stride=1,
                      padding=2),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32,32,5,1,padding=2),
            nn.MaxPool2d(kernel_size=2),#stride和 padding怎么确定？
            nn.Conv2d(32,64,5,1,2), #yanzheng #5 !  32,8,8 ->64,8,8
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64,10)

        )

    def forward(self,x):
        pass