import torch
import torch.nn as nn
import torch.nn.functional as F

class Down(nn.Module):
    def __init__(self, in_c, out_c) -> None:
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, 1, 1),
            nn.LeakyReLU(),
            nn.MaxPool2d(2, 2)
        )

    def forward(self, x):
        return self.maxpool_conv(x)
    
class Up(nn.Module):
    def __init__(self, in_c, out_c, mode="bilinear") -> None:
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode=mode)
        self.dbconv = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, 1, 1),
            nn.LeakyReLU(),
            nn.Conv2d(out_c, out_c, 3, 1, 1),
            nn.LeakyReLU(),
        )

    def forward(self, x1, x2):
        x = self.up(x1)
        x = torch.concat([x, x2], dim=1)
        return self.dbconv(x)




class AutoEncoder(nn.Module):
    def __init__(self, in_c=1, out_c=1) -> None:
        super().__init__()

        self.in_c = in_c
        self.out_c = out_c
        self.input_layer = nn.Sequential(
            nn.Conv2d(self.in_c, 48, 3, 1, 1),
            nn.LeakyReLU(),
        )

        self.down1 = Down(48, 48)
        self.down2 = Down(48, 48)
        self.down3 = Down(48, 48)
        self.down4 = Down(48, 48)
        self.down5 = Down(48, 48)

        self.bottle_neck = nn.Sequential(
            nn.Conv2d(48, 48, 3, 1, 1),
            nn.LeakyReLU(),
        )

        self.up1 = Up(96, 96)
        self.up2 = Up(144, 96)
        self.up3 = Up(144, 96)
        self.up4 = Up(144, 96)
        self.up5 = Up(97, 96)

        self.final = nn.Conv2d(96, out_c, 3, 1, 1)

    def forward(self, x):
        # x = F.pad(x, (0,1,0,1), "constant", 0)
        input = x
        x = self.input_layer(x)
        
        x1 = self.down1(x)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        x = self.down5(x4)

        x = self.bottle_neck(x)

        x = self.up1(x, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.up5(x, input)

        return self.final(x)
