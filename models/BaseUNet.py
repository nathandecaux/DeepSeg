import torch
import torch.nn as nn
import torch.nn.functional as F
class up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        def double_conv(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels,
                          kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )
        super(up,self).__init__()

        if bilinear:
            self.up = nn.Upsample(
                scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranpose2d(in_channels // 2, in_channels // 2,
                                        kernel_size=2, stride=2)

        self.conv = double_conv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # [?, C, H, W]
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)  # why 1?

        return self.conv(x)
class BaseUNet(nn.Module):
    def __init__(self, n_channels=1,n_classes=1):
        super(BaseUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = True
        def double_conv(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels,
                          kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )

        def down(in_channels, out_channels):
            return nn.Sequential(
                nn.MaxPool2d(2),
                double_conv(in_channels, out_channels)
            )

        self.inc = double_conv(self.n_channels, 64)
        self.down1 = down(64, 128)
        self.down_reduced = down(128,128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 512)
        self.down4 = down(512, 512)
        self.up1 = up(1024, 256)
        self.up2 = up(512, 128)
        self.up3 = up(256, 64)
        self.up4 = up(128, 64)
        if 1==2:#not self.supervised:
            self.noise=nn.Dropout2d(0.5,inplace=False)
        else:
            self.noise=nn.Dropout2d(0,inplace=False)
        
        self.out = nn.Conv2d(64, self.n_classes, kernel_size=1)
    def forward(self, x):
        x1 = self.inc(x)
        x1=self.noise(x1)
        x2 = self.down1(x1)
        x2=self.noise(x2)
        x3 = self.down2(x2)
        x3=self.noise(x3)
        x4 = self.down3(x3)
        x4=self.noise(x4)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return self.out(x)