import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

# Downsampling block (encoder path)
class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

# Upsampling block (decoder path)
class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        # Crop x2 to match the size of x1
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x2 = x2[:, :, diffY // 2 : x2.size()[2] - diffY // 2,
                     diffX // 2 : x2.size()[3] - diffX // 2]

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

# Output convolution layer
class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


# U-Net Model Definition
class UNet(nn.Module):
    def __init__(self, n_channels=3, n_seg_classes=1, n_breed_classes=37):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_seg_classes = n_seg_classes
        self.n_breed_classes = n_breed_classes

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024) # Last encoder output, used for classifier

        self.up1 = Up(1024, 512)
        self.up2 = Up(512, 256)
        self.up3 = Up(256, 128)
        self.up4 = Up(128, 64)
        self.outc = OutConv(64, n_seg_classes) # Segmentation output

        # Classifier head
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(1024, n_breed_classes)
        )

    def forward(self, x):
        # Encoder path
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # Breed classification branch
        breed_logits = self.classifier(x5)

        # Decoder path (segmentation)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        seg_logits = self.outc(x) # Segmentation output

        return seg_logits, breed_logits

class AttentionGate(nn.Module):
    def __init__(self, g_channels, x_channels, inter_channels):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(g_channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(x_channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(inter_channels, 1, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, g):
        g1 = self.W_g(g)
        x1 = self.W_x(x)

        psi_val = self.relu(g1 + x1)
        psi_val = self.psi(psi_val)

        return x * psi_val


# Upsampling block with Attention Gate
class UpAttention(nn.Module):
    def __init__(self, in_channels, out_channels, skip_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)

        self.attention = AttentionGate(g_channels=in_channels // 2, x_channels=skip_channels, inter_channels=skip_channels // 2)

        self.conv = DoubleConv(in_channels // 2 + skip_channels, out_channels)

    def forward(self, x1, x2):
        # x1 is from the previous decoder layer (upsampled by ConvTranspose)
        # x2 is from the corresponding encoder layer (skip connection)
        x1 = self.up(x1)

        # Crop x2 to match the size of x1. This is crucial for concatenation
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x2_cropped = x2[:, :, diffY // 2 : x2.size()[2] - diffY // 2,
                             diffX // 2 : x2.size()[3] - diffX // 2]

        # Apply attention to the cropped skip connection x2 using x1 as gating signal
        x2_att = self.attention(x2_cropped, x1)

        # Concatenate x2_att (attended skip connection) with x1 (upsampled feature map)
        x = torch.cat([x2_att, x1], dim=1)
        return self.conv(x)


# Attention U-Net Model Definition
class AttentionUNet(nn.Module):
    def __init__(self, n_channels=3, n_seg_classes=1, n_breed_classes=37):
        super(AttentionUNet, self).__init__()
        self.n_channels = n_channels
        self.n_seg_classes = n_seg_classes
        self.n_breed_classes = n_breed_classes

        # Encoder path
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024) # Bottleneck, and input for classifier

        # Decoder path with Attention Gates
        self.up1 = UpAttention(1024, 512, 512) # x5 (1024) -> up -> 512; x4 (512) -> AG -> 512; concat 512+512 -> conv 512
        self.up2 = UpAttention(512, 256, 256)   # up1_out (512) -> up -> 256; x3 (256) -> AG -> 256; concat 256+256 -> conv 256
        self.up3 = UpAttention(256, 128, 128)   # up2_out (256) -> up -> 128; x2 (128) -> AG -> 128; concat 128+128 -> conv 128
        self.up4 = UpAttention(128, 64, 64)     # up3_out (128) -> up -> 64; x1 (64) -> AG -> 64; concat 64+64 -> conv 64

        self.outc = OutConv(64, n_seg_classes)

        # Classifier head (attached to the deepest encoder output)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(1024, n_breed_classes)
        )

    def forward(self, x):
        # Encoder path
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4) # Bottleneck

        # Breed classification branch (from bottleneck output)
        breed_logits = self.classifier(x5)

        # Decoder path (segmentation)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        seg_logits = self.outc(x) # Segmentation output

        return seg_logits, breed_logits