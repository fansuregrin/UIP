import torch.nn as nn
import torch


class WFEFB(nn.Module):
    def __init__(self, in_dim, out_dim, k_size=3):
        super().__init__()  
        self.norm = nn.InstanceNorm2d(in_dim)
        self.conv1 = nn.Conv2d(in_dim, out_dim, 1, 1, 0)
        self.conv2 = nn.Conv2d(in_dim, out_dim, 1, 1, 0)
        self.conv3 = nn.Conv2d(in_dim, out_dim, 1, 1, 0)
        self.conv4 = nn.Conv2d(in_dim, out_dim, 1, 1, 0)
        self.conv5 = nn.Conv2d(in_dim, out_dim, 1, 1, 0)
        self.sin_conv = nn.Conv2d(out_dim, out_dim, kernel_size=k_size, stride=1, padding=k_size//2)
        self.cos_conv = nn.Conv2d(out_dim, out_dim, kernel_size=k_size, stride=1, padding=k_size//2)
        self.last_conv = nn.Conv2d(out_dim*3, out_dim, 1, 1)
 
    def forward(self, x):
        x = self.norm(x)
        cos_f = self.conv1(x) * torch.cos(self.conv2(x))
        cos_f = self.cos_conv(cos_f)
        x_f = self.conv3(x)
        sin_f = self.conv5(x) * torch.sin(self.conv4(x))
        sin_f = self.sin_conv(sin_f)
        out = self.last_conv(torch.cat((cos_f, x_f, sin_f), dim=1))

        return out


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, activ=nn.ReLU):
        super().__init__()
        self.activ = activ()
        self.conv1 = nn.Conv2d(in_features, hidden_features, 1, 1)
        self.conv2 = nn.Conv2d(hidden_features, out_features, 1, 1)

    def forward(self, x):
        out = self.conv1(x)
        out = self.activ(out)
        out = self.conv2(out)
        return out


class SAWF(nn.Module):
    def __init__(self, in_channels, split, reduction=4):
        super().__init__()
        
        self.split = split
        hidden_dim = max(in_channels//reduction, 4)
        # self feature fusion
        self.sff = MLP(in_channels, hidden_dim, self.split)  # compress intra-channel information
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # self channel attention
        self.sca = MLP(in_channels, hidden_dim, in_channels*self.split) # compress inter-channel information

    def forward(self, features):
        B, C, H, W = features.shape
        sff = self.sff(features).softmax(dim=1).permute(1,0,2,3).unsqueeze(-3) # split*B*1*H*W
        sca = self.avg_pool(features)
        sca = self.sca(sca).reshape(B, C, self.split).permute(2, 0, 1)\
              .softmax(dim=0).unsqueeze(-1).unsqueeze(-1) # split*B*C*1*1

        attn = sff * sca # split*B*C*H*W
        return attn


class WFEFB2(nn.Module):
    def __init__(self, in_dim, out_dim, k_size=3):
        super().__init__()  
        self.norm = nn.InstanceNorm2d(in_dim)
        self.conv1 = nn.Conv2d(in_dim, out_dim, 1, 1, 0)
        self.conv2 = nn.Conv2d(in_dim, out_dim, 1, 1, 0)
        self.conv3 = nn.Conv2d(in_dim, out_dim, 1, 1, 0)
        self.conv4 = nn.Conv2d(in_dim, out_dim, 1, 1, 0)
        self.conv5 = nn.Conv2d(in_dim, out_dim, 1, 1, 0)
        self.sin_conv = nn.Conv2d(out_dim, out_dim, kernel_size=k_size, stride=1, padding=k_size//2)
        self.cos_conv = nn.Conv2d(out_dim, out_dim, kernel_size=k_size, stride=1, padding=k_size//2)
        self.sawf = SAWF(out_dim, 3)
        self.last_conv = nn.Conv2d(out_dim, out_dim, 1, 1)
 
    def forward(self, x):
        x = self.norm(x)
        cos_f = self.conv1(x) * torch.cos(self.conv2(x))
        cos_f = self.cos_conv(cos_f)
        x_f = self.conv3(x)
        sin_f = self.conv5(x) * torch.sin(self.conv4(x))
        sin_f = self.sin_conv(sin_f)
        attn = self.sawf(cos_f + x_f + sin_f)
        out = self.last_conv(attn[0]*cos_f + attn[1]*x_f + attn[2]*sin_f)

        return out