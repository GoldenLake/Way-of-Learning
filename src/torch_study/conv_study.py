"""
1. Normal Convolution
2. Depth-wise Convolution
又叫逐通道卷积,不改变通道
目的是为了减少计算量，提高计算速度。
3. Point-wise Convolution
又叫 1*1 卷积
目的是改变通道数
4. Depthwise Separable Convolution
又叫深度可分离卷积

"""
import torch.nn as nn
import torch

class DEPTHWISECONV(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DEPTHWISECONV, self).__init__()
        # 也相当于分组为1的分组卷积
        self.depth_conv = nn.Conv2d(in_channels=in_ch,
                                    out_channels=in_ch,
                                    kernel_size=3,
                                    stride=1,
                                    padding=1,
                                    groups=in_ch)
        self.point_conv = nn.Conv2d(in_channels=in_ch,
                                    out_channels=out_ch,
                                    kernel_size=1,
                                    stride=1,
                                    padding=0,
                                    groups=1)

    def forward(self, input):
        out = self.depth_conv(input)
        out = self.point_conv(out)
        return out


if __name__ == "__main__":
    net = DEPTHWISECONV(3, 4)
    input = torch.randn(4, 3, 128, 128)
    output = net(input)
    print(output.shape)