import torch
import torch.nn.functional as F
import torch.nn as nn

"""
残差网络的算子融合
"""


# 原生残差结构
class ResidualBlock(nn.Module):
    def __init__(self, in_c, out_c, kernel_size):
        super(ResidualBlock, self).__init__()
        self.in_c = in_c
        self.out_c = out_c
        self.kernel_size = kernel_size
        self.conv2d = nn.Conv2d(in_c, out_c, kernel_size, padding=1)
        self.conv2d_pointwise = nn.Conv2d(in_c, out_c, kernel_size=1)

    def forward(self, x):
        x = self.conv2d(x) + self.conv2d_pointwise(x) + x
        return x

if __name__ == "__main__":
    in_c = 2
    out_c = 2
    kernel_size = 3
    w = 9
    h = 9
    # input
    x = torch.ones(1, in_c, w, h)
    rb = ResidualBlock(in_c, out_c, kernel_size)
    out_rb = rb(x)
    print(out_rb.shape)
