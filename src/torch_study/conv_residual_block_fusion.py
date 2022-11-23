import time
import torch
import torch.nn.functional as F
import torch.nn as nn

in_channels = 2
ou_channels = 2
kernal_size = 3
w = 9
h = 9
x = torch.ones(1, in_channels, w, h)
# 原生写法
t1 = time.time()
conv_2d = nn.Conv2d(in_channels, ou_channels, kernal_size, padding=1)
conv_2d_pointwise = nn.Conv2d(in_channels, ou_channels, 1)
result1 = conv_2d(x) + conv_2d_pointwise(x) + x
t1_f = time.time()

# print(result1)

# 算子融合
#  1）把point-wise卷积变成3*3卷积
#  2）再把输入x变成 3*3卷积，
#  3）把3个卷积写成一个卷积
#  1）
pointwise_to_conv_weight = F.pad(conv_2d_pointwise.weight, [1, 1, 1, 1])  # 两种写法都对
# pointwise_to_conv_weight = F.pad(conv_2d_pointwise.weight, [1,1,1,1,0,0,0,0]) # [2,2,1,1]->[2,2,3,3]
conv_2d_for_pointwise = nn.Conv2d(in_channels, ou_channels, kernal_size, padding=1)
conv_2d_for_pointwise.weight = nn.Parameter(pointwise_to_conv_weight)
conv_2d_for_pointwise.bias = conv_2d_pointwise.bias
# 2)
zeros = torch.unsqueeze(torch.zeros(kernal_size, kernal_size), 0)
stars = torch.unsqueeze(F.pad(torch.ones(1, 1), [1, 1, 1, 1]), 0)
stars_zeros = torch.unsqueeze(torch.cat([stars, zeros], 0), 0)  # 第一个通道
zeros_stars = torch.unsqueeze(torch.cat([stars, zeros], 0), 0)  # 第二个通道
identity_to_conv_weight = torch.cat([stars_zeros, zeros_stars], 0)
identity_to_conv_bias = torch.zeros([ou_channels])
conv_2d_for_identify = nn.Conv2d(in_channels, ou_channels, kernal_size, padding=1)
conv_2d_for_identify.weight = nn.Parameter(identity_to_conv_weight)
conv_2d_for_identify.bias = nn.Parameter(identity_to_conv_bias)

result2 = conv_2d(x) + conv_2d_for_pointwise(x) + conv_2d_for_identify(x)
print('原生方法和算子融合方法输出结果是否一致', torch.all(torch.isclose(result1, result2)))
# 3)
t2 = time.time()
conv_2d_for_fusion = nn.Conv2d(in_channels, ou_channels, kernal_size, padding=1)
conv_2d_for_fusion.weight = nn.Parameter(
    conv_2d.weight.data + conv_2d_for_pointwise.weight.data + conv_2d_for_identify.weight.data)
conv_2d_for_fusion.bias = nn.Parameter(
    conv_2d.bias.data + conv_2d_for_pointwise.bias.data + conv_2d_for_identify.bias.data)
t2_f = time.time()
result3 = conv_2d_for_fusion(x)
print('原生方法和算子融合方法输出结果是否一致', torch.all(torch.isclose(result2, result3)))
print('原生写法时间：', t1_f - t1)
print('算子融合写法时间：', t2_f - t2)
