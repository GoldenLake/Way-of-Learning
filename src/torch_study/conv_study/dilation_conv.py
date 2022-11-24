import torch
import torch.nn as nn
import torch.nn.functional as F

"""
    空洞卷积 也叫膨胀卷积或者扩张卷积
    a[0:5:2,0:5:2]
    表示索引从0到5（不包含5），步长为2取一个数字
    空洞卷积主要有三个作用：
        1.扩大感受野。但需要明确一点，池化也可以扩大感受野，但空间分辨率降低了，相比之下，
        空洞卷积可以在扩大感受野的同时不丢失分辨率，且保持像素的相对空间位置不变。
        简单而言，空洞卷积可以同时控制感受野和分辨率。

        2.获取多尺度上下文信息。当多个带有不同dilation rate的空洞卷积核叠加时，
        不同的感受野会带来多尺度信息，这对于分割任务是非常重要的。

        3.可以降低计算量，不需要引入额外的参数，
        如上图空洞卷积示意图所示，实际卷积时只有带有红点的元素真正进行计算。
    
    如果加上了dilation 公式应该是output = (向下取整)((input + 2*padding -dilation*(kernel_size-1) -1)/stride) + 1
    如果使得输入和输出的维度一样，dilation与padding设置成一样
"""


input = torch.randn((1, 3, 7, 7))
conv = nn.Conv2d(3, 3, kernel_size=(3, 3), dilation=3, padding=3)
out = conv(input)
print(out.shape)
