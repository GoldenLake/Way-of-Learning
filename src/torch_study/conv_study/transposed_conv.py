import torch
import torch.nn as nn
import torch.nn.functional as F

"""
nn.Unfold()函数
    描述：pytorch中的nn.Unfold()函数，在图像处理领域，经常需要用到卷积操作，但是有时我们只需要在图片上进行滑动的窗口操作，
    将图片切割成patch，而不需要进行卷积核和图片值的卷积乘法操作。这是就需要用到nn.Unfold()函数，
    该函数是从一个batch图片中，提取出滑动的局部区域块，也就是卷积操作中的提取kernel filter对应的滑动窗口。
    torch.nn.Unfold(kernel_size,dilation=1,paddding=0,stride=1)
    该函数的输入是（bs,c,h,w),其中bs为batch-size,C是channel的个数。
    而该函数的输出是
    （bs,Cxkernel_size[0]xkernel_size[1],L)
    其中L是特征图或者图片的尺寸根据kernel_size的长宽滑动裁剪后得到的多个patch的数量。
    
    nn.Fold(）函数 该函数是nn.Unfold()函数的逆操作。
"""
# batches_img = torch.rand(1, 2, 4, 4)  # 模拟图片数据（bs,2,4,4），通道数C为2
# print("batches_img:\n", batches_img)
#
# nn_Unfold = nn.Unfold(kernel_size=(2, 2), dilation=1, padding=0, stride=2)
# patche_img = nn_Unfold(batches_img)
# print("patche_img.shape:", patche_img.shape)
# print("patch_img:\n", patche_img)
#
# fold = torch.nn.Fold(output_size=(4, 4), kernel_size=(2, 2), stride=2)
# inputs_restore = fold(patche_img)
# print(inputs_restore)
# print(inputs_restore.size())

"""
    转置卷积（Transpose Convolution）反卷积
    主要作用就是起到上采样的作用
    输入图像大小W*W
    输出图像大小N*N
    N = (W - F + 2P) / S + 1
"""


def get_kernel_matrix(kernel, input_size):
    """
    实现转置卷积的kernel
    :param kernel:
    :param input_size:
    :return:
    """
    kernel_h, kernel_w = kernel.shape
    input_h, input_w = input_size
    num_out_feat_map = (input_h - kernel_h + 1) * (input_w - kernel_w + 1)
    result = torch.zeros((num_out_feat_map, input_h * input_w))  # 初始化结果矩阵, 输出特征图元素个数 * 输入特征图元素个数
    count = 0
    for i in range(0, input_h - kernel_h + 1, 1):
        for j in range(0, input_w - kernel_w + 1, 1):
            padded_kernel = F.pad(kernel, (i, input_h - kernel_h - i, j, input_w - kernel_w - j))  # 填充成跟输入特征图一样的大小
            result[count] = padded_kernel.flatten()
            count += 1
    return result


if __name__ == '__main__':
    kernel = torch.randn(3, 3)
    input = torch.randn(4, 4)
    kernel_matrix = get_kernel_matrix(kernel, input.shape)
    print(kernel_matrix.shape)
    mm_conv2d_output = kernel_matrix @ input.reshape((-1, 1))
    print(mm_conv2d_output.shape)
    pytorch_conv2d_output = F.conv2d(input.unsqueeze(0).unsqueeze(0), kernel.unsqueeze(0).unsqueeze(0))
    print(pytorch_conv2d_output.shape)
    print("kernel.unsqueeze(0).unsqueeze(0)", kernel.unsqueeze(0).unsqueeze(0))
    # 测试 验证二维转置卷积
    mm_transposed_conv2d_output = kernel_matrix.transpose(-1, -2) @ mm_conv2d_output
    pytorch_transposed_conv2d_output = F.conv_transpose2d(pytorch_conv2d_output, kernel.unsqueeze(0).unsqueeze(0))
    print("mm_transposed_conv2d_output", mm_transposed_conv2d_output.reshape((-1, 4)))
    print("pytorch_transposed_conv2d_output", pytorch_transposed_conv2d_output)