import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


def image2emb_naive(image, patch_size, weight):
    # image size [batchsize, c, h, w]
    """
        F.unfold 图像的卷积操作，但是只有卷，没有积，意思就是讲图像进行分块
    """
    patch = F.unfold(image, kernel_size=patch_size, stride=patch_size).transpose(-1, -2)
    patch_embedding = patch @ weight
    # torch.Size([1, 48, 4])
    # 1是bs，4是8*8分成4*4的大小，正好分成四块
    print(patch.shape)
    return patch_embedding


def image2emb_conv(image, kernel, stride):
    conv_output = F.conv2d(image, kernel, stride=stride)
    bs, oc, oh, ow = conv_output.shape
    patch_embedding = conv_output.reshape((bs, oc, oh * ow)).transpose(-1, -2)
    return patch_embedding


if __name__ == '__main__':
    bs, ic, h, w = 1, 3, 8, 8
    patch_size = 4
    # 这是表示分成4块，每块用一个model_dim长度的向量表示其位置信息, 因此model_dim也可以理解为输出通道数目，因为放在了第二维
    model_dim = 8
    max_num_token = 16
    num_classes = 8
    label = torch.randint(num_classes, (bs,))

    image = torch.randn(bs, ic, h, w)
    patch_depth = patch_size * patch_size * ic
    # patch_depth是卷积核的面积诚意输入通道数目
    weight = torch.randn(patch_depth, model_dim)
    patch_embedding_naive = image2emb_naive(image, patch_size, weight)
    print(patch_embedding_naive.shape)
    print(patch_embedding_naive)
    kernel = weight.transpose(0, 1).reshape((-1, ic, patch_size, patch_size))
    patch_embedding_conv = image2emb_conv(image, kernel=kernel, stride=patch_size)
    print(patch_embedding_conv)

    # step2 prepared cls token embedding
    cls_token_embedding = torch.randn(bs, 1, model_dim, requires_grad=True)
    # 在位置的维度，[1, 4, 8] 也就是在4的维度
    token_embedding = torch.cat([cls_token_embedding, patch_embedding_conv], dim=1)
    print(token_embedding.shape)
    # step3 add position embedding
    position_embedding_table = torch.randn(max_num_token, model_dim, requires_grad=True)
    seq_len = token_embedding.shape[1]
    """
    torch.tile(input, dims) → Tensor
        参数：
        input(Tensor) -要重复其元素的张量。
        dims(tuple) -每个维度的重复次数。
    """
    position_embedding = torch.tile(position_embedding_table[:seq_len], [token_embedding.shape[0], 1, 1])
    print(position_embedding_table[:seq_len].shape)
    print([token_embedding.shape[0], 1, 1])
    print(position_embedding.shape)
    token_embedding += position_embedding
    # step4 pass embedding to Transformer encoder layer
    encoder_layer = nn.TransformerEncoderLayer(d_model=model_dim, nhead=8)
    transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
    encoder_output = transformer_encoder(token_embedding)
    # step5 MLP Head
    cls_token_output = encoder_output[:, 0, :]
    linear_layer = nn.Linear(model_dim, num_classes)
    logits = linear_layer(cls_token_output)
    loss_fn = nn.CrossEntropyLoss()
    loss = loss_fn(logits, label)
    print(loss)