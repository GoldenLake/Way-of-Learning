import torch
import torch.nn as nn
import numpy as np


# m = nn.Dropout(p = 0.2)
# input = torch.randn(10, 10)
# out = m(input)
# print(out)


# use numpy to implement dropout
def train(rate, x, w1, b1, w2, b2):
    """
    :param rate: 表示淘汰rate比率的节点
    :param x:
    :param w1:
    :param b1:
    :param w2:
    :param b2:
    :return:
    """
    layer1 = np.maximum(0, np.dot(w1, x) + b1)
    mask1 = np.random.binomial(1, 1 - rate, layer1.shape)
    layer1 = mask1 * layer1
    layer2 = np.maximum(0, np.dot(w2, layer1) + b2)
    mask2 = np.random.binomial(1, 1 - rate, layer2.shape)
    layer2 = mask2 * layer2
    return layer2


def train_another(rate, x, w1, b1, w2, b2):
    """
    :param rate: 表示淘汰rate比率的节点
    :param x:
    :param w1:
    :param b1:
    :param w2:
    :param b2:
    :return:
    """
    layer1 = np.maximum(0, np.dot(w1, x) + b1)
    mask1 = np.random.binomial(1, 1 - rate, layer1.shape)
    layer1 = mask1 * layer1
    layer1 = layer1 / (1 - rate)
    layer2 = np.maximum(0, np.dot(w2, layer1) + b2)
    mask2 = np.random.binomial(1, 1 - rate, layer2.shape)
    layer2 = mask2 * layer2
    layer2 = layer2 / (1 - rate)
    return layer2


def test(rate, x, w1, b1, w2, b2):
    """
    :param rate:
    :param x:
    :param w1:
    :param b1:
    :param w2:
    :param b2:
    :return:
    """
    layer1 = np.maximum(0, np.dot(w1, x) + b1)
    # *(l - rate) 表示测试阶段的输出保持和训练阶段数据分布一致
    layer1 = layer1 * (1 - rate)
    layer2 = np.maximum(0, np.dot(w2, layer1) + b2)
    layer2 = layer2 * (1 - rate)
    return layer2


def test_another(rate, x, w1, b1, w2, b2):
    """
    :param rate:
    :param x:
    :param w1:
    :param b1:
    :param w2:
    :param b2:
    :return:
    """
    layer1 = np.maximum(0, np.dot(w1, x) + b1)
    layer2 = np.maximum(0, np.dot(w2, layer1) + b2)
    return layer2


if __name__ == "__main__":
    rate = 0.1
    train_out = train(rate, 1, 0.1, 0.2, 0.1, 0.2)
    test_out = test(rate, 1, 0.1, 0.2, 0.1, 0.2)
    print(train_out)
    print(test_out)

    # 这种实现方式要好，一方面使得测试和训练的数据分布一致，另一方面减少了测试（推理）的训练量
    train_out_another = train_another(rate, 1, 0.1, 0.2, 0.1, 0.2)
    test_out_another = test_another(rate, 1, 0.1, 0.2, 0.1, 0.2)
    print(train_out_another)
    print(test_out_another)