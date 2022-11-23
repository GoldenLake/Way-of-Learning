## 1. del

VGG16 继承VGG类的时候，del可以用于删除对象中的变量
~~~python
def VGG16(pretrained, in_channels=3, **kwargs):
    model = VGG(make_layers(cfgs["D"], batch_norm=False, in_channels=in_channels), **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url("https://download.pytorch.org/models/vgg16-397923af.pth",
                                              model_dir="./model_data")
        model.load_state_dict(state_dict)

    del model.avgpool
    del model.classifier
    return model
~~~

## 2.上采样
~~~python
    """
        torch.Size([1, 3, 128, 128])
        torch.Size([1, 3, 192, 256])
        torch.Size([1, 3, 200, 300])
    """
    input = torch.randn(1, 3, 64, 64)
    up = nn.UpsamplingBilinear2d(scale_factor=2)
    up1 = nn.UpsamplingBilinear2d(scale_factor=(3, 4))
    up2 = torch.nn.UpsamplingBilinear2d(size=(200, 300))
    output = up(input)
    output1 = up1(input)
    output2 = up2(input)
    print(output.shape)
    print(output1.shape)
    print(output2.shape)
~~~

## 3. Unet 最后的1*1 Conv
~~~python
"""
out_filters[0] = 64
增强网络的输出结果是 BatchSize * 64(c) * h * w
最后做预测，就是进行一次1*1的卷积，把通道改为类别数
"""
self.final = nn.Conv2d(out_filters[0], num_classes, 1)

~~~
