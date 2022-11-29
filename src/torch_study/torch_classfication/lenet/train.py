import torch
import torchvision
import torch.nn as nn
from lenet_study import LeNet
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def main():
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    train_set = torchvision.datasets.CIFAR10(root="data", train=True,
                                             download=False, transform=transform)
    train_loader = DataLoader(train_set, batch_size=8192 * 2,
                              shuffle=True, num_workers=0)

    val_set = torchvision.datasets.CIFAR10(root='data', train=False,
                                           download=True, transform=transform)
    val_loader = DataLoader(val_set, batch_size=8192 * 2,
                            shuffle=False, num_workers=0)
    """
        iter() 函数用来生成迭代器。
        iter(object[, sentinel]) object -- 支持迭代的集合对象。
        next(iterable[, default]) 返回迭代器的下一个项目。
    """
    val_data_iter = iter(val_loader)
    val_image, val_label = next(val_data_iter)
    val_image, val_label = val_image.to(device), val_label.to(device)
    net = LeNet().to(device)
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    save_path = './Lenet.pth'
    history_acc = 0.0
    for epoch in range(50):
        running_loss = 0.0
        for step, (inputs, labels) in enumerate(train_loader, start=0):
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            """
            loss.item()
            1.item（）取出张量具体位置的元素元素值
            2.并且返回的是该位置元素值的高精度值
            3.保持原元素类型不变；必须指定位置
            """
            running_loss += loss.item()
            if step % 2 == 0:
                with torch.no_grad():
                    outputs = net(val_image)  # [batch, 10]
                    # 得到每个batch里最大值的索引
                    predict_y = torch.max(outputs, dim=1)[1]
                    # 计算acc
                    print(val_label.size(0))
                    acc = torch.eq(predict_y, val_label).sum().item() / val_label.size(0)
                    print('[%d, %5d] train_loss: %.3f  test_accuracy: %.3f' %
                          (epoch + 1, step + 1, running_loss / 500, acc))
                    if acc > history_acc:
                        history_acc = acc
                        torch.save(net.state_dict(), save_path)
                    running_loss = 0.0

if __name__ == '__main__':
    main()
