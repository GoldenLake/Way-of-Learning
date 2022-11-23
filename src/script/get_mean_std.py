import glob
import numpy as np
from PIL import Image

if __name__ == '__main__':
    # 图像所放的路径
    path = "b2"
    res = []
    for img_path in glob.glob(path + "\\*.tif"):
        im = Image.open(img_path)
        image_array = np.array(im)
        # 图像shape
        # print("image_array: ", image_array.shape)
        # tiff图像像素范围是0-10000
        image_array = image_array / 10000.
        res.append(image_array)
    # [bs, h, w]  (10, 512, 512)
    # print(np.shape(res))
    # 文件下所有图片一起统计均值
    # tif是单通道图像，如果是RGB图像， axis=(0, 1, 2)表示统计每个通道下所有的图像的均值和标准差
    mean = np.mean(res, axis=(0, 1, 2))
    print(mean)
    std = np.std(res, axis=(0, 1, 2))
    print(std)
