import os
import glob
import shutil


# glob用于正则的遍历文件
# shutil用于文件的复制
for img_path in glob.glob("train2017" + "/*"):
    for img_name in glob.glob(img_path + "/*.jpeg"):
        # print("JPEGImages\\" + os.path.basename(img_name))
        shutil.copy(img_name, "JPEGImages\\" + os.path.basename(img_name))

index = 0
lines = []
with open("ImageSets\\Main\\trainval.txt", mode="r", encoding="utf-8") as f:
    lines = f.readlines()
# # print(line)
f.close()
with open("ImageSets\\Main\\train.txt", mode="w", encoding="utf-8") as f1:
    for line in lines:
        if index % 5 != 0:
            f1.write(line)
        index += 1