import os
os.chdir(r"D:\5-旋转设备无忧运行系统\2-深度学习轴承故障诊断\00 AIMS轴承故障分类\00data\02 AIMS_4712_DE_5c_4000hz\AllData")
files = os.listdir()
path1 = r"D:\5-旋转设备无忧运行系统\2-深度学习轴承故障诊断\00 AIMS轴承故障分类\00data\02 AIMS_4712_DE_5c_4000hz\test"
for file in files:
    path2 = os.path.join(path1,file)
    if not os.path.exists(path2):
          os.makedirs(path2)